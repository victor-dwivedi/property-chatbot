"""Scrape public property listings from Wasalt.sa.

Respects https://wasalt.sa/llms.txt:
  * only public listing content (category pages listed in the public sitemap)
  * moderate crawl rate (fixed delay between requests)
  * NO contact details / personal data are stored (phones, WhatsApp, owner/agent names,
    licence numbers are dropped)
  * every record carries a source URL for "Source: Wasalt.sa" attribution

Usage:  python scraper/wasalt.py [--max-pages 200] [--delay 2]
Output: data/wasalt_listings.jsonl
"""
import argparse, gzip, io, json, re, sys, time
from pathlib import Path

from curl_cffi import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "wasalt_listings.jsonl"
SITEMAP = "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz"
CITIES = ["riyadh", "jeddah", "dammam", "al-khobar", "makkah", "madinah", "dhahran",
          "taif", "abha", "khamis-mushait", "buraidah", "tabuk", "hail", "jazan",
          "al-ahsa", "jubail", "yanbu", "najran", "al-kharj", "qatif"]
NEXT_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def get(url):
    return requests.get(url, impersonate="chrome", timeout=40)


def category_urls():
    raw = get(SITEMAP).content
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    locs = re.findall(r"<loc>([^<]+)</loc>", raw.decode("utf-8", "ignore"))
    city_level = []
    for u in locs:
        slug = u.rsplit("/", 1)[-1]
        m = re.match(r"(.+)-for-(sale|rent)-in-(.+)$", slug)
        if m and m.group(3).lower() in CITIES:  # city-level only, no district pages
            city_level.append(u)
    # order: cycle through cities so a partial run still covers every city
    city_level.sort(key=lambda u: (CITIES.index(re.search(r"-in-(.+)$", u).group(1).lower()), u))
    by_city = {}
    for u in city_level:
        by_city.setdefault(re.search(r"-in-(.+)$", u).group(1).lower(), []).append(u)
    ordered, i = [], 0
    while any(by_city.values()):
        for c in CITIES:
            if by_city.get(c):
                ordered.append(by_city[c].pop(0))
    return ordered


def attr(p, key):
    for a in p.get("attributes") or []:
        if isinstance(a, dict) and a.get("key") == key:
            return a.get("value")
    return None


def clean(p, page_url):
    info = p.get("propertyInfo") if isinstance(p.get("propertyInfo"), dict) else {}
    pf = info.get("propertyFor")
    price = info.get("salePrice") if pf == "sale" else (info.get("expectedRent") or info.get("salePrice"))
    slug = info.get("slug") or str(p.get("id"))
    return {
        "source": "wasalt",
        "id": p.get("id"),
        "url": f"https://wasalt.sa/en/property/{pf}/{slug}",
        "title": info.get("title"),
        "purpose": pf,                                  # sale / rent
        "main_type": info.get("propertyMainType"),      # Residential / Commercial
        "sub_type": info.get("propertySubType"),        # Villa, Apartment, Land ...
        "city": info.get("city"),
        "district": info.get("district") or info.get("zone"),
        "address": info.get("address"),
        "price": price,
        "currency": info.get("currencyType") or "SAR",
        "rent_frequency": info.get("rentFreq") or None,
        "bedrooms": attr(p, "noOfBedrooms"),
        "bathrooms": attr(p, "noOfBathrooms"),
        "built_up_area_sqm": attr(p, "builtUpArea"),
        "land_area_sqm": attr(p, "landArea") or p.get("floorSize"),
        "age": attr(p, "completionYear"),
        "furnishing": info.get("furnishingType"),
        "amenities": [a.get("name") if isinstance(a, dict) else str(a) for a in (p.get("featureAmenities") or []) if a],
        "verified": p.get("isVerified"),
        "published_at": p.get("publishedAt"),
        "city_slug": (re.search(r"-in-([a-z-]+)$", page_url) or [None, None])[1],  # e.g. "taif"
        "found_on": page_url,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=220)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true", help="keep existing data, skip pages already scraped")
    a = ap.parse_args()
    urls = category_urls()[: a.max_pages]
    print(f"{len(urls)} category pages to fetch", flush=True)
    seen, n, done_pages = set(), 0, set()
    OUT.parent.mkdir(exist_ok=True)
    if a.resume and OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            seen.add(r.get("id")); done_pages.add(r.get("found_on"))
        print(f"resuming: {len(seen)} listings already saved, {len(done_pages)} pages done", flush=True)
    with OUT.open("a" if a.resume else "w", encoding="utf-8") as f:
        for i, u in enumerate(urls, 1):
            if u in done_pages:
                continue
            status, props = "-", []
            try:
                r = get(u)
                status = r.status_code
                m = NEXT_RE.search(r.text)
                sr = json.loads(m.group(1))["props"]["pageProps"].get("searchResult") if m else None
                if isinstance(sr, dict):
                    props = sr.get("properties") or []
                elif isinstance(sr, list):
                    props = sr
            except Exception as e:  # keep going on individual failures
                print(f"[{i}] ERR {u}: {e}", flush=True)
            new = 0
            for p in props:
                if not isinstance(p, dict) or p.get("id") in seen:
                    continue
                try:
                    rec = clean(p, u)
                except Exception as e:
                    print(f"[{i}] skip listing {p.get('id')}: {e}", flush=True)
                    continue
                seen.add(p.get("id"))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new += 1
            f.flush()
            n += new
            print(f"[{i}/{len(urls)}] {status} +{new} (total {len(seen)}) {u}", flush=True)
            time.sleep(a.delay)
    print(f"done: {n} listings -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
