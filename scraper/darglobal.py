"""Scrape public project / insight pages from DarGlobal (darglobal.co.uk).

URLs come from the public sitemap referenced in robots.txt. Pages are read with plain HTTP by default;
responses that are a bot-protection (Incapsula) challenge are skipped and reported, never bypassed.
Optional --browser mode renders pages in a normal Playwright Chrome window (no anti-detection tricks).

Usage:  python scraper/darglobal.py [--browser] [--blog 40] [--delay 2]
Output: data/darglobal_pages.jsonl
"""
import argparse, json, re, time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "darglobal_pages.jsonl"
SITEMAP = "https://darglobal.co.uk/sitemap.xml"
UA = {"User-Agent": "Mozilla/5.0 (compatible; property-chatbot-research/1.0)"}
SKIP = re.compile(r"/(thank-you|success|pay-online|privacy-policy|terms|cookie|landing-page|win-a-trip)", re.I)


def sitemap_urls(n_blog):
    xml = requests.get(SITEMAP, headers=UA, timeout=30).text
    urls = [u.rstrip("/") for u in re.findall(r"<loc>([^<]+)</loc>", xml)]
    core = [u for u in urls if not re.search(r"/(blog|press)/", u) and not SKIP.search(u)]
    blog = [u for u in urls if re.search(r"/(blog|press|insights)/.+", u)][:n_blog]
    return list(dict.fromkeys(core + blog))


def blocked(html):
    return "_Incapsula_Resource" in html and len(html) < 5000


def parse(url, html):
    s = BeautifulSoup(html, "lxml")
    for t in s(["script", "style", "noscript", "svg", "header", "footer", "nav", "form", "iframe"]):
        t.decompose()
    title = (s.title.string if s.title else "") or ""
    desc = s.find("meta", attrs={"name": "description"})
    main = s.find("main") or s.body or s
    lines, seen = [], set()
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "span", "div"]):
        if el.find(["p", "div", "li", "h1", "h2", "h3"]):
            continue  # leaf-ish text only
        t = " ".join(el.get_text(" ").split())
        if 3 < len(t) < 600 and t not in seen:
            seen.add(t)
            lines.append(t)
    text = ((desc["content"] + "\n") if desc and desc.get("content") else "") + "\n".join(lines)
    kind = "article" if re.search(r"/(blog|press|insights)/", url) else "project/page"
    return {"source": "darglobal", "url": url, "kind": kind,
            "title": title.replace("| Luxury Real Estate, DarGlobal", "").strip(), "text": text[:12000]}


def fetch_plain(urls, delay):
    for u in urls:
        try:
            r = requests.get(u, headers=UA, timeout=40)
            yield u, r.status_code, r.text
        except Exception as e:
            yield u, 0, str(e)
        time.sleep(delay)


def fetch_browser(urls, delay):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=False)
        pg = b.new_page()
        for u in urls:
            try:
                resp = pg.goto(u, wait_until="domcontentloaded", timeout=60000)
                pg.wait_for_timeout(5000)  # let the page (and any human check you complete) finish loading
                yield u, resp.status if resp else 200, pg.content()
            except Exception as e:
                print(f"error {u}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
                yield u, 0, str(e)
            time.sleep(delay)
        b.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--blog", type=int, default=40, help="how many blog/press/insight posts to include")
    ap.add_argument("--delay", type=float, default=2.0)
    a = ap.parse_args()
    urls = sitemap_urls(a.blog)
    print(f"{len(urls)} DarGlobal URLs", flush=True)
    ok = bad = 0
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for u, code, html in (fetch_browser if a.browser else fetch_plain)(urls, a.delay):
            if code != 200 or blocked(html):
                bad += 1
                print(f"skip {code} {'(bot challenge)' if blocked(html) else ''} {u}", flush=True)
                if code == 0 and bad >= 3 and ok == 0:
                    print("stopping: first pages all errored - see the error lines above", flush=True)
                    break
                continue
            rec = parse(u, html)
            if len(rec["text"]) < 200:
                bad += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
            print(f"ok  {u} ({len(rec['text'])} chars)", flush=True)
    print(f"done: {ok} pages saved, {bad} skipped -> {OUT}")


if __name__ == "__main__":
    main()
