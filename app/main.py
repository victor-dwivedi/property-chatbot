"""Real-estate RAG chatbot over scraped DarGlobal + Wasalt data, answered by a free OpenRouter model."""
import asyncio, json, os, re, time
from collections import Counter
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

def _load_dotenv():
    """Load KEY=VALUE lines from a local .env (for running outside Docker); real env vars win."""
    f = Path(__file__).resolve().parents[1] / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
# "openrouter/free" routes to any currently-available free model; the rest are fallbacks.
MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-31b-it:free,baidu/ernie-5.5-flash-preview:free,nvidia/nemotron-3.5-lightning:free,google/gemma-4-26b-a4b-it:free,inclusionai/ling-3.2-flash:free,openrouter/free",
).split(",") if m.strip()]
TOP_K = int(os.getenv("TOP_K", "12"))

# ---------------------------------------------------------------- data + index
TOKEN = re.compile(r"[a-z0-9؀-ۿ]+")


def tok(s):
    return TOKEN.findall((s or "").lower())


def money(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return f"{v:,.0f}"


def rent_info(r):
    """Wasalt rent frequency comes as {'yearly': {'amount': .., 'freq': 'yearly', 'default_freq': True}, ...}."""
    rf = r.get("rent_frequency")
    if isinstance(rf, dict) and rf:
        opts = [v for v in rf.values() if isinstance(v, dict)]
        best = next((v for v in opts if v.get("default_freq")), opts[0] if opts else None)
        if best:
            return best.get("amount") or r.get("price"), "per " + {"yearly": "year", "monthly": "month", "daily": "day"}.get(best.get("freq"), best.get("freq") or "period")
    if isinstance(rf, str) and rf:
        return r.get("price"), rf
    return r.get("price"), None


def listing_text(r):
    price, per = rent_info(r)
    parts = [
        f"{r.get('title')} — {r.get('sub_type')} for {r.get('purpose')} in {r.get('district')}, {r.get('city')}"
        + (f" ({r['city_slug'].replace('-', ' ').title()})" if r.get("city_slug") else ""),
        f"Price: {money(price)} {r.get('currency')}" + (f" {per}" if per else "") if price else "Price: on request",
    ]
    for k, lbl in [("bedrooms", "Bedrooms"), ("bathrooms", "Bathrooms"), ("built_up_area_sqm", "Built-up area (sqm)"),
                   ("land_area_sqm", "Land area (sqm)"), ("age", "Age"), ("furnishing", "Furnishing"), ("main_type", "Category")]:
        if r.get(k) not in (None, "", []):
            parts.append(f"{lbl}: {r[k]}")
    if r.get("amenities"):
        parts.append("Amenities: " + ", ".join(r["amenities"]))
    return "\n".join(parts)


def fetch_private_data():
    """On the hosted Space the scraped data lives in a PRIVATE HF dataset (not redistributed publicly)."""
    repo = os.getenv("DATA_REPO")
    if repo and not any(DATA_DIR.glob("*.jsonl")):
        from huggingface_hub import snapshot_download
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, repo_type="dataset", token=os.getenv("HF_TOKEN"),
                          local_dir=str(DATA_DIR), allow_patterns=["*.jsonl"])


def load():
    fetch_private_data()
    docs = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source") == "wasalt":
                r["_text"] = listing_text(r)
            else:  # darglobal pages / projects / articles
                r["_text"] = f"{r.get('title','')}\n{r.get('text','')}".strip()
            docs.append(r)
    return docs


DOCS = load()
BM25 = BM25Okapi([tok(d["_text"] + " " + d.get("source", "")) for d in DOCS]) if DOCS else None


def stats():
    w = [d for d in DOCS if d.get("source") == "wasalt"]
    dg = [d for d in DOCS if d.get("source") == "darglobal"]
    city = Counter(d.get("city") for d in w).most_common(12)
    typ = Counter(d.get("sub_type") for d in w).most_common(10)
    return (f"Knowledge base: {len(dg)} DarGlobal pages and {len(w)} Wasalt listings.\n"
            f"Wasalt listings by city: {', '.join(f'{c} ({n})' for c, n in city)}.\n"
            f"Wasalt listings by type: {', '.join(f'{t} ({n})' for t, n in typ)}.")


STATS = stats() if DOCS else "Knowledge base is empty."

# ------------------------------------------------------------- retrieval
CITY_WORDS = {c.lower() for c in (d.get("city") for d in DOCS) if c}


def parse_budget(q):
    m = re.search(r"(under|below|less than|max(?:imum)?|up to|<)\s*(?:sar|sr|aed|usd|\$)?\s*([\d.,]+)\s*(k|m|million|thousand)?", q, re.I)
    if not m:
        return None
    v = float(m.group(2).replace(",", ""))
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get((m.group(3) or "").lower(), 1)
    return v * mult


AR2EN = {
    "الرياض": "riyadh", "جدة": "jeddah", "جده": "jeddah", "الدمام": "dammam", "الخبر": "khobar", "مكة": "makkah",
    "المدينة": "madinah", "الظهران": "dhahran", "الطائف": "taif", "أبها": "abha", "ابها": "abha", "بريدة": "buraidah",
    "تبوك": "tabuk", "حائل": "hail", "جازان": "jazan", "الأحساء": "ahsa", "الاحساء": "ahsa", "الجبيل": "jubail",
    "ينبع": "yanbu", "نجران": "najran", "الخرج": "kharj", "القطيف": "qatif", "خميس مشيط": "khamis mushait",
    "شقق": "apartment", "شقة": "apartment", "فلل": "villa", "فيلا": "villa", "فيلل": "villa", "أرض": "land",
    "ارض": "land", "أراضي": "land", "اراضي": "land", "عمارة": "building", "عماره": "building", "مستودع": "warehouse",
    "معرض": "showroom", "مكتب": "office", "دور": "floor", "استراحة": "rest house", "شاليه": "chalet",
    "للإيجار": "rent", "للايجار": "rent", "إيجار": "rent", "ايجار": "rent", "للبيع": "sale", "بيع": "sale",
    "شراء": "buy", "غرف": "bedroom", "غرفة": "bedroom", "مليون": "million", "ألف": "thousand", "الف": "thousand",
    "أقل من": "under", "اقل من": "under", "دار جلوبال": "darglobal", "مشروع": "project", "مشاريع": "project",
    "ترامب": "trump", "عمان": "oman", "دبي": "dubai", "قطر": "qatar", "إسبانيا": "spain", "لندن": "london",
}


def normalize(q):
    """Append English equivalents of Arabic keywords so Arabic questions match the English listing text."""
    extra = [en for ar, en in AR2EN.items() if ar in q]
    return f"{q} {' '.join(extra)}" if extra else q


def retrieve(q, k=TOP_K):
    if not BM25:
        return []
    q = normalize(q)
    ql = q.lower()
    scores = BM25.get_scores(tok(q))
    want_dg = any(w in ql for w in ["darglobal", "dar global", "trump", "project", "developer", "missoni", "aida", "w residences"])
    want_ws = "wasalt" in ql
    purpose = "rent" if re.search(r"\brent|lease|rental", ql) else ("sale" if re.search(r"\b(buy|sale|purchase)", ql) else None)
    beds = re.search(r"(\d+)\s*(?:bed|br|bhk|bedroom)", ql)
    budget = parse_budget(q)
    ranked = []
    for i, d in enumerate(DOCS):
        s = scores[i]
        if d.get("source") == "darglobal":
            s += 2.0 if want_dg else 0
            s -= 3.0 if want_ws else 0
        else:
            if want_dg and not want_ws:
                s -= 2.0
            if (d.get("city") and d["city"].lower() in ql) or (d.get("city_slug") and d["city_slug"].replace("-", " ") in ql):
                s += 3
            st = (d.get("sub_type") or "").lower()
            if st and re.search(r"\b" + re.escape(st.rstrip("s")) + r"(e?s)?\b", ql):
                s += 4
            if purpose and d.get("purpose") == purpose:
                s += 1.5
            if beds and str(d.get("bedrooms")) == beds.group(1):
                s += 2
            if budget and d.get("price"):
                try:
                    s += 2 if float(d["price"]) <= budget else -4
                except ValueError:
                    pass
        ranked.append((s, i))
    ranked.sort(reverse=True)
    out, seen = [], set()
    for s, i in ranked:
        if s <= 0 or len(out) >= k:
            break
        key = DOCS[i]["_text"]
        if key in seen:  # identical listings re-posted under different ids
            continue
        seen.add(key)
        out.append(DOCS[i])
    # if the user asked for a property type that the top results lack (e.g. no villas in that city),
    # add the best-scoring listings of that type from anywhere so the bot can offer alternatives
    types = {(d.get("sub_type") or "").lower() for d in DOCS if d.get("sub_type")}
    wanted = [t for t in types if re.search(r"\b" + re.escape(t.rstrip("s")) + r"(e?s)?\b", ql)]
    if wanted and not any((d.get("sub_type") or "").lower() in wanted for d in out):
        extra = [DOCS[i] for s, i in ranked if (DOCS[i].get("sub_type") or "").lower() in wanted
                 and (not purpose or DOCS[i].get("purpose") == purpose)][:4]
        out = out[: max(0, k - len(extra))] + extra
    return out


# ------------------------------------------------------------- LLM
SYSTEM = """You are a helpful real-estate assistant. You answer ONLY from the CONTEXT provided, which was scraped
from public pages of DarGlobal (darglobal.co.uk — international luxury developer) and Wasalt (wasalt.sa — Saudi property portal).
Rules:
- Use only facts in CONTEXT; if the answer isn't there, say so and suggest checking the live website.
- If nothing matches exactly (e.g. no villas in the requested city), say so clearly, then offer the closest
  matches from CONTEXT (same type in other cities, or other types in that city) and label them as alternatives.
- When recommending listings, give title, location, price, key specs and the URL as a markdown link.
- Cite the source on every factual answer, e.g. "Source: Wasalt.sa" or "Source: DarGlobal".
- Prices/availability may have changed since scraping — remind the user to verify on the live site.
- The UI automatically shows a photo card (price, location, specs) for every listing/project URL you link,
  so keep the text short: one intro line, then up to 6 bullets of the form
  "**Title** — District, City — Price — key specs — [View](URL)". Do NOT use tables.
- Be concise; use bullet points for lists. Reply in the user's language (English or Arabic)."""


async def complete(messages):
    if not OPENROUTER_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY is not configured on the server.")
    last_err = None
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(3):  # free models are often briefly rate-limited: cycle the list, then back off
            for model in MODELS:
                try:
                    r = await c.post("https://openrouter.ai/api/v1/chat/completions",
                                     headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                                              "HTTP-Referer": os.getenv("APP_URL", "https://localhost"),
                                              "X-Title": "Property Chatbot"},
                                     json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1600})
                    j = r.json()
                    if r.status_code == 200 and j.get("choices"):
                        txt = (j["choices"][0]["message"].get("content") or "").strip()
                        if txt:
                            return txt, j.get("model", model)
                    last_err = f"{model}: {r.status_code} {str(j.get('error', j))[:200]}"
                except Exception as e:
                    last_err = f"{model}: {e}"
            await asyncio.sleep(3 * (attempt + 1))
    raise HTTPException(502, f"All free models failed. Last error: {last_err}")


# ------------------------------------------------------------- API
app = FastAPI(title="DarGlobal & Wasalt Property Chatbot")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    message: str
    history: list[Msg] = []


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "docs": len(DOCS), "models": MODELS, "key_configured": bool(OPENROUTER_KEY)}


@app.get("/api/stats")
def api_stats():
    return {"summary": STATS,
            "sources": Counter(d.get("source") for d in DOCS)}


@app.post("/api/chat")
async def chat(req: ChatReq):
    q = req.message.strip()[:2000]
    if not q:
        raise HTTPException(400, "Empty message")
    # include the previous user turn so follow-ups ("cheaper ones?") keep their context
    prev = " ".join(m.content for m in req.history[-2:] if m.role == "user")
    hits = retrieve(f"{prev} {q}".strip())
    ctx = "\n\n".join(f"[{i+1}] ({h['source']}) URL: {h.get('url')}\n{h['_text'][:1500]}" for i, h in enumerate(hits))
    messages = [{"role": "system", "content": f"{SYSTEM}\n\n{STATS}"}]
    messages += [{"role": m.role, "content": m.content[:3000]} for m in req.history[-6:] if m.role in ("user", "assistant")]
    messages.append({"role": "user", "content": f"CONTEXT:\n{ctx or '(no matching documents)'}\n\nQUESTION: {q}"})
    t = time.time()
    answer, model = await complete(messages)
    return {"answer": answer, "model": model, "latency_s": round(time.time() - t, 2),
            "cards": cards_for(answer, hits)}


def card(d):
    if d.get("source") == "wasalt":
        price, per = rent_info(d)
        return {"source": "wasalt", "url": d.get("url"), "title": d.get("title"),
                "image": (d.get("images") or [None])[0], "images": d.get("images") or [],
                "price": f"{money(price)} {d.get('currency') or 'SAR'}" + (f" {per}" if per else "") if price else "Price on request",
                "location": ", ".join(x for x in [d.get("district"), (d.get("city_slug") or d.get("city") or "").replace("-", " ").title()] if x),
                "type": d.get("sub_type"), "purpose": d.get("purpose"),
                "beds": d.get("bedrooms"), "baths": d.get("bathrooms"),
                "area": d.get("built_up_area_sqm") or d.get("land_area_sqm")}
    return {"source": "darglobal", "url": d.get("url"), "title": d.get("title") or "DarGlobal",
            "image": (d.get("images") or [None])[0], "images": d.get("images") or [],
            "summary": (d.get("text") or "")[:160], "type": d.get("kind")}


def cards_for(answer, hits, limit=8):
    """Cards for the documents the model actually referenced (by URL), in the order it mentioned them."""
    pos = {}
    for h in hits:
        u = (h.get("url") or "").rstrip("/")
        i = answer.find(u) if u else -1
        if i >= 0 and u not in pos:
            pos[u] = (i, h)
    chosen = [h for _, h in sorted(pos.values(), key=lambda x: x[0])]
    if not chosen:  # model cited nothing by link: show top listing hits only when it clearly listed properties
        chosen = [h for h in hits if h.get("source") == "wasalt"][:4] if re.search(r"SAR|ريال", answer) else []
    return [card(h) for h in chosen[:limit]]
