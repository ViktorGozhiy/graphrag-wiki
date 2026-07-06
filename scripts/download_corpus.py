# ABOUTME: Downloads a connected corpus of Wikipedia articles for GraphRAG experiments.
# ABOUTME: Seeds from one article, follows in-text links, saves plain text + metadata as JSONL.

import json
import time
from pathlib import Path

import requests

from graphrag_wiki.wikitext import clean_extract

API = "https://en.wikipedia.org/w/api.php"
UA = "graphrag-wiki/0.1 (educational GraphRAG learning project)"

SEED = "Roman Empire"
TARGET_COUNT = 40
OUT_FILE = Path("data/raw/corpus.jsonl")

# Meta/list pages are entity-poor and useless for a knowledge graph.
SKIP_PREFIXES = ("List of", "Timeline of", "Outline of", "Index of", "Bibliography of")

session = requests.Session()
session.headers.update({"User-Agent": UA})


def api_get(params):
    """Call the MediaWiki API with retries, returning parsed JSON."""
    params = {**params, "format": "json", "formatversion": "2"}
    last_error = None
    for attempt in range(3):
        try:
            response = session.get(API, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def seed_links(seed):
    """Return the seed article's mainspace links in order of appearance."""
    data = api_get({"action": "parse", "page": seed, "prop": "links"})
    titles = []
    for link in data["parse"]["links"]:
        title = link["title"]
        if link["ns"] != 0 or not link.get("exists"):
            continue
        if "(disambiguation)" in title or title.startswith(SKIP_PREFIXES):
            continue
        titles.append(title)
    return titles


def fetch_article(title):
    """Fetch one article's plain text and categories, following redirects."""
    data = api_get({
        "action": "query",
        "prop": "extracts|categories",
        "titles": title,
        "explaintext": 1,
        "exsectionformat": "wiki",
        "cllimit": "max",
        "clshow": "!hidden",
        "redirects": 1,
    })
    pages = data["query"]["pages"]
    if not pages:
        return None
    page = pages[0]
    text = clean_extract(page.get("extract", ""))
    if page.get("missing") or not text:
        return None
    categories = [c["title"].removeprefix("Category:") for c in page.get("categories", [])]
    return {
        "pageid": page["pageid"],
        "title": page["title"],
        "url": f"https://en.wikipedia.org/?curid={page['pageid']}",
        "categories": categories,
        "text": text,
    }


def main():
    candidates = [SEED] + seed_links(SEED)
    print(f"Seed '{SEED}' offers {len(candidates) - 1} candidate links; collecting {TARGET_COUNT} articles.")

    corpus = []
    seen_ids = set()
    for title in candidates:
        if len(corpus) >= TARGET_COUNT:
            break
        article = fetch_article(title)
        if article is None:
            print(f"  skip (missing/empty): {title}")
            continue
        if article["pageid"] in seen_ids:
            continue
        seen_ids.add(article["pageid"])
        corpus.append(article)
        print(f"  [{len(corpus):2d}/{TARGET_COUNT}] {article['title']} "
              f"— {len(article['text']):,} chars, {len(article['categories'])} cats")
        time.sleep(0.2)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for article in corpus:
            out.write(json.dumps(article, ensure_ascii=False) + "\n")

    total_chars = sum(len(a["text"]) for a in corpus)
    print(f"\nSaved {len(corpus)} articles to {OUT_FILE} ({total_chars:,} chars total).")


if __name__ == "__main__":
    main()
