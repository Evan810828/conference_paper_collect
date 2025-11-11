import re
import csv
import json
import time
import html
import argparse
from pathlib import Path
from urllib.parse import urljoin, quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from rapidfuzz import fuzz, process as rf_process
import bibtexparser

# Conference configurations
CONFERENCE_CONFIGS = {
    "neurips_2025": {
        "base_url": "https://neurips.cc",
        "papers_list_url": "https://neurips.cc/virtual/2025/papers.html",
        "year": "2025",
        "url_pattern": r"/virtual/2025/(poster|oral)/\d+$",
        "booktitle": "Advances in Neural Information Processing Systems (NeurIPS 2025)",
    },
    "iclr_2026": {
        "base_url": "https://iclr.cc",
        "papers_list_url": "https://iclr.cc/virtual/2026/papers.html",
        "year": "2026",
        "url_pattern": r"/virtual/2026/(poster|oral)/\d+$",
        "booktitle": "International Conference on Learning Representations (ICLR 2026)",
    },
    # Add more conferences as needed
}

MAX_WORKERS = 20  # Concurrent workers for faster processing
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
})

# --- Keywords: Multimodal + Agent (add/remove as needed) ---
KEYWORDS_ANY = [
    # Core multimodal keywords
    r"\bmultimodal\b", r"\bmulti-modal\b", r"\bmllm\b", r"\bvlm\b", r"\bvla\b",
    r"vision[-\s]?language", r"audio[-\s]?(language|visual|text)", r"speech[-\s]?text",
    r"video[-\s]?(language|qa|understanding|reasoning)", r"audio[-\s]?visual",
    r"multisensory", r"cross[-\s]?modal", r"image[-\s]?text", r"video[-\s]?text",
    # Agent / tool use / embodied
    r"\bagent(s)?\b", r"tool[-\s]?use", r"tool[-\s]?calling", r"\bembodied\b",
    r"autonomous", r"\bassistant\b", r"web[-\s]?agent", r"computer[-\s]?use",
    r"browser[-\s]?agent", r"robot(ic)?[-\s]?agent", r"workflow", r"orchestrator",
    r"planner", r"planner[-\s]?executor"
]

# Minimum fuzzy matching threshold for title/abstract (lower = more lenient)
FUZZY_TITLE_THRESHOLD = 60  # Title fuzzy matching threshold

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get(url, **kwargs):
    resp = SESSION.get(url, timeout=15, **kwargs)
    resp.raise_for_status()
    return resp

def is_multimodal_agent(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    for pat in KEYWORDS_ANY:
        if re.search(pat, t):
            return True
    return False

def parse_papers_list(html_text: str, config: dict):
    """Parse paper list page to extract title and detail links for each paper"""
    soup = BeautifulSoup(html_text, "html.parser")
    links = []
    # Each item in the list page is a hyperlink (poster/oral), directly scrape visible anchor tags
    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(strip=True)
        # Match pattern like /virtual/2025/poster/ID or /virtual/2025/oral/ID
        if re.search(config["url_pattern"], href):
            abs_url = urljoin(config["base_url"], href)
            if title:
                links.append({"title": title, "url": abs_url})
    # Deduplicate by url
    uniq = {}
    for x in links:
        uniq[x["url"]] = x
    return list(uniq.values())

def clean_text(s):
    if not s:
        return s
    return re.sub(r"\s+", " ", html.unescape(s)).strip()

def try_openreview_pdf(title: str):
    """
    Attempt to search OpenReview by title and return a possible PDF link (returns None if not found).
    Note: OpenReview search API is not stable, using two-step approach:
      1) Search by title keywords using /notes?query=title
      2) Use fuzzy matching in results to find the most similar title and extract content.pdf (if available)
    """
    try:
        q = quote(title)
        # Try OpenReview v2 search API (may change; returns None on failure)
        url = f"https://api.openreview.net/notes?details=replyCount&sort=tmdate:desc&term={q}"
        r = get(url, timeout=8)
        data = r.json()
        if not isinstance(data, dict) or "notes" not in data:
            return None
        candidates = []
        for note in data.get("notes", [])[:5]:  # Only check top 5 results
            note_title = note.get("content", {}).get("title", "")
            if not note_title:
                continue
            score = fuzz.token_set_ratio(title, note_title)
            candidates.append((score, note))
        if not candidates:
            return None
        best = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
        if best[0] < 70:
            return None
        pdf_rel = best[1].get("content", {}).get("pdf")
        if isinstance(pdf_rel, str) and pdf_rel:
            if pdf_rel.startswith("http"):
                return pdf_rel
            return urljoin("https://openreview.net", pdf_rel)
    except Exception:
        return None
    return None

def parse_detail_page(url: str):
    """
    Parse individual paper detail page: title, authors, abstract, venue/time, project page/proceedings link, etc.
    """
    r = get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    # Title - skip navigation and find the actual paper title
    title = ""
    for h2 in soup.find_all("h2"):
        h2_text = clean_text(h2.get_text())
        # Skip "Main Navigation" and similar navigation elements
        if h2_text and h2_text.lower() not in ["main navigation", "navigation"]:
            title = h2_text
            break
    
    # Authors (usually in 'h3' or directly below title)
    authors = []
    for h3 in soup.find_all("h3"):
        # Authors on detail page are usually "Name · Name · ..."
        if " · " in h3.get_text():
            authors = [clean_text(x) for x in h3.get_text().split("·")]
            break
    # Abstract
    abstract = ""
    abs_link = None
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True).lower() == "abstract":
            abs_link = a
            break
    if abs_link:
        # Abstract is usually in a text block below on the same page (directly visible, may not require second request)
        # Find paragraphs after "Abstract:" or just traverse all paragraphs and find the longest
        paras = [clean_text(p.get_text()) for p in soup.find_all(["p", "div"]) if p.get_text()]
        # Simple heuristic: longest paragraph containing multiple sentences
        paras = sorted(paras, key=lambda s: len(s or ""), reverse=True)
        if paras:
            abstract = paras[0]
    # Venue/time metadata
    meta_time_loc = ""
    for h5 in soup.find_all(["h5"]):
        meta_time_loc = clean_text(h5.get_text())
        if meta_time_loc:
            break
    # Project page / proceedings
    project_url, proceedings_url = "", ""
    for a in soup.find_all("a", href=True):
        t = a.get_text(strip=True).lower()
        if "project page" in t:
            project_url = a["href"]
        if "proceedings" in t:
            proceedings_url = a["href"]  # May be general entry, not necessarily direct link to paper
    return {
        "title": title,
        "authors": ", ".join([a for a in authors if a]),
        "abstract": abstract,
        "time_location": meta_time_loc,
        "page_url": url,
        "project_url": project_url,
        "proceedings_url": proceedings_url,
    }

def build_bib_entries(rows, config: dict):
    """
    Generate simple BibTeX entries (use placeholder fields if year/volume/pages are missing; try to include openreview_pdf)
    """
    entries = []
    conf_name = config["booktitle"].split("(")[1].split(")")[0].lower().replace(" ", "_")
    for i, r in enumerate(rows, 1):
        key = re.sub(r"[^a-z0-9]+", "_", r["title"].lower())[:60] or f"{conf_name}_{i}"
        authors_bib = " and ".join([x.strip() for x in r["authors"].split(",") if x.strip()])
        entry = {
            "ENTRYTYPE": "inproceedings",
            "ID": key,
            "author": authors_bib or "Unknown",
            "title": r["title"],
            "booktitle": config["booktitle"],
            "year": config["year"],
            "url": r["page_url"],
        }
        if r.get("openreview_pdf"):
            entry["pdf"] = r["openreview_pdf"]
        elif r.get("project_url"):
            entry["pdf"] = r["project_url"]
        entries.append(entry)
    return entries

def process_single_item(item):
    """Process a single paper item - for parallel execution"""
    try:
        detail = parse_detail_page(item["url"])
        text_for_match = " ".join([
            detail.get("title",""), detail.get("abstract","")
        ]).lower()
        hit = is_multimodal_agent(text_for_match)
        # Additional layer: fuzzy title matching (for titles clearly containing "Agent", "Vision-Language", "Audio", etc.)
        if not hit:
            title = detail.get("title", "")
            for kw in ["agent", "vision", "audio", "language", "multimodal", "video"]:
                if fuzz.token_set_ratio(title.lower(), kw) >= FUZZY_TITLE_THRESHOLD:
                    hit = True
                    break
        if not hit:
            return None

        # Try to match PDF on OpenReview (optional) - Skipped for speed
        # pdf_url = try_openreview_pdf(detail["title"])
        pdf_url = None

        row = {
            **detail,
            "openreview_pdf": pdf_url or "",
        }
        return row
    except Exception as e:
        print(f"\n[WARN] Failed on {item['url']}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Scrape conference papers focused on multimodal and agent topics"
    )
    parser.add_argument(
        "--conference",
        type=str,
        default="neurips_2025",
        choices=list(CONFERENCE_CONFIGS.keys()),
        help="Conference name (e.g., neurips_2025, iclr_2026)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for CSV and BibTeX files"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of concurrent workers"
    )
    
    args = parser.parse_args()
    
    # Get conference configuration
    config = CONFERENCE_CONFIGS[args.conference]
    
    # Output files
    out_csv = args.output_dir / f"{args.conference}_multimodal_agents.csv"
    out_bib = args.output_dir / f"{args.conference}_multimodal_agents.bib"
    
    print(f"Fetching papers list from {config['papers_list_url']}...")
    html_text = get(config["papers_list_url"]).text
    all_items = parse_papers_list(html_text, config)
    print(f"Found {len(all_items)} candidate items from {args.conference.upper()} virtual site.")

    results = []
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_single_item, item): item for item in all_items}
        
        for future in tqdm(as_completed(futures), total=len(all_items), desc="Parse details"):
            row = future.result()
            if row:
                results.append(row)

    # Export to CSV
    if results:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "title","authors","abstract","time_location",
                "page_url","project_url","proceedings_url","openreview_pdf"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Saved {out_csv.resolve()} ({len(results)} rows)")

    else:
        print("No matching papers found. Consider loosening KEYWORDS_ANY.")

if __name__ == "__main__":
    main()
