#!/usr/bin/env python3
"""
Conference Paper Manager - Collect and rank papers from academic conferences
Supports filtering by multiple keyword profiles defined in keywords_config.json
"""

import re
import csv
import json
import html
import argparse
from pathlib import Path
from urllib.parse import urljoin, quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from rapidfuzz import fuzz

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
}

MAX_WORKERS = 20
FUZZY_TITLE_THRESHOLD = 60

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
})


# ============================================================================
# Keyword Profile Management
# ============================================================================

def load_keyword_profiles(config_file: Path):
    """Load keyword profiles from JSON configuration file"""
    if not config_file.exists():
        raise FileNotFoundError(f"Keywords config file not found: {config_file}")
    
    with config_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("profiles", {})


def combine_profiles(profiles_data: dict, selected_profiles: list):
    """
    Combine multiple keyword profiles into one.
    Returns (patterns, weighted_keywords)
    """
    all_patterns = []
    all_weighted = {}
    
    for profile_name in selected_profiles:
        if profile_name not in profiles_data:
            print(f"Warning: Profile '{profile_name}' not found in config, skipping...")
            continue
        
        profile = profiles_data[profile_name]
        
        # Add patterns
        if "patterns" in profile:
            all_patterns.extend(profile["patterns"])
        
        # Merge weighted keywords (if same keyword appears in multiple profiles, use max weight)
        if "weighted" in profile:
            for keyword, weight in profile["weighted"].items():
                if keyword in all_weighted:
                    all_weighted[keyword] = max(all_weighted[keyword], weight)
                else:
                    all_weighted[keyword] = weight
    
    # Remove duplicate patterns
    all_patterns = list(set(all_patterns))
    
    return all_patterns, all_weighted


# ============================================================================
# Paper Collection Functions
# ============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get(url, **kwargs):
    resp = SESSION.get(url, timeout=15, **kwargs)
    resp.raise_for_status()
    return resp


def clean_text(s):
    if not s:
        return s
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def matches_keywords(text: str, patterns: list) -> bool:
    """Check if text matches any of the keyword patterns"""
    if not text:
        return False
    t = text.lower()
    for pat in patterns:
        if re.search(pat, t):
            return True
    return False


def parse_papers_list(html_text: str, config: dict):
    """Parse paper list page to extract title and detail links"""
    soup = BeautifulSoup(html_text, "html.parser")
    links = []
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(strip=True)
        
        if re.search(config["url_pattern"], href):
            abs_url = urljoin(config["base_url"], href)
            if title:
                links.append({"title": title, "url": abs_url})
    
    # Deduplicate by url
    uniq = {}
    for x in links:
        uniq[x["url"]] = x
    return list(uniq.values())


def parse_detail_page(url: str):
    """Parse individual paper detail page"""
    r = get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Title
    title = ""
    for h2 in soup.find_all("h2"):
        h2_text = clean_text(h2.get_text())
        if h2_text and h2_text.lower() not in ["main navigation", "navigation"]:
            title = h2_text
            break
    
    # Authors
    authors = []
    for h3 in soup.find_all("h3"):
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
        paras = [clean_text(p.get_text()) for p in soup.find_all(["p", "div"]) if p.get_text()]
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
            proceedings_url = a["href"]
    
    return {
        "title": title,
        "authors": ", ".join([a for a in authors if a]),
        "abstract": abstract,
        "time_location": meta_time_loc,
        "page_url": url,
        "project_url": project_url,
        "proceedings_url": proceedings_url,
    }


def process_single_item(item, patterns):
    """Process a single paper item"""
    try:
        detail = parse_detail_page(item["url"])
        text_for_match = " ".join([
            detail.get("title", ""), 
            detail.get("abstract", "")
        ]).lower()
        
        hit = matches_keywords(text_for_match, patterns)
        
        # Fuzzy matching fallback
        if not hit:
            title = detail.get("title", "")
            for kw in ["agent", "vision", "audio", "language", "multimodal", "video"]:
                if fuzz.token_set_ratio(title.lower(), kw) >= FUZZY_TITLE_THRESHOLD:
                    hit = True
                    break
        
        if not hit:
            return None
        
        return detail
        
    except Exception as e:
        print(f"\n[WARN] Failed on {item['url']}: {e}")
        return None


def collect_papers(conference: str, profiles: list, config_file: Path, 
                   output_dir: Path, max_workers: int = MAX_WORKERS):
    """
    Collect papers from conference based on keyword profiles
    """
    # Load keyword profiles
    profiles_data = load_keyword_profiles(config_file)
    patterns, _ = combine_profiles(profiles_data, profiles)
    
    print(f"Using {len(profiles)} profile(s): {', '.join(profiles)}")
    print(f"Total patterns: {len(patterns)}")
    
    # Get conference config
    config = CONFERENCE_CONFIGS[conference]
    
    # Output file
    profile_str = "_".join(profiles)
    out_csv = output_dir / f"{conference}_{profile_str}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFetching papers list from {config['papers_list_url']}...")
    html_text = get(config["papers_list_url"]).text
    all_items = parse_papers_list(html_text, config)
    print(f"Found {len(all_items)} candidate items from {conference.upper()} virtual site.")
    
    results = []
    
    # Process papers in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_item, item, patterns): item 
            for item in all_items
        }
        
        for future in tqdm(as_completed(futures), total=len(all_items), desc="Processing"):
            row = future.result()
            if row:
                results.append(row)
    
    # Export to CSV
    if results:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "title", "authors", "abstract", "time_location",
                "page_url", "project_url", "proceedings_url"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nSaved {out_csv.resolve()} ({len(results)} papers)")
        return out_csv
    else:
        print("\nNo matching papers found.")
        return None


# ============================================================================
# Paper Ranking Functions
# ============================================================================

def calculate_relevance_score(paper, patterns, weighted_keywords):
    """Calculate relevance score for a paper"""
    text = " ".join([
        paper.get("title", ""),
        paper.get("abstract", "")
    ]).lower()
    
    score = 0.0
    matched_keywords = []
    
    # Pattern-based matching
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            score += len(matches) * 2
            matched_keywords.extend(matches)
    
    # Weighted keyword matching
    for keyword, weight in weighted_keywords.items():
        keyword_lower = keyword.lower()
        count = text.count(keyword_lower)
        if count > 0:
            score += count * weight
            if keyword not in matched_keywords:
                matched_keywords.append(keyword)
    
    # Bonus for title matches
    title_lower = paper.get("title", "").lower()
    for keyword, weight in weighted_keywords.items():
        if keyword.lower() in title_lower:
            score += weight * 2
    
    return score, matched_keywords


def rank_papers(csv_file: Path, patterns: list, weighted_keywords: dict):
    """Rank papers from CSV by relevance score"""
    papers = []
    
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score, keywords = calculate_relevance_score(row, patterns, weighted_keywords)
            papers.append((row, score, keywords))
    
    papers.sort(key=lambda x: x[1], reverse=True)
    return papers


def generate_markdown_report(ranked_papers, top_k, output_file, 
                            conference_name="NeurIPS 2025", show_scores=True):
    """Generate markdown report of top papers"""
    top_papers = ranked_papers[:top_k]
    
    lines = []
    lines.append("# Recommended Papers")
    lines.append("")
    lines.append(f"**Conference:** {conference_name}")
    lines.append(f"**Total Papers:** {len(top_papers)}")
    lines.append("")
    
    for idx, (paper, score, keywords) in enumerate(top_papers, 1):
        title = paper.get("title", "Untitled")
        lines.append(f"## {idx}. {title}")
        lines.append("")
        
        # Authors
        authors = paper.get("authors", "")
        if authors and authors != "Main Navigation":
            lines.append(f"**Authors:** {authors}")
        else:
            lines.append(f"**Authors:** Not available")
        lines.append("")
        
        # Abstract
        abstract = paper.get("abstract", "No abstract available")
        if "San Diego Mexico City Select Year" in abstract:
            parts = abstract.split("Abstract:")
            if len(parts) > 1:
                abstract = parts[1].split("Live content")[0].strip()
            else:
                sentences = [s.strip() for s in abstract.split(".") if len(s.strip()) > 50]
                if sentences:
                    abstract = ". ".join(sentences[:5]) + "."
        
        lines.append(f"**Abstract:** {abstract}")
        lines.append("")
        
        # Links
        page_url = paper.get("page_url", "")
        project_url = paper.get("project_url", "")
        
        if page_url:
            lines.append(f"**Conference Page:** {page_url}")
            lines.append("")
        
        if project_url:
            lines.append(f"**Project Page:** {project_url}")
            lines.append("")
        
        # Relevance info
        if show_scores:
            lines.append(f"**Relevance Score:** {score:.1f}")
            lines.append("")
            lines.append(f"**Matched Keywords:** {', '.join(set(keywords[:10]))}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return len(top_papers)


def rank_and_report(input_csv: Path, profiles: list, config_file: Path,
                   output_md: Path = None, top_k: int = 200, 
                   conference_name: str = "NeurIPS 2025", show_scores: bool = True):
    """
    Rank papers and generate markdown report
    """
    if not input_csv.exists():
        print(f"Error: Input file {input_csv} does not exist")
        return
    
    # Load keyword profiles
    profiles_data = load_keyword_profiles(config_file)
    patterns, weighted = combine_profiles(profiles_data, profiles)
    
    print(f"Using {len(profiles)} profile(s) for ranking: {', '.join(profiles)}")
    
    # Set default output
    if output_md is None:
        output_md = input_csv.parent / f"{input_csv.stem}_top_{top_k}.md"
    
    print(f"\nReading papers from {input_csv}...")
    ranked_papers = rank_papers(input_csv, patterns, weighted)
    
    print(f"Found {len(ranked_papers)} papers")
    print(f"\nTop {min(top_k, len(ranked_papers))} papers by relevance score:")
    for i, (paper, score, keywords) in enumerate(ranked_papers[:min(10, top_k)], 1):
        print(f"{i}. [{score:.1f}] {paper.get('title', 'Untitled')[:80]}...")
    
    if len(ranked_papers) > 10:
        print("...")
    
    print(f"\nGenerating markdown report...")
    num_papers = generate_markdown_report(
        ranked_papers, top_k, output_md, conference_name, show_scores
    )
    
    print(f"Successfully generated report with {num_papers} papers")
    print(f"Saved to: {output_md.resolve()}")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Conference Paper Manager - Collect and rank papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect papers on multimodal agents from NeurIPS 2025
  python paper_manager.py collect --conference neurips_2025 --profiles multimodal_agent
  
  # Collect papers matching multiple profiles
  python paper_manager.py collect --conference neurips_2025 --profiles multimodal_agent llm_reasoning
  
  # Rank collected papers
  python paper_manager.py rank --input neurips_2025_multimodal_agent.csv --profiles multimodal_agent --top-k 50
  
  # Collect and rank in one go
  python paper_manager.py collect --conference neurips_2025 --profiles multimodal_sentiment --then-rank --top-k 100
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ===== Collect command =====
    collect_parser = subparsers.add_parser("collect", help="Collect papers from conference")
    collect_parser.add_argument(
        "--conference",
        type=str,
        required=True,
        choices=list(CONFERENCE_CONFIGS.keys()),
        help="Conference name"
    )
    collect_parser.add_argument(
        "--profiles",
        type=str,
        nargs="+",
        required=True,
        help="One or more keyword profiles to use (from keywords_config.json)"
    )
    collect_parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "keywords_config.json",
        help="Path to keywords config file"
    )
    collect_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory for CSV files"
    )
    collect_parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of concurrent workers"
    )
    collect_parser.add_argument(
        "--then-rank",
        action="store_true",
        help="Automatically rank papers after collection"
    )
    collect_parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Number of top papers in report (if --then-rank is used)"
    )
    
    # ===== Rank command =====
    rank_parser = subparsers.add_parser("rank", help="Rank papers from CSV")
    rank_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file"
    )
    rank_parser.add_argument(
        "--profiles",
        type=str,
        nargs="+",
        required=True,
        help="One or more keyword profiles to use for ranking"
    )
    rank_parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "keywords_config.json",
        help="Path to keywords config file"
    )
    rank_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file (default: auto-generated)"
    )
    rank_parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Number of top papers to include"
    )
    rank_parser.add_argument(
        "--conference",
        type=str,
        default="NeurIPS 2025",
        help="Conference name for report"
    )
    rank_parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Hide relevance scores in report"
    )
    
    args = parser.parse_args()
    
    if args.command == "collect":
        output_csv = collect_papers(
            args.conference,
            args.profiles,
            args.config,
            args.output_dir,
            args.max_workers
        )
        
        if args.then_rank and output_csv:
            print("\n" + "="*80)
            print("Starting ranking process...")
            print("="*80 + "\n")
            rank_and_report(
                output_csv,
                args.profiles,
                args.config,
                top_k=args.top_k,
                conference_name=CONFERENCE_CONFIGS[args.conference]["booktitle"],
                show_scores=True
            )
    
    elif args.command == "rank":
        rank_and_report(
            args.input,
            args.profiles,
            args.config,
            args.output,
            args.top_k,
            args.conference,
            not args.no_scores
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
