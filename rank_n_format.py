import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Keywords for relevance scoring (same as in conference_paper_collect.py)
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

# Weighted keywords for more fine-grained relevance scoring
WEIGHTED_KEYWORDS = {
    # High priority - core multimodal agent concepts
    "multimodal": 10,
    "multi-modal": 10,
    "vision-language": 8,
    "vision language": 8,
    "agent": 10,
    "agents": 10,
    "vla": 8,
    "vlm": 8,
    "mllm": 8,
    
    # Medium priority - related concepts
    "embodied": 7,
    "autonomous": 6,
    "tool use": 7,
    "tool-use": 7,
    "tool calling": 7,
    "reasoning": 5,
    "grounding": 6,
    "cross-modal": 7,
    "audio-visual": 6,
    "video-language": 7,
    
    # Lower priority - supporting concepts
    "assistant": 4,
    "workflow": 4,
    "planner": 5,
    "orchestrator": 4,
    "image-text": 5,
    "video understanding": 5,
}


def calculate_relevance_score(paper):
    """
    Calculate relevance score for a paper based on keyword matching.
    Returns (score, matched_keywords)
    """
    text = " ".join([
        paper.get("title", ""),
        paper.get("abstract", "")
    ]).lower()
    
    score = 0.0
    matched_keywords = []
    
    # Pattern-based matching
    for pattern in KEYWORDS_ANY:
        matches = re.findall(pattern, text)
        if matches:
            # Add base score for pattern match
            score += len(matches) * 2
            matched_keywords.extend(matches)
    
    # Weighted keyword matching
    for keyword, weight in WEIGHTED_KEYWORDS.items():
        keyword_lower = keyword.lower()
        # Count occurrences
        count = text.count(keyword_lower)
        if count > 0:
            score += count * weight
            if keyword not in matched_keywords:
                matched_keywords.append(keyword)
    
    # Bonus for title matches (title keywords are more important)
    title_lower = paper.get("title", "").lower()
    for keyword, weight in WEIGHTED_KEYWORDS.items():
        if keyword.lower() in title_lower:
            score += weight * 2  # Double weight for title matches
    
    return score, matched_keywords


def rank_papers(csv_file):
    """
    Read papers from CSV and rank them by relevance.
    Returns list of (paper_dict, score, matched_keywords)
    """
    papers = []
    
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score, keywords = calculate_relevance_score(row)
            papers.append((row, score, keywords))
    
    # Sort by score in descending order
    papers.sort(key=lambda x: x[1], reverse=True)
    
    return papers


def extract_institutions(authors_str):
    """
    Extract institution names from authors string.
    For now, just return a placeholder since institution info isn't in the CSV.
    """
    # The CSV format doesn't include institutions separately
    # Return empty string for now
    return ""


def generate_markdown_report(
    ranked_papers,
    top_k,
    output_file,
    conference_name="NeurIPS 2025"
):
    """
    Generate a markdown report in the format shown in the image.
    """
    # Take top k papers
    top_papers = ranked_papers[:top_k]
    
    # Build markdown content
    lines = []
    lines.append("# Recommended Papers")
    lines.append("")
    lines.append(f"**Total Papers:** {len(top_papers)}")
    lines.append("")
    
    for idx, (paper, score, keywords) in enumerate(top_papers, 1):
        # Title
        title = paper.get("title", "Untitled")
        lines.append(f"## {idx}. {title}")
        lines.append("")
        
        # Institutions (extracted from authors if available)
        authors = paper.get("authors", "")
        if authors and authors != "Main Navigation":  # Skip navigation artifacts
            lines.append(f"**Institutions:** {authors}")
        else:
            lines.append(f"**Institutions:** Not available")
        lines.append("")
        
        # Abstract
        abstract = paper.get("abstract", "No abstract available")
        # Clean up abstract if it contains navigation text
        if "San Diego Mexico City Select Year" in abstract:
            # Try to extract just the actual abstract content
            parts = abstract.split("Abstract:")
            if len(parts) > 1:
                abstract = parts[1].split("Live content")[0].strip()
            else:
                # Just take the longest paragraph-like segment
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
        
        # Relevance info (optional - can be commented out if not desired)
        lines.append(f"**Relevance Score:** {score:.1f}")
        lines.append("")
        lines.append(f"**Matched Keywords:** {', '.join(set(keywords[:10]))}")
        lines.append("")
        
        # Separator
        lines.append("---")
        lines.append("")
    
    # Write to file
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return len(top_papers)


def main():
    parser = argparse.ArgumentParser(
        description="Rank papers by relevance and generate markdown report"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file from paper collection script"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file (default: input_name_top_k.md)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Number of top papers to include in report"
    )
    parser.add_argument(
        "--conference",
        type=str,
        default="NeurIPS 2025",
        help="Conference name for the report"
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show relevance scores in the report"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Set default output file
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_top_{args.top_k}.md"
    
    print(f"Reading papers from {args.input}...")
    ranked_papers = rank_papers(args.input)
    
    print(f"Found {len(ranked_papers)} papers")
    print(f"\nTop {min(args.top_k, len(ranked_papers))} papers by relevance score:")
    for i, (paper, score, keywords) in enumerate(ranked_papers[:args.top_k], 1):
        print(f"{i}. [{score:.1f}] {paper.get('title', 'Untitled')[:80]}...")
    
    print(f"\nGenerating markdown report...")
    num_papers = generate_markdown_report(
        ranked_papers,
        args.top_k,
        args.output,
        args.conference
    )
    
    print(f"Successfully generated report with {num_papers} papers")
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
