import os
import json
import argparse
import re
import unicodedata
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# --- Helper Functions for Post-Processing ---

def is_page_number(text: str) -> bool:
    """Checks if text matches common page number patterns."""
    return bool(re.match(r"^(page|pg\.?)\s+\d+(\s+of\s+\d+)?$", text, re.IGNORECASE))

def is_mostly_punctuation(text: str, threshold: float = 0.8) -> bool:
    """Checks if text consists mostly of punctuation characters."""
    if not text or len(text) <= 5: return False
    punct_count = sum(1 for c in text if unicodedata.category(c).startswith('P'))
    return (punct_count / len(text)) >= threshold

def is_only_symbols_or_digits(text: str) -> bool:
    """Checks if text contains only digits, symbols, spaces, or basic punctuation."""
    return bool(re.fullmatch(r"[\d\s.\-/:()]+", text))

def is_date(text: str) -> bool:
    """Checks for common date patterns."""
    return bool(
        re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b", text, re.IGNORECASE) or
        re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    )

def is_table_of_contents_entry(text: str) -> bool:
    """Checks if text appears to be a table of contents entry with dotted leaders."""
    # Pattern: text followed by dots/periods and ending with a number (page number)
    # Examples: "Introduction to Foundation Level Extensions\n............................................................................\n6"
    #          "Chapter 1: Getting Started .................... 10"
    #          "Section 2.1 Overview ..................... 25"
    
    # Check for dotted leaders pattern: 3 or more consecutive dots/periods
    has_dots = bool(re.search(r'\.{3,}', text))
    
    # Check if it ends with a number (likely page number), allowing for whitespace and newlines
    ends_with_number = bool(re.search(r'\s*\d+\s*$', text))
    
    # Additional pattern: Check for common TOC structures
    # Look for patterns like "Section X.Y" or "Chapter N" followed by dots and numbers
    has_section_pattern = bool(re.search(r'(section|chapter|\d+\.?\d*)\s+.*\.{3,}.*\d+', text, re.IGNORECASE))
    
    return has_dots and ends_with_number or has_section_pattern

def is_reference_or_citation(text: str) -> bool:
    """Checks if text appears to be a reference, citation, or identifier that starts with brackets."""
    # Pattern: text that starts with brackets like [ISTQB-Web], [1], [Smith2021], etc.
    # Examples: "[ISTQB-Web]", "[1]", "[Smith et al., 2021]", "[RFC-3986]"
    
    stripped_text = text.strip()
    
    # Check if text starts with opening bracket and contains closing bracket
    starts_with_bracket = stripped_text.startswith('[')
    has_closing_bracket = ']' in stripped_text
    
    # Additional patterns for common reference formats
    # Pattern 1: [word-word] or [word_word] (like [ISTQB-Web])
    reference_pattern = bool(re.match(r'^\[[A-Za-z0-9_-]+\]', stripped_text))
    
    # Pattern 2: [number] (like [1], [23])
    numeric_ref_pattern = bool(re.match(r'^\[\d+\]', stripped_text))
    
    # Pattern 3: [Author Year] or [Author et al., Year] (like [Smith 2021])
    citation_pattern = bool(re.match(r'^\[[A-Za-z]+.*\d{4}.*\]', stripped_text))
    
    return (starts_with_bracket and has_closing_bracket) and (
        reference_pattern or numeric_ref_pattern or citation_pattern
    )

def is_valid_heading_start(text: str) -> bool:
    """Returns False only if the first alphabetic character is lowercase."""
    for char in text:
        if char.isalpha():
            return not char.islower()
    return True  # Allow non-alphabetic starts (e.g., numbers, symbols)

def infer_numbering_heading(text: str) -> str | None:
    """Infers heading level from numbered prefixes like '1.1' or 'A.'."""
    text = text.strip()
    decimal_match = re.match(r'^(\d+(?:\.\d+){0,2})\s+', text)
    if decimal_match:
        level = min(decimal_match.group(1).count('.') + 1, 3)
        return f"H{level}"
    return None

# --- New Functions for Pipeline Branching ---

def is_document_style_uniform(blocks: List[Dict[str, Any]], std_dev_threshold: float = 0.5, unique_count_threshold: int = 2) -> bool:
    """
    Determines if a document's styling is 'flat' or uniform, making clustering
    by font style ineffective.
    """
    if len(blocks) < 10:  # Not enough data to be sure, assume varied for safety
        return False

    fontsizes = [b.get('fontsize', 0) for b in blocks]
    if not fontsizes:
        return True  # No font data, so we must use rules

    # 1. Check if the standard deviation of font sizes is very low
    if np.std(fontsizes) < std_dev_threshold:
        return True

    # 2. Check if there are very few unique font sizes
    if len(set(fontsizes)) <= unique_count_threshold:
        return True

    return False

def label_blocks_by_rule(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assigns hierarchical labels (Title, H1, H2, H3) for low-variance docs
    using a rule-based scoring system.
    """
    # 1. Pre-computation: Sort blocks and calculate the vertical space before each block.
    # This helps identify headings, which often have more space above them.
    blocks.sort(key=lambda b: (b['page_number'], -b['features'].get('y_coordinate', 0)))

    for i, block in enumerate(blocks):
        # The first block on a page is given a large default space.
        if i == 0 or blocks[i-1]['page_number'] != block['page_number']:
            block['features']['space_before'] = 100
        else:
            prev_block = blocks[i - 1]
            space = prev_block['features'].get('y_coordinate', 0) - block['features'].get('y_coordinate', 0)
            block['features']['space_before'] = max(0, space)

    # 2. Labeling: Iterate through blocks and apply rules based on features.
    for i, block in enumerate(blocks):
        features = block['features']
        text = block['text'].strip()
        label = "Body"  # Start with the default label

        # Rule 0: Filter out noise like page numbers or isolated symbols.
        if is_page_number(text) or (features['word_count'] == 1 and not text.isalnum()):
            block['predicted_label'] = "Noise"
            continue

        # Rule 1: The very first block of the document is likely the Title if it's short.
        if i == 0 and features['word_count'] < 25:
            label = "Title"
        else:
            # Rule 2: Use a scoring system for all other blocks to find headings.
            heading_score = 0

            # --- Attribute Scores ---
            # Style: Bold text is a strong indicator of a heading.
            if features.get('is_bold', 0):
                heading_score += 3
            if features.get('is_italics', 0):
                heading_score += 3
            if features.get('is_underline', 0):
                heading_score += 3
            if features.get('is_all_caps', 0):
                heading_score += 3
            if text.isupper() and features['word_count'] > 1:
                heading_score += 1

            # Layout: More space before a block suggests it's a new section.
            space_before = features.get('space_before', 0)
            if space_before > 25:  # Significant gap
                heading_score += 2
            elif space_before > 15: # Moderate gap
                heading_score += 1

            # Content: Headings are typically shorter than body text.
            word_count = features['word_count']
            if word_count < 10:  # Very short text
                heading_score += 2
            elif word_count < 20: # Moderately short
                heading_score += 1

            # Pattern: Numbered headings (e.g., "1.1 Introduction") are a very strong signal.
            if infer_numbering_heading(text):
                heading_score += 4

            # --- Thresholding ---
            # Apply labels hierarchically based on the final score.
            if heading_score >= 7:
                label = "H1"
            elif heading_score >= 5:
                label = "H2"
            elif heading_score >= 3:
                label = "H3"
            # If the score is below the H3 threshold, the label remains "Body".

        block['predicted_label'] = label

    # 3. Final Cleanup: Remove blocks labeled as "Noise" from the final output.
    return [b for b in blocks if b.get("predicted_label") != "Noise"]
# --- Core Processing Steps ---

def load_blocks(filepath: str) -> List[Dict[str, Any]]:
    """Loads and parses the JSON file containing block data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {filepath}: {e}")
        return []

def perform_clustering(blocks: List[Dict[str, Any]], n_clusters: int) -> List[Dict[str, Any]]:
    """Normalizes features and clusters blocks using K-Means."""
    if len(blocks) < n_clusters:
        print(f"Warning: Number of blocks ({len(blocks)}) is less than n_clusters. Adjusting to {len(blocks)}.")
        n_clusters = len(blocks)

    feature_vectors = np.array([block['feature_vector'] for block in blocks])
    scaled_features = StandardScaler().fit_transform(feature_vectors)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(scaled_features)

    for i, block in enumerate(blocks):
        block['cluster_label'] = int(cluster_labels[i])
    return blocks

def map_clusters_to_labels(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assigns initial semantic labels (Title, H1, etc.) based on cluster font size."""
    cluster_labels = {block['cluster_label'] for block in blocks}

    avg_font_sizes = {
        i: np.mean([b['fontsize'] for b in blocks if b['cluster_label'] == i])
        for i in cluster_labels
    }

    sorted_clusters = sorted(avg_font_sizes.items(), key=lambda item: item[1], reverse=True)

    semantic_map = ["Title", "H1", "H2", "H3", "Body"]
    # Adjust map size to number of actual clusters found
    label_map = {cluster_id: semantic_map[i] for i, (cluster_id, _) in enumerate(sorted_clusters) if i < len(semantic_map)}

    for block in blocks:
        # Default to 'Body' if a cluster is beyond the primary semantic labels
        block['predicted_label'] = label_map.get(block.get('cluster_label'), 'Body')
    return blocks

def post_process_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies a single pass of rules to clean and correct block labels from ANY pipeline."""

    # 0. Detect repeating text (likely running headers/footers)
    text_occurrences = defaultdict(list)
    for block in blocks:
        text_key = block['text'].strip()
        if text_key:
            text_occurrences[text_key].append(block['page_number'])

    # Flag text that appears on more than 2 unique pages as repeating
    repeating_texts = {
        text for text, pages in text_occurrences.items()
        if len(set(pages)) > 2
    }

    # 1. Process each block once with a set of ordered rules
    title_found = False
    for block in blocks:
        text = block["text"].strip()

        # Rule 1: Immediately reject definite non-headings with high confidence
        if (text in repeating_texts or
            is_page_number(text) or
            is_date(text) or
            is_table_of_contents_entry(text) or
            is_reference_or_citation(text) or
            is_mostly_punctuation(text) or
            is_only_symbols_or_digits(text)):
            block["predicted_label"] = "Body"
            continue

        # Rule 2: Re-classify based on numbering (strong signal), overrides previous labels
        inferred_level = infer_numbering_heading(text)
        if inferred_level:
            block["predicted_label"] = inferred_level

        # Rule 3: Final checks for heading/title candidates
        label = block["predicted_label"]
        if label.startswith("H") or label == "Title":
            # Ensure only one Title exists, demote others
            if label == "Title":
                if not title_found:
                    title_found = True
                else:
                    block['predicted_label'] = 'H1'

            # Demote headings that are too long or start with a lowercase letter
            word_count = block["features"].get("word_count", 0)
            if word_count > 15 or not is_valid_heading_start(text):
                block["predicted_label"] = "Body"

    return blocks

def save_blocks(blocks: List[Dict[str, Any]], filepath: str):
    """Saves the final processed blocks to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(blocks, f, ensure_ascii=False, indent=4)
        print(f"  - Success: Saved {len(blocks)} processed blocks -> '{filepath}'")
    except Exception as e:
        print(f"  - FAILED: Could not write to output file {filepath}. Error: {e}")

# --- Main Pipeline Orchestration ---

def run_pipeline_for_file(json_path: str, output_path: str, n_clusters: int = 5):
    """
    Executes the full labeling pipeline for a single file, choosing the best
    strategy (clustering vs. rule-based) based on the document's styling.
    """
    blocks = load_blocks(json_path)
    if not blocks:
        print(f"  - SKIPPING: No data loaded from '{json_path}'.")
        return

    # BRANCH: Decide which pipeline to use based on font style analysis
    if is_document_style_uniform(blocks):
        print("  - INFO: Uniform document style detected. Using rule-based pipeline.")
        blocks = label_blocks_by_rule(blocks)
    else:
        print("  - INFO: Varied document style detected. Using clustering pipeline.")
        blocks = perform_clustering(blocks, n_clusters)
        blocks = map_clusters_to_labels(blocks)

    # Apply a final, universal post-processing step to clean up results from either pipeline
    blocks = post_process_blocks(blocks)

    save_blocks(blocks, output_path)

def process_directory(input_dir: str, output_dir: str):
    """Processes all JSON files in an input directory."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output directory: {os.path.abspath(output_dir)}\n")

    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    if not json_files:
        print("No JSON files found in the input directory.")
        return

    print(f"Found {len(json_files)} JSON file(s) to process...")
    for filename in json_files:
        print(f"\nProcessing '{filename}'...")
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        run_pipeline_for_file(input_path, output_path)

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cluster text blocks and assign hierarchical labels.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing JSON files with feature vectors.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the final labeled JSON files will be saved.")
    args = parser.parse_args()

    # --- How to Run ---
    #  python cluster_and_label.py --input_dir output_vectors --output_dir output_labeled
    process_directory(args.input_dir, args.output_dir)
































