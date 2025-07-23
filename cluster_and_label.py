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
    n_clusters = len(cluster_labels)

    avg_font_sizes = {
        i: np.mean([b['fontsize'] for b in blocks if b['cluster_label'] == i])
        for i in cluster_labels
    }

    sorted_clusters = sorted(avg_font_sizes.items(), key=lambda item: item[1], reverse=True)
    
    semantic_map = ["Title", "H1", "H2", "H3", "Body"]
    label_map = {cluster_id: semantic_map[i] for i, (cluster_id, _) in enumerate(sorted_clusters)}

    for block in blocks:
        block['predicted_label'] = label_map.get(block.get('cluster_label'), 'Body')
    return blocks

def post_process_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies a single pass of rules to clean and correct block labels."""

    # 0. Detect repeating text (running headers/footers)
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

    # 1. Detect running headers
    header_candidates = defaultdict(list)
    for block in blocks:
        if block.get('features', {}).get('y_norm', 0) > 0.90:
            header_candidates[block['text'].strip()].append(block['page_number'])
    
    repeated_header_texts = {text for text, pages in header_candidates.items() if len(set(pages)) > 2}

    # 2. Process each block once with a set of ordered rules
    title_found = False
    for block in blocks:
        text = block["text"].strip()

        # Rule 1: Immediately reject definite non-headings with high confidence
        if (text in repeating_texts or
            is_page_number(text) or
            is_date(text) or
            is_mostly_punctuation(text) or
            is_only_symbols_or_digits(text)):
            block["predicted_label"] = "Body"
            continue

        # Rule 2: Re-classify based on numbering (strong signal)
        inferred_level = infer_numbering_heading(text)
        if inferred_level:
            block["predicted_label"] = inferred_level
        
        # Rule 3: Final checks for heading/title candidates
        label = block["predicted_label"]
        if label.startswith("H") or label == "Title":
            if label == "Title":
                if not title_found:
                    title_found = True
                else:
                    block['predicted_label'] = 'H1'
            
            word_count = block["features"].get("word_count", 0)
            if word_count > 8 or not is_valid_heading_start(text):
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
    """Executes the full clustering and labeling pipeline for a single file."""
    blocks = load_blocks(json_path)
    if not blocks:
        print(f"  - SKIPPING: No data loaded from '{json_path}'.")
        return

    blocks = perform_clustering(blocks, n_clusters)
    blocks = map_clusters_to_labels(blocks)
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































# import os
# import json
# import argparse
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import MiniBatchKMeans
# import re
# import unicodedata

# def cluster_and_label(input_json_path, n_clusters=5):
#     """
#     Loads blocks with feature vectors, normalizes them, clusters them using
#     K-Means, and assigns hierarchical labels based on cluster properties.
# a
#     Args:
#         input_json_path (str): Path to the JSON file containing blocks with
#                                'feature_vector' keys.
#         n_clusters (int): The number of clusters to form (e.g., 5 for
#                           Title, H1, H2, H3, Body).

#     Returns:
#         list[dict]: A list of the original block dictionaries, now with added
#                     'cluster_label' and 'predicted_label' keys. Returns an
#                     empty list if processing fails.
#     """
#     try:
#         with open(input_json_path, 'r', encoding='utf-8') as f:
#             blocks = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         print(f"Could not read or parse JSON file {input_json_path}. Error: {e}")
#         return []

#     if not blocks:
#         print("JSON file is empty. Nothing to process.")
#         return []

#     # 1. Normalize Features
#     feature_vectors = np.array([block['feature_vector'] for block in blocks])
#     # Handle case with fewer blocks than clusters
#     if len(blocks) < n_clusters:
#         print(f"Warning: Number of blocks ({len(blocks)}) is less than n_clusters ({n_clusters}). Adjusting n_clusters.")
#         n_clusters = len(blocks)

#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(feature_vectors)

#     # 2. Cluster
#     kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
#     kmeans.fit(scaled_features)
#     cluster_labels = kmeans.labels_

#     # Assign the raw cluster label back to each block
#     for i, block in enumerate(blocks):
#         block['cluster_label'] = int(cluster_labels[i])

#     # 3. Map Clusters to Semantic Labels
#     # Group blocks by cluster
#     clusters = {i: [] for i in range(n_clusters)}
#     for block in blocks:
#         clusters[block['cluster_label']].append(block)

#     # Calculate average font size for each cluster
#     avg_font_sizes = {}
#     for i, cluster_blocks in clusters.items():
#         if cluster_blocks:
#             avg_font_sizes[i] = np.mean([b['fontsize'] for b in cluster_blocks])
#         else:
#             avg_font_sizes[i] = 0 # Handle empty clusters

#     # Sort cluster indices by font size (descending)
#     sorted_clusters = sorted(avg_font_sizes.items(), key=lambda item: item[1], reverse=True)
    
#     # Define the hierarchy of labels
#     semantic_labels = ["Title", "H1", "H2", "H3", "Body"]
#     # Ensure we don't have more labels than clusters
#     semantic_labels = semantic_labels[:n_clusters] 

#     # Create the mapping from cluster_label -> semantic_label
#     label_map = {cluster_id: semantic_labels[i] for i, (cluster_id, _) in enumerate(sorted_clusters)}
    
#     # Assign the predicted semantic label
#     for block in blocks:
#         block['predicted_label'] = label_map[block['cluster_label']]

#     # 4. Post-processing
#     def is_mostly_punctuation(text, threshold=0.8):
#         """Multilingual-safe: Checks if text has mostly punctuation."""
#         if not text:
#             return False
#         punct_count = sum(1 for c in text if unicodedata.category(c).startswith('P'))
#         return len(text) > 5 and (punct_count / len(text)) >= threshold

#     def is_only_symbols_or_digits(text):
#         """Reject if text has only digits, symbols, spaces, or punctuation."""
#         return bool(re.fullmatch(r"[\d\s\.\-:()]+", text))

#     def is_page_number(text):
#         return re.match(r"Page\s+\d+\s+of\s+\d+", text, re.IGNORECASE)

#     def is_date(text):
#         return (
#             re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b", text, re.IGNORECASE) or
#             re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text) or
#             re.search(r"\b\d{4}\b", text)
#         )

#     def is_valid_heading_start(text):
#         """
#         Multilingual-safe: Checks if the first letter is valid for a heading.
#         - Returns False ONLY if the first letter is explicitly lowercase.
#         - Returns True for uppercase letters and uncased letters (e.g., Japanese).
#         """
#         for char in text:
#             if char.isalpha():
#                 return not char.islower()
#         return False # No alphabetic characters found
#     def infer_numbering_heading(text):
#         """
#         Infers heading level based on common numbering patterns at the start of the line.
#         Returns one of {'H1', 'H2', 'H3'} if a clear structure is found, else None.
#         """

#         text = text.strip()

#         # Normalize spaces and dots (just in case)
#         text = re.sub(r'\s+', ' ', text)
#         text = re.sub(r'[．。]', '.', text)  # for CJK full-width dots

#         # Patterns
#         decimal_match = re.match(r'^(\d+(?:\.\d+){0,2})\s+', text)  # e.g. 1, 1.1, 1.1.1
#         roman_match = re.match(r'^(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+', text)  # e.g. I., II.
#         alpha_match = re.match(r'^([A-Z])\.\s+', text)  # A. Heading

#         # Decide level based on pattern
#         if decimal_match:
#             level = decimal_match.group(1).count('.') + 1
#             return f"H{min(level, 3)}"
#         elif roman_match:
#             return "H1"
#         elif alpha_match:
#             return "H1"
        
#         return None
    
#     # Sort blocks by their position in the document for logical processing
#     blocks.sort(key=lambda b: (b['page_number'], -b['y0_top']))
    
#     # Correct for multiple titles
#     title_found = False
#     for block in blocks:
#         if block['predicted_label'] == 'Title':
#             if not title_found:
#                 title_found = True
#             else:
#                 # Demote subsequent titles to H1
#                 block['predicted_label'] = 'H1'

#     # Post-processing rule: long headings are likely body text
#     for block in blocks:
#         label = block["predicted_label"]
#         word_count = block["features"].get("word_count", 0)
#         if label in {"Title", "H1", "H2", "H3"} and word_count > 8:
#             block["predicted_label"] = "Body"

#     for block in blocks:
#         text = block["text"].strip()

#         if is_page_number(text):
#             block["predicted_label"] = "Body"

#         elif is_date(text):
#             block["predicted_label"] = "Body"

#         elif is_mostly_punctuation(text):
#             block["predicted_label"] = "Body"

#         elif is_only_symbols_or_digits(text):
#             block["predicted_label"] = "Body"

#         elif label in {"Title", "H1", "H2", "H3"} and not is_valid_heading_start(text):
#             block["predicted_label"] = "Body"

#     for block in blocks:
#         text = block["text"].strip()

#         # Try infer numbering heading
#         inferred_level = infer_numbering_heading(text)
#         if inferred_level:
#             block["predicted_label"] = inferred_level



#     return blocks


# def process_directory(input_dir, output_dir):
#     """
#     Processes all JSON files in an input directory, performs clustering and
#     labeling, and saves the results to a new output directory.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Input (vectors) directory: {os.path.abspath(input_dir)}")
#     print(f"Output (clustered) directory: {os.path.abspath(output_dir)}\n")

#     json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    
#     if not json_files:
#         print("No JSON files found in the input directory.")
#         return

#     print(f"Found {len(json_files)} JSON file(s) to process...")

#     for json_filename in json_files:
#         json_path = os.path.join(input_dir, json_filename)
#         output_path = os.path.join(output_dir, json_filename)
        
#         print(f"\nProcessing '{json_path}'...")
        
#         labeled_data = cluster_and_label(json_path)

#         if labeled_data:
#             try:
#                 with open(output_path, 'w', encoding='utf-8') as f:
#                     json.dump(labeled_data, f, ensure_ascii=False, indent=4)
#                 print(f"  - Success: Clustered and labeled {len(labeled_data)} blocks -> '{output_path}'")
#             except Exception as e:
#                 print(f"  - FAILED: Could not write to output file {output_path}. Error: {e}")
#         else:
#             print(f"  - SKIPPING: No data processed for '{json_path}'.")

# # --- Main Execution Block ---
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="Cluster text blocks and assign hierarchical labels.",
#         formatter_class=argparse.RawTextHelpFormatter
#     )
#     parser.add_argument(
#         '--input_dir',
#         type=str,
#         required=True,
#         help="Directory containing JSON files with feature vectors."
#     )
#     parser.add_argument(
#         '--output_dir',
#         type=str,
#         required=True,
#         help="Directory where the final labeled JSON files will be saved."
#     )

#     args = parser.parse_args()
    
#     # --- How to Run ---
#     # 1. Save this script as `cluster_and_label.py`.
#     # 2. You should have two folders:
#     #    - `output_vectors`: Contains the JSON outputs from the vector builder script.
#     #    - `output_labeled`: An empty folder for this script's final output.
#     # 3. Run from your terminal:
#     #
#     #    python cluster_and_label.py --input_dir output_vectors --output_dir output_labeled

#     process_directory(args.input_dir, args.output_dir)
