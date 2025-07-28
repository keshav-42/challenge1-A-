import os
import json
import argparse
import re
from typing import List, Dict, Any, Tuple
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextLineHorizontal, LAParams, LTTextContainer
from collections import Counter

# --- NEW: Define Keywords that FORCE a new block ---
# These are words that, if found at the start of a line, will create a new block
# regardless of font style.
HEADING_KEYWORDS = {
    "introduction", "overview", "methodology",
    "appendix",
}

def get_line_style(line: LTTextLineHorizontal) -> Tuple[str, float]:
    """Calculates the most common font name and size for a line."""
    styles = [
        (char.fontname, round(char.size, 2))
        for char in line if isinstance(char, LTChar)
    ]
    if not styles:
        return ("Unknown", 0.0)
    # Return the most common style tuple
    return Counter(styles).most_common(1)[0][0]


def parse_and_feature_engineer(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Parses a PDF by grouping lines into blocks. A new block is started if
    the font style changes OR if a line starts with a heading keyword.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return []

    document_name = os.path.splitext(os.path.basename(pdf_path))[0]
    all_blocks = []

    try:
        for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=LAParams()), 0):
            current_block_lines = []
            last_line_style = None

            def finalize_block():
                """Combines lines in the current buffer into a single block."""
                if not current_block_lines:
                    return

                # Combine text and calculate bounding box
                full_text = "\n".join([line.get_text().strip() for line in current_block_lines])
                x0 = min(line.x0 for line in current_block_lines)
                y0 = min(line.y0 for line in current_block_lines)
                x1 = max(line.x1 for line in current_block_lines)
                y1 = max(line.y1 for line in current_block_lines)
                
                # Use the style from the last line as representative
                fontname, fontsize = last_line_style

                all_blocks.append({
                    "text": full_text,
                    "fontname": fontname,
                    "fontsize": fontsize,
                    "bbox": (x0, y0, x1, y1),
                    "page_number": page_num,
                    "page_width": page_layout.width,
                    "page_height": page_layout.height,
                    "document_name": document_name,
                })

            # Find all text lines on the page
            lines = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    lines.extend([line for line in element if isinstance(line, LTTextLineHorizontal)])

            # Process lines sequentially
            for line in lines:
                line_text = line.get_text().strip()
                if not line_text:
                    continue

                current_line_style = get_line_style(line)
                first_word = line_text.split(' ', 1)[0].lower().strip('•*: ')

                # --- MODIFIED CORE LOGIC ---
                # Check if we need to start a new block
                starts_with_keyword = first_word in HEADING_KEYWORDS
                style_is_different = last_line_style and current_line_style != last_line_style

                if current_block_lines and (style_is_different or starts_with_keyword):
                    finalize_block()
                    current_block_lines = [] # Reset for the new block

                current_block_lines.append(line)
                last_line_style = current_line_style

            finalize_block() # Finalize any remaining lines on the page

        # --- Stage 2: Feature Computation (This part remains largely the same) ---
        final_featured_blocks = []
        for block in all_blocks:
            text = block['text']
            x0, y0, x1, y1 = block['bbox']
            
            # Compute features
            is_bold = 1 if "bold" in block['fontname'].lower() else 0
            word_count = len(text.split())
            centering_offset = abs((x0 + (x1 - x0) / 2) - (block['page_width'] / 2))

            # Assemble final data structure
            final_block_data = {
                'text': text,
                'fontname': block['fontname'],
                'fontsize': block['fontsize'],
                'page_number': block['page_number'],
                'document_name': block['document_name'],
                'features': {
                    'font_size': block['fontsize'],
                    'indentation': round(x0, 2),
                    'is_bold': is_bold,
                    'y_coordinate': round(y1, 2),
                    'centering_offset': round(centering_offset, 2),
                    'word_count': word_count,
                }
            }
            final_featured_blocks.append(final_block_data)

    except Exception as e:
        print(f"Could not process {pdf_path}. Error: {e}")
        return []

    return final_featured_blocks


def process_directory(input_dir: str, output_dir: str):
    """Processes all PDF files in a directory and saves the features as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output directory: {os.path.abspath(output_dir)}\n")

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process...")
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_filename)
        print(f"\nProcessing '{pdf_filename}'...")

        blocks = parse_and_feature_engineer(pdf_path)

        if blocks:
            json_filename = os.path.splitext(pdf_filename)[0] + '.json'
            json_path = os.path.join(output_dir, json_filename)
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(blocks, f, ensure_ascii=False, indent=2)
                print(f"  - Success: Extracted {len(blocks)} blocks -> '{json_filename}'")
            except Exception as e:
                print(f"  - FAILED: Could not write JSON for {pdf_filename}. Error: {e}")
        else:
            print(f"  - SKIPPING: No text blocks extracted from '{pdf_filename}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract text blocks from PDFs using style and semantic keyword breaks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing PDF files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the feature-rich JSON output.")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)

