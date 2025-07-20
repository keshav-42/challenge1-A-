import os
import json
import argparse
from collections import Counter
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LAParams

def extract_and_compute_features(pdf_path):
    """
    Extracts styled text blocks from a PDF and computes structural features.

    This single-pass function parses a PDF to extract text blocks and immediately
    computes features like boldness, centering, and text length. This creates
    a feature-rich dataset ready for analysis.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a
                    text block and contains its properties and computed features.
                    Returns an empty list if the file cannot be processed.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return []

    featured_blocks = []
    
    # Layout parameters to control text block grouping
    laparams = LAParams(char_margin=3.5, line_margin=0.5,detect_vertical=True)

    def find_chars_recursive(layout_obj):
        """Recursively finds all LTChar objects within a layout object."""
        if isinstance(layout_obj, LTChar):
            yield layout_obj
        if hasattr(layout_obj, "_objs"):
            for obj in layout_obj._objs:
                yield from find_chars_recursive(obj)

    try:
        for page_layout in extract_pages(pdf_path, laparams=laparams):
            # Page-level information
            current_page_number = page_layout.pageid
            page_width = page_layout.width
            page_height = page_layout.height

            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    # --- 1. EXTRACT RAW DATA ---
                    x0, y0, x1, y1 = element.bbox
                    width = x1 - x0
                    height = y1 - y0
                    text = element.get_text().strip()

                    if not text:
                        continue # Skip empty blocks

                    # Find most common font style
                    chars = find_chars_recursive(element)
                    font_styles = [(char.fontname, char.size) for char in chars]
                    
                    if font_styles:
                        most_common_style = Counter(font_styles).most_common(1)[0][0]
                        fontname = most_common_style[0]
                        fontsize = round(most_common_style[1], 2)
                    else:
                        fontname = "Unknown"
                        fontsize = 0

                    # --- 2. COMPUTE FEATURES ---
                    # Boldness feature
                    is_bold = 1 if "bold" in fontname.lower() else 0
                    
                    # Text length feature
                    text_length = len(text)

                    # Word count feature (NEW)
                    word_count = len(text.split())
                    
                    # Centering feature
                    block_center_x = x0 + (width / 2)
                    page_center_x = page_width / 2
                    centering_offset = abs(block_center_x - page_center_x)

                    # --- 3. STORE COMBINED DATA ---
                    block_data = {
                        'text': text,
                        'x0': round(x0, 2),
                        'y0_top': round(y1, 2),
                        'fontname': fontname,
                        'fontsize': fontsize,
                        'width': round(width, 2),
                        'height': round(height, 2),
                        'page_number': current_page_number,
                        'page_width': round(page_width, 2),
                        'page_height': round(page_height, 2),
                        'features': {
                            'font_size': fontsize,
                            'indentation': round(x0, 2),
                            'is_bold': is_bold,
                            'centering_offset': round(centering_offset, 2),
                            'text_length': text_length,
                            'word_count':word_count,
                            'y_coordinate': round(y1, 2)
                        }
                    }
                    featured_blocks.append(block_data)
    except Exception as e:
        print(f"Could not process {pdf_path}. Error: {e}")
        return []

    return featured_blocks

def process_directory(input_dir, output_dir):
    """
    Processes all PDF files in a directory, computes features, and saves
    the enriched data as JSON files in an output directory.
    """
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
        print(f"\nProcessing '{pdf_path}'...")

        blocks = extract_and_compute_features(pdf_path)

        if blocks:
            json_filename = os.path.splitext(pdf_filename)[0] + '.json'
            json_path = os.path.join(output_dir, json_filename)

            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(blocks, f, ensure_ascii=False, indent=4)
                print(f"Successfully extracted and featured {len(blocks)} blocks -> '{json_path}'")
            except Exception as e:
                print(f"Could not write to JSON file {json_path}. Error: {e}")
        else:
            print(f"No text blocks extracted from '{pdf_path}'.")


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract text blocks from PDFs and compute structural features.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input_dir', type=str, required=True, help="Directory containing PDF files."
    )
    parser.add_argument(
        '--output_dir', type=str, required=True, help="Directory to save the feature-rich JSON output."
    )

    args = parser.parse_args()
    
    # --- How to Run ---
    # 1. Save this script (e.g., `process_pdfs.py`).
    # 2. Create a folder `input_pdfs` and place your PDFs inside.
    # 3. Create an empty folder `output_features`.
    # 4. Run from your terminal:
    #
    #    python process_pdfs.py --input_dir input_pdfs --output_dir output_features
    
    process_directory(args.input_dir, args.output_dir)
