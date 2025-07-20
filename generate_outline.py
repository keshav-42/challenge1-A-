import os
import json
import argparse

def generate_outline(input_json_path):
    """
    Loads labeled text blocks and generates a structured outline JSON.

    Args:
        input_json_path (str): Path to the JSON file containing labeled blocks.

    Returns:
        dict: A dictionary representing the structured outline. Returns None
              if processing fails or no title/headings are found.
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not read or parse JSON file {input_json_path}. Error: {e}")
        return None

    if not blocks:
        return None

    # Sort blocks by their position in the document to ensure correct order
    blocks.sort(key=lambda b: (b.get('page_number', 0), -b.get('y0_top', 0)))

    # Find the document title
    doc_title = "Untitled Document" # Default title
    for block in blocks:
        if block.get('predicted_label') == 'Title':
            doc_title = block.get('text', 'Untitled Document')
            break # Use the first title found

    # Extract all headings (H1, H2, H3, etc.)
    outline_items = []
    heading_labels = {'H1', 'H2', 'H3'}
    for block in blocks:
        label = block.get('predicted_label')
        if label in heading_labels:
            outline_item = {
                "level": label,
                "text": block.get('text', ''),
                "page": block.get('page_number', 0)
            }
            outline_items.append(outline_item)
            
    # If no headings were found, we can consider the outline empty
    if not outline_items:
        print("Warning: No H1, H2, or H3 headings found to create an outline.")
        
    # Construct the final outline object
    final_outline = {
        "title": doc_title,
        "outline": outline_items
    }

    return final_outline


def process_directory(input_dir, output_dir):
    """
    Processes all labeled JSON files in a directory and generates structured
    outlines for each.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input (labeled) directory: {os.path.abspath(input_dir)}")
    print(f"Output (outline) directory: {os.path.abspath(output_dir)}\n")

    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the input directory.")
        return

    print(f"Found {len(json_files)} JSON file(s) to process...")

    for json_filename in json_files:
        json_path = os.path.join(input_dir, json_filename)
        output_path = os.path.join(output_dir, json_filename)
        
        print(f"\nProcessing '{json_path}'...")
        
        outline_data = generate_outline(json_path)

        if outline_data:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(outline_data, f, ensure_ascii=False, indent=2)
                print(f"  - Success: Generated outline -> '{output_path}'")
            except Exception as e:
                print(f"  - FAILED: Could not write to output file {output_path}. Error: {e}")
        else:
            print(f"  - SKIPPING: No outline generated for '{json_path}'.")

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate structured outlines from labeled text blocks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory containing the labeled JSON files from the clustering step."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory where the final outline JSON files will be saved."
    )

    args = parser.parse_args()
    
    # --- How to Run ---
    # 1. Save this script as `generate_outline.py`.
    # 2. You should have two folders:
    #    - `output_labeled`: Contains the JSON outputs from the clustering script.
    #    - `output_outlines`: An empty folder for this script's final output.
    # 3. Run from your terminal:
    #
    #   python generate_outline.py --input_dir output_labeled --output_dir output_outlines 

    process_directory(args.input_dir, args.output_dir)
