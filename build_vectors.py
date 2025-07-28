import os
import json
import argparse

def create_feature_vectors(input_json_path):
    """
    Loads text blocks from a JSON file and constructs a feature vector for each.

    Args:
        input_json_path (str): Path to the JSON file containing extracted blocks
                               with a 'features' dictionary.

    Returns:
        list[dict]: A list of the original block dictionaries, with a new
                    'feature_vector' key added to each. Returns an empty
                    list if the file cannot be processed.
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not read or parse JSON file {input_json_path}. Error: {e}")
        return []

    blocks_with_vectors = []
    for block in blocks:
        features = block.get('features', {})
        
        # Ensure all required features are present, with defaults if not
        font_size = features.get('font_size', 0)
        indentation = features.get('indentation', 0)
        is_bold = features.get('is_bold', 0)
        centering_offset = features.get('centering_offset', 999)
        word_count = features.get('word_count', 0)
        y_coordinate = features.get('y_coordinate', 0)

        # --- New Feature: Indent-to-Font Ratio ---
        # Handle potential division by zero if font_size is 0
        if font_size > 0:
            indent_to_font_ratio = indentation / font_size
        else:
            indent_to_font_ratio = 0
            
        # --- Construct the Feature Vector ---
        feature_vector = [
            font_size,
            indentation,
            is_bold,
            centering_offset,
            word_count,
            y_coordinate,
            round(indent_to_font_ratio, 3) # Round for consistency
        ]
        
        # Add the vector to the block's data
        block['feature_vector'] = feature_vector
        blocks_with_vectors.append(block)
        
    return blocks_with_vectors

def process_directory(input_dir, output_dir):
    """
    Processes all JSON files in an input directory, creates feature vectors,
    and saves the results to a new output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input JSON directory: {os.path.abspath(input_dir)}")
    print(f"Output (features) directory: {os.path.abspath(output_dir)}\n")

    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the input directory.")
        return

    print(f"Found {len(json_files)} JSON file(s) to process...")

    for json_filename in json_files:
        json_path = os.path.join(input_dir, json_filename)
        output_path = os.path.join(output_dir, json_filename)
        
        print(f"\nProcessing '{json_path}'...")

        # Create the feature vectors for the current file
        featured_data = create_feature_vectors(json_path)

        if featured_data:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(featured_data, f, ensure_ascii=False, indent=4)
                print(f"  - Success: Built vectors for {len(featured_data)} blocks -> '{output_path}'")
            except Exception as e:
                print(f"  - FAILED: Could not write to output file {output_path}. Error: {e}")
        else:
            print(f"  - SKIPPING: No data processed for '{json_path}'.")

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Construct feature vectors from JSON files of text blocks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory containing the JSON files from the feature extraction step."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory where the JSON files with feature vectors will be saved."
    )

    args = parser.parse_args()
    
    # --- How to Run ---
    # 1. Save this script as `build_vectors.py`.
    # 2. You should have two folders:
    #    - `output_features`: Contains the JSON outputs from the first script.
    #    - `output_vectors`: An empty folder for this script's output.
    # 3. Run from your terminal:
    #
    #    python build_vectors.py --input_dir output --output_dir output_vectors

    process_directory(args.input_dir, args.output_dir)
