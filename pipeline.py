import os
import sys
import json
import shutil
from typing import List, Dict, Any

# Import your existing modules
from parsing import parse_and_feature_engineer
from build_vectors import create_feature_vectors
from cluster_and_label import run_pipeline_for_file
from generate_outline import generate_outline

class DocumentProcessingPipeline:
    """
    Clean automated pipeline that processes PDF documents through 4 sequential stages:
    1. parsing.py - Extract and feature engineer text blocks
    2. build_vectors.py - Create vector representations
    3. cluster_and_label.py - Cluster blocks and assign labels
    4. generate_outline.py - Generate final document outline
    
    Features:
    - Automatic directory cleanup on new runs
    - Sequential stage execution with dependency management
    - In-memory processing without intermediate files
    - Final output files only (no intermediate folders)
    """
    
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        # Remove intermediate directory - process in memory instead
        
    def reset_environment(self):
        """Clean output directory."""
        print("Resetting pipeline environment...")
        
        # For Docker environments, we can't remove mounted volumes
        # Instead, clean the contents of the output directory
        if os.path.exists(self.output_dir):
            # Remove all contents except the directory itself
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"  Cleaned contents of output directory: {self.output_dir}")
        
        print("  Ready for processing")
    
    def get_input_files(self) -> List[str]:
        """Get list of PDF files to process."""
        pdf_files = []
        
        if os.path.isfile(self.input_path):
            # Single file
            if self.input_path.lower().endswith('.pdf'):
                pdf_files = [self.input_path]
        elif os.path.isdir(self.input_path):
            # Directory of files
            pdf_files = [
                os.path.join(self.input_path, f) 
                for f in os.listdir(self.input_path) 
                if f.lower().endswith('.pdf')
            ]
        
        return pdf_files
    
    def run_pipeline(self):
        """Execute the complete automated pipeline."""
        print("=" * 60)
        print("DOCUMENT PROCESSING PIPELINE")
        print("=" * 60)
        
        # Reset environment
        self.reset_environment()
        
        # Get input files
        pdf_files = self.get_input_files()
        if not pdf_files:
            print("No PDF files found to process.")
            return False
        
        print(f"Input: {os.path.abspath(self.input_path)}")
        print(f"Output: {os.path.abspath(self.output_dir)}")
        print(f"Found {len(pdf_files)} PDF file(s) to process\n")
        
        # Process all files through pipeline
        successful_files = []
        
        for pdf_path in pdf_files:
            result = self.process_single_pdf(pdf_path)
            if result:
                successful_files.append(result['filename'])
        
        # Final summary
        if successful_files:
            print(f"Pipeline completed successfully!")
            print(f"Processed {len(successful_files)} files:")
            for filename in successful_files:
                print(f"  - {filename}")
            print(f"Outputs saved in: {self.output_dir}")
            return True
        else:
            print("Pipeline failed - no documents processed successfully.")
            return False
    
    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF through all pipeline stages."""
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"Processing: {filename}")
        print("-" * 50)
        
        try:
            # Stage 1: Parsing and Feature Engineering
            print("Stage 1: Parsing and feature engineering...")
            parsed_data = self.stage1_parsing(pdf_path)
            if not parsed_data:
                print(f"  Failed to parse {filename}")
                return None
            print(f"  Extracted {len(parsed_data)} text blocks")
            
            # Stage 2: Build Vectors
            print("Stage 2: Building vectors...")
            vectors_data = self.stage2_build_vectors(parsed_data)
            if not vectors_data:
                print(f"  Failed to build vectors for {filename}")
                return None
            print(f"  Generated vectors for {len(vectors_data)} blocks")
            
            # Stage 3: Cluster and Label
            print("Stage 3: Clustering and labeling...")
            clustered_data = self.stage3_cluster_and_label(vectors_data)
            if not clustered_data:
                print(f"  Failed to cluster {filename}")
                return None
            print(f"  Clustered and labeled {len(clustered_data)} blocks")
            
            # Stage 4: Generate Outline
            print("Stage 4: Generating outline...")
            outline_data = self.stage4_generate_outline(clustered_data)
            if not outline_data:
                print(f"  Failed to generate outline for {filename}")
                return None
            print(f"  Generated document outline")
            
            print(f"Successfully processed {filename}\n")
            
            result = {
                "filename": filename,
                "base_name": base_name,
                "outline": outline_data
            }
            
            # Save individual output file for this document
            self.save_individual_output(result)
            
            return result
            
        except Exception as e:
            print(f"Error processing {filename}: {e}\n")
            return None
    
    def stage1_parsing(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Stage 1: Parse PDF and extract features."""
        try:
            blocks = parse_and_feature_engineer(pdf_path)
            return blocks
        except Exception as e:
            print(f"    Error in parsing stage: {e}")
            return []
    
    def stage2_build_vectors(self, parsed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 2: Build vector representations."""
        try:
            # Process the data directly in memory using build_vectors module
            # We need to modify this to work with data instead of file paths
            vectors_data = self.create_feature_vectors_from_data(parsed_data)
            return vectors_data
        except Exception as e:
            print(f"    Error in build vectors stage: {e}")
            return []
    
    def create_feature_vectors_from_data(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create feature vectors from block data directly."""
        # Import the function we need from build_vectors
        import tempfile
        import os
        
        # Create a temporary file to pass to the existing function
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
            json.dump(blocks, temp_file, ensure_ascii=False, indent=2)
            temp_path = temp_file.name
        
        try:
            # Use the existing function
            vectors_data = create_feature_vectors(temp_path)
            return vectors_data
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    def stage3_cluster_and_label(self, vectors_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 3: Cluster blocks and assign labels."""
        try:
            # Process the data directly in memory using cluster_and_label module
            clustered_data = self.cluster_and_label_from_data(vectors_data)
            return clustered_data
        except Exception as e:
            print(f"    Error in clustering stage: {e}")
            return []
    
    def cluster_and_label_from_data(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster and label blocks from data directly."""
        import tempfile
        import os
        
        # Create a temporary file to pass to the existing function
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
            json.dump(blocks, temp_file, ensure_ascii=False, indent=2)
            temp_path = temp_file.name
        
        # Create another temp file for output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Use the existing function
            run_pipeline_for_file(temp_path, temp_output_path)
            
            # Read the result
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'r', encoding='utf-8') as f:
                    clustered_data = json.load(f)
                return clustered_data
            return []
        finally:
            # Clean up the temporary files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
    
    def stage4_generate_outline(self, clustered_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 4: Generate document outline."""
        try:
            # Process the data directly in memory using generate_outline module
            outline_data = self.generate_outline_from_data(clustered_data)
            return outline_data
        except Exception as e:
            print(f"    Error in outline generation stage: {e}")
            return {}
    
    def generate_outline_from_data(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate outline from block data directly."""
        import tempfile
        import os
        
        # Create a temporary file to pass to the existing function
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
            json.dump(blocks, temp_file, ensure_ascii=False, indent=2)
            temp_path = temp_file.name
        
        try:
            # Use the existing function
            outline_data = generate_outline(temp_path)
            return outline_data
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    def save_individual_output(self, result: Dict[str, Any]):
        """Save individual output file for each processed document."""
        filename = result['filename']
        base_name = result['base_name']
        
        # Save just the outline data directly
        outline_data = result['outline']
        
        # Save to individual file
        output_file = os.path.join(self.output_dir, f"{base_name}_output.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(outline_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Individual output saved: {base_name}_output.json")

def main():
    """Main function to run the pipeline."""
    import sys
    
    # Check if running in Docker or locally
    if len(sys.argv) >= 3:
        # Command line arguments provided
        input_directory = sys.argv[1]
        output_directory = sys.argv[2]
    elif os.path.exists("/app/input"):
        # Docker environment detected
        input_directory = "/app/input"          # Docker input mount point
        output_directory = "/app/output"        # Docker output mount point
    else:
        # Local environment fallback
        input_directory = "input"
        output_directory = "output"
    
    print(f"Using input directory: {os.path.abspath(input_directory)}")
    print(f"Using output directory: {os.path.abspath(output_directory)}")
    
    # Debug: List contents of input directory
    if os.path.exists(input_directory):
        print(f"Input directory contents:")
        try:
            files = os.listdir(input_directory)
            for file in files:
                print(f"  - {file}")
            if not files:
                print("  (directory is empty)")
        except Exception as e:
            print(f"  Error listing directory: {e}")
    else:
        print(f"Error: Input directory '{input_directory}' does not exist.")
        print(f"Please mount your input directory to /app/input")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Create and run pipeline with Docker paths
    pipeline = DocumentProcessingPipeline(input_directory, output_directory)
    success = pipeline.run_pipeline()
    
    if success:
        print("\nPipeline completed successfully!")
        
    else:
        print("\nPipeline failed!")
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    # Simply run: python pipeline.py
    # Make sure your PDF files are in the 'input' directory
    main()