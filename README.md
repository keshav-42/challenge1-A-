# PDF Document Processing Pipeline

A comprehensive automated pipeline for processing PDF documents through parsing, vectorization, clustering, and outline generation.

## Approach

This solution implements a 4-stage sequential pipeline that processes PDF documents to extract hierarchical document structure:

1. **Parsing & Feature Engineering** (`parsing.py`)

   - Extracts text blocks from PDF documents using pdfminer.six
   - Performs feature engineering on extracted content (font size, position, styling)
   - Outputs structured JSON with text blocks and metadata

2. **Vector Building** (`build_vectors.py`)

   - Creates feature vectors from parsed text blocks
   - Uses numerical features like font size, position, word count, styling attributes
   - Generates standardized feature representations for machine learning

3. **Clustering & Labeling** (`cluster_and_label.py`)

   - Uses MiniBatchKMeans clustering to group similar text blocks
   - Applies intelligent document-type detection (uniform vs varied styling)
   - Falls back to rule-based classification for uniform documents
   - Assigns hierarchical labels (Title, H1, H2, H3, Body) based on clustering results
   - Includes advanced post-processing for text classification accuracy

4. **Outline Generation** (`generate_outline.py`)
   - Creates hierarchical document outlines from labeled blocks
   - Generates structured representations of document content
   - Produces final JSON output with clean document hierarchy

## Models and Libraries Used

### Core Dependencies

- **Python 3.11**: Base runtime environment
- **pdfminer.six**: PDF parsing and text extraction (primary PDF processor)
- **scikit-learn**: Machine learning algorithms and clustering (MiniBatchKMeans, StandardScaler)
- **numpy**: Numerical computing and array operations
- **matplotlib**: Visualization for clustering analysis
- **seaborn**: Enhanced statistical visualization

### Machine Learning Approach

- **MiniBatchKMeans**: Primary clustering algorithm for grouping text blocks by similarity
- **StandardScaler**: Feature normalization for consistent clustering results
- **Rule-based Classification**: Fallback system for documents with uniform styling
- **Post-processing Pipeline**: Advanced text pattern detection and hierarchy adjustment

### Key Algorithms

- **Font-size based hierarchy**: Uses visual cues to determine heading levels
- **Pattern detection**: Identifies table of contents, references, page numbers
- **Same-page hierarchy adjustment**: Prevents multiple H1 headings on same page
- **Document style analysis**: Automatically chooses between clustering and rule-based approaches

## How to Build and Run

### Prerequisites

- Docker installed on your system
- Input PDF files ready for processing

### Building the Docker Image

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

### Running the Solution

```bash
# For Windows Git Bash:
docker run --rm -v //$(pwd)/input:/app/input -v //$(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier

# For Linux/Mac:
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

### Expected Behavior

The container will:

1. **Automatically scan** `/app/input` directory for PDF files
2. **Process each PDF** through the 4-stage pipeline
3. **Generate individual outputs** in `/app/output` directory:
   - `filename_output.json` for each `filename.pdf`
   - Each output contains the structured document outline
4. **Clean processing**: No intermediate files, only final outputs

### Output Format

Each generated JSON file contains:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Section Title",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Subsection Title",
      "page": 2
    }
  ]
}
```

## Architecture

```
Input PDFs → Parsing → Vectorization → Clustering → Outline → Output JSONs
     ↓           ↓            ↓           ↓          ↓
   /app/input  Stage 1     Stage 2    Stage 3   Stage 4   /app/output
```

## Features

- ✅ **Intelligent Document Analysis**: Automatically detects document styling patterns
- ✅ **Dual Processing Modes**: Clustering for styled documents, rules for uniform documents
- ✅ **Advanced Post-Processing**: Pattern detection for references, TOC, page numbers
- ✅ **Hierarchy Optimization**: Font-size based heading level adjustment
- ✅ **Clean Output**: Only final JSON files, no intermediate directories
- ✅ **Error Handling**: Graceful failure handling per document
- ✅ **Docker Optimized**: Minimal dependencies, efficient containerized execution
- ✅ **Network Isolated**: Runs without external network access

## Technical Highlights

- **Adaptive Pipeline**: Chooses optimal processing strategy based on document characteristics
- **Font-size Intelligence**: Uses visual hierarchy cues for accurate heading detection
- **Pattern Recognition**: Identifies and correctly classifies special text patterns
- **Memory Efficient**: In-memory processing without persistent intermediate files
- **Robust Classification**: Multiple fallback mechanisms ensure reliable document structure extraction
