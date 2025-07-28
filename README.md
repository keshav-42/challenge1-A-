# PDF Document Processing Pipeline

A comprehensive automated pipeline for processing PDF documents through parsing, vectorization, clustering, and outline generation.

## Approach

This solution implements a 4-stage sequential pipeline:

1. **Parsing & Feature Engineering** (`parsing.py`)

   - Extracts text blocks from PDF documents
   - Performs feature engineering on extracted content
   - Outputs structured JSON with text blocks and metadata

2. **Vector Building** (`build_vectors.py`)

   - Creates feature vectors from parsed text blocks
   - Uses NLP techniques for text representation
   - Generates embeddings for downstream processing

3. **Clustering & Labeling** (`cluster_and_label.py`)

   - Clusters similar text blocks using machine learning
   - Assigns meaningful labels to clustered content
   - Organizes document structure semantically

4. **Outline Generation** (`generate_outline.py`)
   - Creates hierarchical document outlines
   - Generates structured representations of document content
   - Produces final JSON output with document hierarchy

## Models and Libraries Used

### Core Dependencies

- **Python 3.11**: Base runtime environment
- **NumPy & Pandas**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning algorithms and clustering
- **SciPy**: Scientific computing utilities

### PDF Processing

- **PyPDF2**: PDF parsing and text extraction
- **pdfplumber**: Advanced PDF content extraction
- **pdf2image**: PDF to image conversion
- **Pillow**: Image processing

### Natural Language Processing

- **NLTK**: Natural language toolkit for text processing
- **spaCy**: Advanced NLP pipeline
- **transformers**: Pre-trained language models
- **sentence-transformers**: Semantic text embeddings
- **PyTorch**: Deep learning framework

### Machine Learning & Clustering

- **UMAP**: Dimensionality reduction
- **HDBSCAN**: Hierarchical density-based clustering
- **OpenCV**: Computer vision operations

### System Dependencies

- **Poppler**: PDF rendering utilities
- **Tesseract OCR**: Optical character recognition

## Architecture

```
Input PDFs → Parsing → Vectorization → Clustering → Outline → Output JSONs
     ↓           ↓            ↓           ↓          ↓
   /app/input  Stage 1     Stage 2    Stage 3   Stage 4   /app/output
```

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
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

### Expected Behavior

The container will:

1. **Automatically scan** `/app/input` directory for PDF files
2. **Process each PDF** through the 4-stage pipeline
3. **Generate individual outputs** in `/app/output` directory:
   - `filename_output.json` for each `filename.pdf`
   - Each output contains the structured document outline

### Directory Structure

```
/app/
├── input/              # Mounted input directory (your PDFs)
├── output/             # Mounted output directory (generated JSONs)
├── pipeline.py         # Main pipeline orchestrator
├── parsing.py          # Stage 1: PDF parsing
├── build_vectors.py    # Stage 2: Vector generation
├── cluster_and_label.py # Stage 3: Clustering & labeling
├── generate_outline.py # Stage 4: Outline generation
└── requirements.txt    # Python dependencies
```

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

## Features

- ✅ **Clean Environment**: Automatically resets on each run
- ✅ **Error Handling**: Graceful failure handling per document
- ✅ **Progress Tracking**: Clear stage-by-stage output
- ✅ **Individual Outputs**: Separate JSON for each PDF
- ✅ **Docker Optimized**: Efficient containerized execution
- ✅ **Network Isolated**: Runs without external network access

## Troubleshooting

### Common Issues

1. **No output files**: Check that PDFs are valid and in the input directory
2. **Processing failures**: Verify PDF files are not corrupted or password-protected
3. **Memory issues**: Large PDFs may require more container memory

### Logs

The container provides detailed logging for each processing stage, making it easy to identify and debug issues.
