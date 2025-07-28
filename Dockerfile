# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY *.py ./

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the pipeline
CMD ["python", "pipeline.py"]

# docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
# docker run --rm -v //$(pwd)/input:/app/input -v //$(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier