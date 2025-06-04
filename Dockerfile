# Use Python 3.11 as base image
FROM python:3.11-slim

# Set build arguments (passed from GitHub Actions)
ARG PYTHON_VERSION=3.11
ARG BUILD_DATE
ARG VERSION

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Add labels for metadata
LABEL org.opencontainers.image.title="Multi-Agent Plan Execute"
LABEL org.opencontainers.image.description="LangGraph-based multi-agent system"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.version="${VERSION}"

# Expose port (adjust if your agent uses a different port)
EXPOSE 8000

# Set the default command to run your agent
# Note: Make sure to set OPENAI_API_KEY environment variable when running the container
CMD ["python", "agent.py"] 