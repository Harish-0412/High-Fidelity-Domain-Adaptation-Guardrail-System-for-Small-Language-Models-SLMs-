FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building Python packages and health checks
RUN apt-get update && apt-get install -y curl gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and source files
COPY . .

# Install dependencies and the application itself
RUN pip install --default-timeout=1000 --no-cache-dir -e .[runtime]

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
