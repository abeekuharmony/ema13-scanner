FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scanner/ scanner/
COPY main.py .

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

# Unbuffered output for real-time logs
CMD ["python", "-u", "main.py"]
