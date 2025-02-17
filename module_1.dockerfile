# Use an official Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and data folder into the container
COPY script.py .
COPY data/ ./data/

# Expose a port if your app provides an HTTP endpoint (e.g., for health checks)
EXPOSE 5000

# Set the default command to run your script.
# If your app becomes a web service (e.g., with Flask), it should listen on port 5000.
CMD ["python", "script.py"]
