# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference script and model
COPY inference.py .
COPY model.h5 .

# Expose port for API
EXPOSE 5000

# Command to run FastAPI app
CMD ["python", "inference.py"]
