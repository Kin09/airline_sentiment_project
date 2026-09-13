# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for compilation and huggingface models
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first (to leverage Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Flask runs on
EXPOSE 5001

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
