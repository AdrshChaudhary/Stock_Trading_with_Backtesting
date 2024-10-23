# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies for TA-Lib and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make \
    libffi-dev \
    libssl-dev \
    wget \
    tar \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and build TA-Lib from source
RUN mkdir -p /tmp/ta-lib && \
    cd /tmp/ta-lib && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    rm -rf /tmp/ta-lib

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the container
COPY . .

# Expose the port Dash will run on
EXPOSE 8050

# Set environment variables for Docker
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]
