# Use the base image of Python 3.9 on Ubuntu
FROM python:3.9-slim

# Update Ubuntu package index and install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libffi-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /flask

# Copy the required files to run your application
COPY ./requirements.txt /flask/requirements.txt


# Upgrade pip
RUN pip install --upgrade pip

RUN pip install geopandas
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY ./blueprints /flask/blueprints
COPY ./models /flask/models
COPY ./scripts /flask/scripts
COPY ./main.py /flask/main.py


# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "/flask/main.py"]