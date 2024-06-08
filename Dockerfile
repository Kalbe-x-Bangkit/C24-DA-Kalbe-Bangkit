# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim
EXPOSE 8080

# Install pip and any other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy local code to the container image
WORKDIR /app
COPY . ./

# Run the web service on container startup
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
