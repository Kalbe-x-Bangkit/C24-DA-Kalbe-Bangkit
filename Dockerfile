# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim
EXPOSE 8080

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy local code to the container image
WORKDIR /app
COPY . ./

# Run the web service on container startup
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]