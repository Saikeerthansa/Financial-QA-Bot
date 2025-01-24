# Use an official lightweight Python image
FROM python:3.9-slim

# Set environment variables to avoid buffering issues and enable best practices
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501

# Set the working directory inside the container
WORKDIR /Financial-QA-Bot

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install  --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Set the Streamlit server to be accessible externally
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_RUN_ON_SAVE=true

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
