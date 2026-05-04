# Use official Python image
FROM python:3.10-slim

# Install ffmpeg (CRITICAL for your video processing)
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy all your GitHub files into the container
COPY . .

# Install your Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (Hugging Face requires this specific port)
EXPOSE 7860

# Command to run the app (Assuming your app is Flask and the file is main.py)
# If your file is app.py, change "main:app" to "app:app"
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", 400", "main:app"]
