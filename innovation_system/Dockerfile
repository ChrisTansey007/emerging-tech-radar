# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# This is specific to innovation_system requirements
# It assumes the build context is the project root directory.
COPY ./innovation_system/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download NLTK resources
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copy the .env file from project root to /app.
# This makes it available for load_dotenv() if settings.py is also in /app or can find it.
# The .env file itself will be sourced by docker-compose using env_file on the host,
# but copying it here ensures that if the image is run standalone (docker run) AND
# the .env file was part of the build context, it could be found.
# For docker-compose, env_file is the primary way to get .env vars into the container's environment.
# However, load_dotenv() in settings.py needs a .env file to load from within the container's filesystem
# if we are not relying purely on environment variables being set by Docker Compose.
# Let's keep it for flexibility, assuming settings.py might try to load it from /app/.env
COPY .env .

# Copy the entire innovation_system directory into the container at /app/innovation_system
# This makes the path inside the container /app/innovation_system/main/run.py
COPY ./innovation_system/ ./innovation_system/

# Default command to run the application
ENTRYPOINT ["python", "innovation_system/main/run.py"]
# Example of how to set default arguments if ENTRYPOINT is just ["python"]:
# CMD ["innovation_system/main/run.py", "--sectors", "AI", "--force-collect"]
