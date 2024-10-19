# Use the official Python 3.12 slim image from Docker Hub
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy your current directory to the container
COPY . .

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Install dependencies (if pyproject.toml exists)
RUN poetry install

RUN poetry lock 

# Keep the container running for interactive development
CMD ["tail", "-f", "/dev/null"]
