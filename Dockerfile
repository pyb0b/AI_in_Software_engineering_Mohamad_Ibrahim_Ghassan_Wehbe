# Use a lightweight Python base image
FROM python:3.12.2

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install Poetry and dependencies
RUN pip install pipx && pipx install poetry
RUN ln -s /root/.local/bin/poetry /usr/local/bin/poetry && poetry install

# Expose the port your FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI app
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["poetry", "run", "uvicorn", "src.app:app"]
CMD ["poetry", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
