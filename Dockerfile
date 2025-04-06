FROM python:3.12

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set an environment variable for debug mode
ENV DEBUG_MODE=True

# Expose the port your backend will run on (e.g., 8080)
EXPOSE 8080

# Command to run the backend
CMD ["python", "./app.py"]