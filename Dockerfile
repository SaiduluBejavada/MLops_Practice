# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file if you have one
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files
COPY scripts/ .

# Expose the port the app runs on
EXPOSE 50702

# Command to run the application
CMD ["python", "Bank_Dataset.py"]
