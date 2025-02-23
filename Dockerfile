# Use the official Python image with the desired version
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install TensorFlow explicitly
RUN pip install --upgrade pip
RUN pip install tensorflow==2.16.2

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .
