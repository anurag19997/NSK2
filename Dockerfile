# Use the official miniconda base image
FROM continuumio/miniconda3
FROM python:3.12-slim

# Install Tkinter and X11 tools
RUN apt-get update && apt-get install -y \
    python3-tk \
    x11-apps \
    && apt-get clean

# Set the DISPLAY env var for runtime (can also set at docker-compose level)
ENV DISPLAY=host.docker.internal:0.0

# Set a working directory
WORKDIR /app

# Copy the environment file and create the conda environment
COPY nsfull_env_nobuilds.yml .
RUN conda env create -f nsfull_env_nobuilds.yml

# Make sure the environment is activated:
ENV PATH /opt/conda/envs/nskfull/bin:$PATH

# Copy your application code into the container
COPY . .

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start the Jupyter Notebook (adjust options as needed)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
