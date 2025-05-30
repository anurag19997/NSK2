# Use the official miniconda base image
FROM continuumio/miniconda3

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
