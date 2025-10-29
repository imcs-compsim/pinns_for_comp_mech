
# Use the official Miniconda image as the base, as recommended by the documentation for complex dependencies
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Install missing system libraries required by gmsh
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglu1-mesa \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcursor1 \
        libxft2 \
        libxinerama1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the environment file and source code
# The env.yaml includes all necessary packages, including dev dependencies and a specific backend (tensorflow).
COPY . /app

# Create the Conda environment using the env.yaml file, activate it, and install the CompSim-PINN package.
RUN conda env create -f env.yaml
RUN conda run -n CompSim-PINN pip install -e .

# Define the default command to run when the container starts.
CMD ["conda", "run", "-n", "CompSim-PINN", "python", "elasticity_3d/linear_elasticity/block_under_shear.py"]
