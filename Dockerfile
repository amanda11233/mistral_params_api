FROM nvidia/cuda:12.1.1-base-ubuntu20.04
# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
# Install the CUDA toolkit
RUN apt-get update -y && apt-get install -y cuda-toolkit
# Set the LD_LIBRARY_PATH environment variable
# ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH    
# Update the package list and install necessary dependencies
RUN apt-get update -y && apt-get install -y software-properties-common wget
# Add the deadsnakes PPA to access Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa
# Update the package list again and install Python 3.11
RUN apt-get update -y && apt-get install -y python3.11 python3.11-distutils

#Installing necessary libraries 
RUN apt-get install -y python3-dev libffi-dev libssl-dev gcc libcap-dev 

# Get pip for Python3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

#Installing VIM
RUN apt-get install vim

# Verify pip version (this should use the freshly installed pip)
RUN pip --version
# Clean up the apt cache to reduce image size
RUN apt-get clean
# Set the default Python version to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
# Verify Python installation
RUN python3 --version

#Final update
RUN apt-get update -y

# Define the working directory
WORKDIR /mistral_param
# Add your application code to the container
COPY . /mistral_param

# Copy the SSL certificates
# COPY ./docker/localhost.crt ./mistral_param/localhost.crt
# COPY ./docker/localhost.key ./mistral_param/localhost.key
# COPY ./docker/localhost.crt ./localhost.crt
# COPY ./docker/localhost.key ./localhost.key

# Install Python dependencies from requirements.txt with a timeout of 120 seconds
RUN pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -U bitsandbytes
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m bitsandbytes
# Expose port 8000 for FastAPI server
EXPOSE 8000
# Start the FastAPI server
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]