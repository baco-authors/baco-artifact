# Use a smaller base image
FROM ubuntu:20.04
# FROM pytorch/pytorch

ENV SUITESPARSE_PATH=/home/data/suitesparse \
  FROST_PATH=/home/data/FROSTT \
  DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /home

# Install dependencies
RUN apt-get update && apt-get install -y \
  apt-utils \
  build-essential \
  python3.10 \
  python3-pip \
  cmake \
  git \
  curl \
  unzip \
  libomp-dev \
  zlib1g-dev \
  libssl-dev \
  ruby \
  autoconf automake libtool libgsl-dev \
  pkg-config \
  libopenmpi-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# install baco
RUN ls
RUN \
  cd /home && \
  git clone https://github.com/baco-authors/baco.git && \
  cd baco && \
  # apt-get install -y pip && \
  pip3 install --upgrade pip && \
  pip3 install -e .

# install baselines
RUN \
  cd /home/baco && \
  bash install_baselines.sh && cd /home && \
  pip3 install scikit-learn==1.0.2

# set environment variables
ENV PYTHONPATH="/home/baco:/home/baco/extra_packages/CCS/bindings/python/"
ENV LD_LIBRARY_PATH="/usr/local/lib"

RUN git clone https://github.com/lrubens/taco.git && cd taco && git checkout grpc && cd /home/taco && mkdir -p /home/data/FROSTT/facebook && cp facebook.tns /home/data/FROSTT/facebook 

RUN ls && cd .
# Create build directory, build the project, and clean up
RUN cd /home/taco && mkdir build && \
  cd build &&   
  cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON .. && \
  make -j8 && \
  mv ../cpp_taco_* . && \
  cd ..

# Here we assume that "cpp_taco_*" files are meant to stay in "/app/build". 
# If that's not the case, please adjust the path accordingly.

# COPY download_frostt.sh 
RUN apt-get update && apt-get install numactl wget -y
ENV HYPERMAPPER_HOME=/home/baco
RUN cd /home/taco && ls && chmod +x download_suitesparse.sh && chmod +x download_frostt.sh && chmod +x extract.sh
RUN /home/taco/download_suitesparse.sh && /home/taco/download_frostt.sh && /home/taco/extract.sh
RUN cd /home/data/suitesparse/ && wget https://sparse.tamu.edu/MM/SNAP/amazon0312.tar.gz && tar -xzf amazon0312.tar.gz && cd -
WORKDIR /home/taco

ENV OMP_PROC_BIND=true
COPY taco_run.sh build/
RUN chmod +x build/taco_run.sh
WORKDIR /home/taco/build
ENTRYPOINT ["/home/taco/build/taco_run.sh"]


