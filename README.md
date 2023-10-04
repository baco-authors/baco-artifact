# The BACO Artifact

This is a repo to build and test BaCO and to recreate the results in the paper BaCO: A Fast and Portable Bayesian Compiler Optimization Framework. The repo sets up the user to easily run the two main benchmarks of the paper: TACO and RISE/ELEVATE and subsequently plot the results. Each benchmark, as well as the plotting scipt, is built in its own separate docker file and they are then linked via a common docker volume. Note that we have omitted HPVM2FPGA. It requires over 60 GB of disk space to install and produces several hundred GBs of auxiliary files when run.


# Installation
Begin with cloning the repository
```git clone https://github.com/baco-authors/baco-artifact```. 
Subsequently, the easiest way to build the three docker images is to run ```./install_all.sh```. 

For the interested reader, in what follows, we describe the building of each individual image.

### Building BACO
Both benchmarks begin with building BaCO using pip and setuptools. The baselines (ytopt and opentuner) are subsequently installed using the ```./install_baselines.sh``` command from the BaCO repository. Lastly, the customized packages and the library *libcconfigspace* for ytopt, are linked to the python setup. 

### Building TACO
TACO is built into a container named *taco*. In addition to installing BaCO and downloading and building TACO, the docker image will also download the required tensors needed for the experiments. 

### Building RISE/ELEVATE


### Building the plotting script
The plotting scripts in _plots_ are built into a container named *plot* that downloads and runs python 3.8.


### Setting up the results and plots Volume
This artifact makes use of a shared Volume between the three Docker images to access results files and save plots. This volume will be called *baco_data* and is set up as ```sudo docker volume create baco_data```.

# Running the benchmarks
Running the complete set of experiments reported in the paper takes several hundred compute hours and as such, we have provided the option to run fewer repetitions to lessen the computational load. In the paper, each experiment is run 30 times, the trends we show in the paper are clearly seen with fewer samples as well. 

Each benchmark is run using the docker ```run``` functionality. Note that when running the benchmarks, the output directory needs to be bound to the volume *baco_data* for the plot script to be able to access them.

### Running TACO



### Running RISE/ELEVATE

# Running the plot scripts



