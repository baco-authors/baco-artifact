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
RISE/ELEVATE is built into a container named *rise*. In the docker image RISE/ELEVATE, BaCO and an OpenCL runtime environment for Nvidia GPUs is set up. Alternatively, a version using a CPU-based runtime environment can be built. 

### Building the plotting script
The plotting scripts in _plots_ are built into a container named *plot* that downloads and runs python 3.8.


### Setting up the results and plots Volume
Thplot script makes use of a docker Volume to provide the results and extract plots. This volume will be called *baco_data* and is set up as ```sudo docker volume create baco_data```.

# Running the benchmarks
Running the complete set of experiments reported in the paper takes several hundred compute hours and as such, we have provided the option to run fewer repetitions to lessen the computational load. In the paper, each experiment is run 30 times, the trends we show in the paper are clearly seen with fewer samples as well. 

Each benchmark is run using the docker ```run``` functionality. Note that when running the benchmarks, the output directory needs to be bound to the volume *baco_data* for the plot script to be able to access them.

### Running TACO (2 human minutes + 10 compute-hour)

```shell
# start taco run script and run experiments
docker run -it taco
```

Collect the results using a different terminal
```shell
# get id of docker 
docker ps

# copy results 
docker cp <docker_id>:/home/taco/build/experiments .  
```

### Running RISE/ELEVATE (2 human minutes + 10 compute-hour)

```shell
# start interactive bash session
docker run --gpus all -it rise bash

# navigate to repository and run experiments 
cd /home/shine
./run_rise.sh 30
./run_ablation.sh 30
```

Collect the results using a different terminal
```shell
# get id of docker 
docker ps

# copy results 
docker cp <docker_id>:/home/shine/artifact/results/rise .  
```

### Running RISE/ELEVATE - CPU version
If no GPU is available, the benchmarks can be run using an OpenCL CPU runtime environment. However, this can result in different results. 

Preparation 
```shell
# init submodules 
git submodule update --init

# build docker 
docker build -t rise_cpu ./rise_elevate/rise_elevate_cpu
```

Run experiments 
```shell
# start interactive bash session
docker run -it rise_cpu bash

# navigate to repository and run experiments 
cd /home/shine
./run_rise.sh 30
./run_ablation.sh 30
```

Collect the results using a different terminal
```shell
# get id of docker 
docker ps

# copy results 
docker cp <docker_id>:/home/shine/artifact/results/rise .  
```

# Running the plot scripts

There are two plot scripts in the plot docker image. The first script plots the main results (Figure 5 and 7 and Figure 11 in the appendix). The second script plots the ablation graphs (Figures 8,9, and 10). The plot scrips are run as ```docker run plot -v baco_data:/app/results```. This will make the plotting script both read data from and save the plots to the baco_data volume. 

The generated plots will be called *bar_plot.pdf* (Fig. 5), *line_plot_small.pdf* (Fig. 7), *line_plot_large.pdf* (Fig. 11), *ablation_spmm_1* (Fig. 8), *ablation_spmm_2* (Fig. 9), and *ablation_rise_1* (Fig. 10).

To access the data from the container run
```docker run --rm -v baco_data:/tmp_storage -v $(pwd)/output:/output ubuntu sh -c 'cp -r /tmp_storage /output'```.

