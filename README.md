# The BACO Artifact

This is a repo to build and test BaCO and to recreate the results in the paper BaCO: A Fast and Portable Bayesian Compiler Optimization Framework. The repo sets up the user to easily run the two main benchmarks of the paper: TACO and RISE/ELEVATE and subsequently plot the results. Each benchmark, as well as the plotting scipt, is built in its own separate docker file. Note that we have omitted HPVM2FPGA. It requires over 60 GB of disk space to install and produces several hundred GBs of auxiliary files when run.


# Installation
Begin with cloning the repository
```shell
git clone https://github.com/baco-authors/baco-artifact
cd baco-artifact
``` 
Subsequently, the easiest way to build the three docker images is to run
```shell
./install_all.sh
``` 

For the interested reader, in what follows, we describe the building of each individual image.

### Building BACO
Both benchmarks begin with building BaCO using pip and setuptools. The baselines (ytopt and opentuner) are subsequently installed using the ```./install_baselines.sh``` command from the BaCO repository. Lastly, the customized packages and the library *libcconfigspace* for ytopt, are linked to the python setup. 

### Building TACO
TACO is built into a container named *taco*. In addition to installing BaCO and downloading and building TACO, the docker image will also download the required tensors needed for the experiments. 

### Building RISE/ELEVATE
RISE/ELEVATE is built into a container named *rise*. In the docker image RISE/ELEVATE, BaCO and an OpenCL runtime environment for Nvidia GPUs is set up. Alternatively, a version using a CPU-based runtime environment can be built. 

### Building the plotting script
The plotting scripts in _plots_ are built into a container named *plot* that downloads and runs python 3.8. The plots are made with matplotlib and seaborn.

# Running the benchmarks
Running the complete set of experiments reported in the paper takes several hundred compute hours and as such, we have provided the option to run fewer repetitions to lessen the computational load. In the paper, each experiment is run 30 times, the trends we show in the paper are clearly seen with fewer samples as well. 

Each benchmark is run using the docker ```run``` functionality. We bind the outputfolder of the benchmark to the host *results* folder to access the csv files outside the container. 

### Running TACO (2 human minutes + 10 compute-hours)

```shell
# start taco run script and run experiments
docker run -it -v "$(pwd)"/results:/home/taco/build/experiments taco
```

Collect the results using a different terminal
```shell
# get id of docker 
docker ps

# copy results 
docker cp <docker_id>:/home/taco/build/experiments results  
```

### Running RISE/ELEVATE (2 human minutes + 10 compute-hours)

```shell
# start interactive bash session
docker run --gpus all -it "$(pwd)"/results:/home/shine/artifact/results rise bash

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
docker cp <docker_id>:/home/shine/artifact/results/rise results  
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
docker cp <docker_id>:/home/shine/artifact/results/rise results  
```

# Running the plot scripts (2 compute minutes)

There are two plot scripts in the plot docker image. The first script plots the main results (Figure 5 and 7 and Figure 11 [app]). The second script plots the ablation graphs (Figures 8, 9, and 10). The plot scrips are run as 
```shell
docker run -v "$(pwd)"/results:/app/results -v "$(pwd)"/plots:/app/plots plot
```
This will plot the data you have in results on host.

The generated plots will be called *bar_plot.pdf* (Fig. 5), *line_plot_small.pdf* (Fig. 7), *line_plot_large.pdf* (Fig. 11), *ablation_spmm_1* (Fig. 8), *ablation_spmm_2* (Fig. 9), and *ablation_rise_1* (Fig. 10).
