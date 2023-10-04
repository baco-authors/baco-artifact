# The BACO Artifact

This is a repo to build and test BaCO and to recreate the results in the paper BaCO: A Fast and Portable Bayesian Compiler Optimization Framework. The repo sets up the user to easily run the two main benchmarks of the paper: TACO and RISE/ELEVATE and subsequently plot the results. Each benchmark, as well as the plotting scipts, is built in its own separate docker file and they are linked via a common Volume. 


# Installation
Begin with cloning the repository
```git clone https://github.com/baco-authors/baco-artifact```. 
Subsequently, the easiest way to build the three docker images is to run ```./install_all.sh```. 

For the interested reader, in what follows, we describe the building of each individual image.


### Building TACO


### Building RISE/ELEVATE


### Building the plotting script
The plotting scripts in _plots_ are built into a container named *plot* that downloads and runs python 3.8.


# Setting up the results Volume


# Running the benchmarks

### Running TACO

### Running RISE/ELEVATE

# Running the plot scripts



