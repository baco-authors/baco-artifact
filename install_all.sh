#!/bin/bash

docker build -t taco ./taco
docker build -t rise ./rise_elevate/rise_elevate_gpu
docker build -t plot ./plotting

mkdir -p plots
mkdir -p results
mkdir -p results/taco
