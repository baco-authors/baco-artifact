#!/bin/bash


NUM_ITER=$1
echo "Running taco-taco_dse for $NUM_ITER iterations"

if [ "$NUM_ITER" -eq "0" ]; then
  echo "0 iterations specified"
else

  pip3 install scikit-learn==1.0.2

  # ablation
  for mat in filter3D email-Enron amazon0312; do
    for i in `eval echo {1..$NUM_ITER}`;do
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -t simple -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m ytopt -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -t hamming -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -t kendall -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -t naive -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -t nms -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -t nolog -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -t nolsp -o SpMM
    done
  done

  #SpMM
  for mat in cage12 scircuit laminar_duct3D; do
    for i in `eval echo {1..$NUM_ITER}`;do
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m bayesian_optimization -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m opentuner -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m ytopt_ccs -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m random_sampling -o SpMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m embedding_random_sampling -o SpMM
    done
  done



  #SDDMM
  for mat in email-Enron ACTIVSg10K Goodwin_040; do
    for i in `eval echo {1..$NUM_ITER}`;do
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m bayesian_optimization -o SDDMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m opentuner -o SDDMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m ytopt_ccs -o SDDMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m random_sampling -o SDDMM
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m embedding_random_sampling -o SDDMM
    done
  done


  #SPMV
  for mat in laminar_duct3D cage12 filter3D; do
    for i in `eval echo {1..$NUM_ITER}`;do
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m bayesian_optimization -o SpMV
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m opentuner -o SpMV
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m ytopt_ccs -o SpMV
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m random_sampling -o SpMV
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -m embedding_random_sampling -o  SpMV
    done
  done


  #TTV
  for mat in facebook uber3; do
    for i in `eval echo {1..$NUM_ITER}`;do
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m bayesian_optimization -o TTV
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m opentuner -o TTV -t ot
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m ytopt_ccs -o TTV
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m random_sampling -o TTV
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m embedding_random_sampling -o TTV
    done
  done


  #MTTKRP
  for mat in uber nips chicago; do
    for i in `eval echo {1..$NUM_ITER}`;do
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m bayesian_optimization -o MTTKRP
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m opentuner -o MTTKRP
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m ytopt_ccs -o MTTKRP
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m random_sampling -o MTTKRP
      ./bin/taco-taco_dse -mat $mat/$mat.tns -n 10 -c $i -m embedding_random_sampling -o MTTKRP
    done
  done



  pip3 install scikit-learn==1.3.0
  for mat in filter3D email-Enron amazon0312; do
    for i in `eval echo {1..$NUM_ITER}`;do
      ./bin/taco-taco_dse -mat $mat/$mat.mtx -n 10 -c $i -t RF -o SpMM
    done
  done
  pip3 install scikit-learn==1.0.2
fi