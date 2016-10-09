#!/bin/bash
results_table="armsprop"
results_db="amosca02"
results_host="gpuvm1"

if [ "$2" != "" ]; then
  random_seeds=`tail -n +$2 random_seeds`
else
  random_seeds=`cat random_seeds`
fi
  echo $random_seed
for exp in *.yaml; do
  for random_seed in $random_seeds; do
    ~/git/toupee-keras/examples/mlp.py $exp --seed=$random_seed --results-table=$results_table --results-db=$results_db --results-host=$results_host
  done
done
