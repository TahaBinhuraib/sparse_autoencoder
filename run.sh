#!/bin/bash

latent_dim_values=(16 32 64)
l1_coeff_values=(0.001 0.01 0.0001 0.00001 0.005 0.0005 0.05 0.004 0.006 0.008 0.01)

for latent_dim in "${latent_dim_values[@]}"; do
  for l1_coeff in "${l1_coeff_values[@]}"; do
    python train.py --latent_dim $latent_dim --l1_coeff $l1_coeff
    sleep 3
    echo "---------------------"
  done
done
