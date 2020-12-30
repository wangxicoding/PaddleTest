#!/bin/bash

optimizer=(sgd adam adagrad)

for opt in ${optimizer[@]}; do
  echo "---- opt=${opt} ----"
  fleetrun --gpus 0,1 train_dyn_rnn.py \
      --num_epochs=1 \
      --enable_ce \
      --use_gradient_merge=True \
      --batch_size=16 \
      --max_step=40 \
      --base_optimizer=${opt} \
      --embedding_type="dense"
  
  fleetrun --gpus 0,1 train_dyn_rnn.py \
      --num_epochs=1 \
      --enable_ce \
      --batch_size=64 \
      --max_step=10 \
      --base_optimizer=${opt} \
      --embedding_type="dense"
done
