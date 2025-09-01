# DEFAULT HYPER PARAMS
# LEARNING_RATE = 1e-4
# # OF EXTRA : 4
# SOFTMAX TEMPERATURE : 0.8
# WARMUP : 5 (5%)
# ORTHOGONAL WEIGHT : 1.0

seed=(0)
learning_rates=(1e-04)
n_passages=(5)
softmax_temps=(0.8)
n_special_tokens=(4)
local_weights=(1.0)
warmup_ratio=(5)


# Loop through each combination of softmax_temp and learning_rate
for s in "${seed[@]}"; do
  for np in "${n_passages[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for temp in "${softmax_temps[@]}"; do
        for ns in "${n_special_tokens[@]}"; do
          for lw in "${local_weights[@]}"; do
            for warmup in "${warmup_ratio[@]}"; do
              # model_name="MVP_extra_${ns}_seed_${s}_RDLLM_ortho_${lw}_warmup_${warmup}_temp_${temp}"
              model_name="MVP"
                # Run the training script with the current combination
                CUDA_VISIBLE_DEVICES=0,1 python3 train.py --name $model_name \
                --seed ${s}\
                --base_model t5-base \
                --train-files /PATH/TO/TRAIN/FILE/train_col100_sampled_100_5.jsonl \
                --eval-files /PATH/TO/TRAIN/FILE/dl21.jsonl \
                --do_train --learning_rate $lr --gradient_accumulation_steps 2\
                --train_batch_size 16 --eval_batch_size 1 --num_workers 0 \
                --max_input_length 256 --max_output_length 5 --n_passages ${np}\
                --sub-mode normalize_positive_shuffle_pos_sort_posfirst \
                --lr_scheduler linear --fid \
                --pooling_type rv --n_special_tokens $ns --special_pooling max \
                --local_weight $lw \
                --decoding_strategy single \
                --target_seq token \
                --num_train_epochs 1 \
                --dist_option rank_inverse --softmax_temp $temp \
                --warmup_steps $warmup \
                --eval_steps 2000 \
                --wandb
              sleep 5
            done
          done
        done
      done
    done
  done
done