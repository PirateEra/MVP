# 체크포인트를 Disk에 저장하려면
# --save_in_disk 사용. 
# DEFAULT HYPER PARAMS
# LEARNING_RATE = 1e-4
# # OF EXTRA : 4
# SOFTMAX TEMPERATURE : 0.8
# WARMUP : 5 (5%)
# ORTHOGONAL WEIGHT : 1.0


softmax_temps=(0.8)
learning_rates=(1e-04)
seed=(0)
len_data=(5)
n_special_tokens=(4)
loss_types=(listnet)
score_types=(dot)
local_weights=(1.0)
warmup_ratio=(5)

# Loop through each combination of softmax_temp and learning_rate
for temp in "${softmax_temps[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for data in "${len_data[@]}"; do
      for loss_type in "${loss_types[@]}"; do
        for score_type in "${score_types[@]}"; do
          for s in "${seed[@]}"; do
            for lw in "${local_weights[@]}"; do
              for ns in "${n_special_tokens[@]}"; do
                for warmup in "${warmup_ratio[@]}"; do

                # Construct the unique model name by appending temp and lr to the base name
                model_name="MVT5_s_9_extra_${ns}_seed_${s}_RDLLM_col_100_${data}_ortho_${lw}_warmup_${warmup}_temp_${temp}_REPRODUCE"

                  # Run the training script with the current combination
                  CUDA_VISIBLE_DEVICES=0,1 python3 train.py --name $model_name \
                  --seed ${s}\
                  --base_model t5-base \
                  --train-files /home/tako/kjun/data/train/train_col100_sampled_100_5.jsonl \
                  --eval-files /home/tako/kjun/data/validation/dl21.jsonl \
                  --do_train --learning_rate $lr --train_batch_size 16 --eval_batch_size 1 --num_workers 0 --max_input_length 256 --prompt_type orig \
                  --sub-mode listwise_onlypos_normalize_positive_shuffle_pos_sort_posfirst \
                  --lr_scheduler linear --fid --gradient_accumulation_steps 2 \
                  --max_output_length 5 --listwise_k ${data} --n_passages ${data} \
                  --pooling_type rv --n_special_tokens $ns --special_pooling max \
                  --local_weight $lw \
                  --decoding_strategy single \
                  --target_seq token \
                  --num_train_epochs 1 \
                  --dist_option rank_inverse --softmax_temp $temp \
                  --loss_type $loss_type \
                  --score_type $score_type \
                  --warmup_steps $warmup \
                  --eval_steps 2000\
                  --wandb

                sleep 5
                done
              done
            done
          done
        done
      done
    done
  done
done




