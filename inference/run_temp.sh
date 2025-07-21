# BEIR_DATA=(bioasq fever scidocs fiqa)
step=(15000)
BEIR_DATA=(dl19 dl20)

for step in "${step[@]}"; do
    for data in "${BEIR_DATA[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python3 evaluation_rank_origin.py --input_path ./eval_data/baseline/${data}.jsonl \
            --output_path ./outputs/listt5-${data}.jsonl --topk 100 --pooling_type extra\
            --n_special_tokens 4\
            --model_path /data/kjun/checkpoints/MVT5_RDLLM/MVT5_s_11_extra_4_seed_0_RDLLM_col_200_5_ortho_0.5_warmup_3000/tfmr_0_step21000
    done
done



# CUDA_VISIBLE_DEVICES=0 python3 evaluation_rank_origin.py --input_path /home/tako/kjun/MVT5_inference/eval_data/else/bioasq.jsonl \
#     --output_path ./outputs/listt5-bioasq.jsonl --topk 100 --pooling_type rv\
#     --n_special_tokens 4\
#     --special_loc 4\
#     --model_path /data/kjun/checkpoints/MVT5_v2/test/tfmr_0_step23000