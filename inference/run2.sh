# BEIR_DATA=(nfcorpus news scifact touche )
BEIR_DATA=(dbpedia-entity)
for data in ${BEIR_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=1 python3 evaluation_rank_origin.py --input_path eval_data/baseline/${data}.jsonl \
        --output_path ./outputs/listt5-${data}.jsonl --topk 100 --pooling_type text\
        --n_special_tokens 4\
        --model_path /data/kjun/checkpoints/MVT5_RDLLM/MVT5_s_11_extra_4_seed_0_RDLLM_col_100_5_ortho_1.0_warmup_5_temp_0.8_FIDLIGHT/tfmr_0_step15000 \
        --store_result true
done

# ortho=(0.7)
# wamup=(5 10)
# temp=(0.5 0.6 0.7 0.8 0.9 1.0)

# for data in ${BEIR_DATA[@]}; do
#     for ortho_value in ${ortho[@]}; do
#         for warmup in ${wamup[@]}; do
#             for t in ${temp[@]}; do
#                 CUDA_VISIBLE_DEVICES=0 python3 evaluation_rank_origin.py --input_path eval_data/baseline/${data}.jsonl \
#                     --output_path ./outputs/listt5-${data}.jsonl --topk 100 --pooling_type extra\
#                     --n_special_tokens 4\
#                     --model_path /data/kjun/checkpoints/TUNING/ortho_${ortho_value}/warmup_${warmup}/temp_${t} \
#                     --store_result true
#             done
#         done
#     done
# done