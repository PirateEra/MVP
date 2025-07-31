TEST_DATA=(dl19 dl20)
for data in ${TEST_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python3 evaluation.py --input_path eval_data/${data}.jsonl \
        --output_path ./outputs/mvp-${data}.jsonl --topk 100 \
        --n_special_tokens 4 \
        --model_path /data/kjun/checkpoints/MVT5_RDLLM/MVP_REPRODUCE_CONDA_CHECK/tfmr_0_step13000 \
        --measure_flops
done
