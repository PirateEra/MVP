# TEST_DATA=(dl19 dl20 trec-covid nfcorpus signal news robust04 scifact touche dbpedia-entity)
TEST_DATA=(dl19 dl19_rev dl19_shuffle dl20 dl20_rev dl20_shuffle news news_rev news_shuffle)
# TEST_DATA=(dl19 dl20)
# deze krijgen een error op snellius
# TEST_DATA=(news)
# TEST_DATA=(robust04)
# TEST_DATA=(touche)
# additional flags
# --model_path Jun421/MVP-3b
# --model_path Jun421/MVP-base
# --measure_flops 
# --model_path ../checkpoints/MVP/best-epoch=00-global_step=24999.0.ckpt
for data in ${TEST_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python3 ./evaluation.py --input_path ./eval_data/${data}.jsonl \
        --output_path ./outputs/mvp-${data}.jsonl --topk 100 \
        --n_special_tokens 4 \
        --model_path ../checkpoints/MVP/best-epoch=00-global_step=24999.0.ckpt
done
