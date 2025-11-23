TEST_DATA=(dl19 dl20 trec-covid nfcorpus signal news robust04 scifact touche dbpedia-entity)
# TEST_DATA=(dl19 dl19_rev dl19_shuffle dl20 dl20_rev dl20_shuffle news news_rev news_shuffle)
# TEST_DATA=(dl19 dl20)
# deze krijgen een error op snellius
# TEST_DATA=(news)
# TEST_DATA=(robust04)
# TEST_DATA=(touche)
# additional flags
# --model_path Jun421/MVP-3b
# --model_path Jun421/MVP-base
# --measure_flops 
# --model_path ../checkpoints/MVP/tfmr_0_step25000 (4 views)
# --model_path ../checkpoints_1views/MVP/tfmr_0_step16000
# --model_path ../checkpoints_2views/MVP/tfmr_0_step23000
# --model_path ../checkpoints_3views/MVP/tfmr_0_step24000
# --model_path ../checkpoints_5views/MVP/tfmr_0_step17000
# --model_path ../checkpoints_6views/MVP/tfmr_0_step17000
for data in ${TEST_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python3 ./evaluation.py --input_path ./eval_data/${data}.jsonl \
        --output_path ./outputs/mvp-${data}.jsonl --topk 100 \
        --n_special_tokens 4 \
        --model_path ../checkpoints_2views/MVP/tfmr_0_step23000
done
