set -e
set -u

TEST_DATA=(dl19 dl20 news signal scifact)
# TEST_DATA=(dl19)
# TEST_DATA=(dl20)
# TEST_DATA=(news)
# TEST_DATA=(signal)
# TEST_DATA=(scifact)

NOISE_MODE=(none junk random)
RETRIEVE_Ks=(1 10 20 30 40 50 60 70 80 90 100)

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "Starting experiments at: $(date)"
echo "⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶"

for data in ${TEST_DATA[@]}; do
    for noise in ${NOISE_MODE[@]}; do
        for retrieve_k in ${RETRIEVE_Ks[@]}; do
            echo "Running: data=${data}, noise=${noise}, retrieve_k=${retrieve_k}"
            
            # Create nested subdirectories: logs/data/noise/
            DATA_LOG_DIR="${LOG_DIR}/${data}/${noise}"
            mkdir -p "$DATA_LOG_DIR"
            
            RUN_ID="k${retrieve_k}"
            LOG_FILE="${DATA_LOG_DIR}/${RUN_ID}.log"

            {
                echo "⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶"
                echo "Run ID: ${data}_${noise}_${RUN_ID}"
                echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "Parameters:"
                echo "  data: $data"
                echo "  noise: $noise"
                echo "  retrieve_k: $retrieve_k"
                echo "⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶"
                echo
            } 2>&1 | tee "$LOG_FILE"
                
            python ./anchor_robust_test.py \
                --input_path ./eval_data/${data}.jsonl \
                --topk 100 \
                --n_special_tokens 4 \
                --model_path Jun421/MVP-base \
                --noise "${noise}" \
                --retrieve_k "${retrieve_k}" 2>&1 | tee -a "$LOG_FILE"
            
            {
                echo
                echo "⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶"
                echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "Exit code: $?"
                echo "⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶"
            } 2>&1 | tee -a "$LOG_FILE"
            
            echo "---"
        done
    done
done

echo "⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶⠶"
echo "All experiments completed at: $(date)"