from enum import Enum
import json
from pathlib import Path
import torch
import argparse
import random
import string
import numpy as np
from evaluation import MVPEvaluator
from MVP import AggregationStrategy
from beir_eval import run_direct_rerank_eval
from tqdm import tqdm
from scipy.stats import kendalltau, weightedtau, spearmanr

# My idea behind this test:
# Usually, a reranker model looks at the group pointwise, implying it looks at passage A in isolation
# However, MVP does not. It looks at it list wise, so the entire group, and it asks what the average is or ideal topic looks like
# For this specific group, it does so with anchor vectors. Making them, with cross attention between all passages. So it builds them from the passages
# Now i question this by doing the following, I give it a query-passage group. 100 passages and 1 query
# Now normally you would have 100 decently similar/relavant passages to the query, and the anchor vectors would make sense
# But what if, we had 99 random junk passages (noise) and 1 actual relevant query to the passage

# Will it be able to put that relevant query at the top of the list? If not, then it ranks relevant to the group
# Which implies that adding noise to this group makes the MVP model very sensitive to noise and not robust.

# Additionally i check the score, which will prove more than it simply wrongfully ranking this one relevant passage.
# The idea behind the score is that its based on the dot product between a anchor vector and the passage 
# (it takes the mean of all individual dot products of the passage with the anchor vectors (views))
# So essentially, the MPV model encodes the query into these anchor vectors implying that if you are close to the anchors, you are close to the query
# meaning your score is high and you will rank high. However, the hypothesis is now, that if the anchor vectors are based on a bunch of noise
# And 1 relevant passage. Then it will most likely say that the 1 relevant passage is far away from the anchors implying the model has no idea what its actually ranking.
# Which again, implies it is not robust to noise within the ranking group.

# --- Starting Anchor Robustness Test ---

# Query: how long is life cycle of flea
# Target Passage Start:  5. Cancel. A flea can live up to a year, but its ...
# ------------------------------
# Score in Context A (Real Neighbors):   1.2502
# Score in Context B (Junk Neighbors):   1.0490
# ------------------------------
# Stability Delta: 0.2013
# CONCLUSION: FAIL. The model is not robust to noise.

# --- Starting Anchor Robustness Test ---

# Query: are naturalization records public information
# Target Passage Start:  Civil Records Definition. Civil records are a gro...
# ------------------------------
# Score in Context A (Real Neighbors):   -0.8951
# Score in Context B (Junk Neighbors):   -1.4499
# ------------------------------
# Stability Delta: 0.5549
# CONCLUSION: FAIL. The model is not robust to noise.

# Conclusion drawn from my results:
# The results show a significant Score Drift (from 1.25 down to 1.04), which confirms my hypothesis.
# Here is what this implies. Usually, we expect a robust model to see the query and the relevant passage, 
# calculate their similarity, and give us a consistent score (like 1.25), regardless of what else is in the list.

# However, MVP failed to do this. Because it operates listwise and builds the anchor vectors using cross-attention over the entire group,
# it effectively used all of the noise to confuse itself. When we fed it 99 junk passages, the model constructed Anchor Vectors that were heavily influenced by that noise. 
# Essentially, the "ideal topic" (the Anchor) became a "garbage topic."

# Now, looking at the math: the score is just the mean of the dot products between the passage and these anchors. 
# Since our 1 relevant passage is actually good, it is mathematically "far away" from a "garbage anchor." 
# This caused the dot product (the score) to drop by 0.20.

# This proves the point. The MVP model does not know that the passage is relevant. 
# It only knows that the passage fits well relative to the group. 
# When the group becomes noise, the model loses its reference point and penalizes the good passage.

# Final Conclusion. The model is highly sensitive to the setup of the group (context-dependent) and is not robust to noise.

# Do 10, top 10, and 90 from like the top 1000. To be more realistic
# Or compute the anchor vector only on like 75, and then check if it still works good for the other 25 to see if it really encodes the point

# Do 2 tests, one with noise and one without noise. To see if you get a better ranking for the one without the noise
# That way you can prove the noise does affect the noise, so we can see if its sensitive.

# a proposed fix if the hypothesis is true, is to build the anchor vector only with the relevant documents by first doing pointwise


def center_padded_print(
    text: str = "",
    max_width: int = 80,
    pad_token: str = "=",
    add_newline: bool = True,
):
    text_width = len(text)
    if text_width > 0:
        text = f" {text} "
        text_width += 2

    pad_width = max(0, max_width - text_width)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    text = f"{pad_token * pad_left}{text}{pad_token * pad_right}"
    if add_newline:
        text = f"\n{text}"

    print(text)


class NoisePassages(Enum):
    NONE = "none"
    JUNK = "junk"
    RANDOM = "random"
    WORST1000 = "worst1000"


class RobustnessTester(MVPEvaluator):
    def generate_junk_context(self, count=90, length=50):
        junk_list = []
        chars = string.ascii_letters + string.digits
        for _ in range(count):
            noise = ''.join(random.choices(chars, k=length))
            junk_list.append(noise)
        return junk_list
    
    def generate_junk_passage_results(self, count, length=50):
        noise_txts = self.generate_junk_context(count, length)
        junk = []
        base_pid = 1_000_000_000
        for idx, noise_txt in enumerate(noise_txts):
            passage_result = {
                "text": noise_txt,
                "title": "",
                "bm25_score": 0,
                "pid": base_pid + idx
            }
            junk.append(passage_result)
        return junk

    # Steals passages from OTHER queries in the dataset, to get real noise rather than junk
    def get_random_distractors(self, current_idx, count=90):
        distractors = set() # Use a set to prevent duplicate query-passage
        total_instances = len(self.test_file)
        
        # Make sure there can't be an infinite loop
        if total_instances <= 1:
            return self.generate_junk_context(count)

        while len(distractors) < count:
            # Grab random row from the file
            rand_idx = random.randint(0, total_instances - 1)
            if rand_idx == current_idx: 
                continue # Skip the current query-row
            
            other_instance = self.test_file[rand_idx]
            other_results = other_instance[self.args.firststage_result_key]
            
            if not other_results: continue
                
            random_pass = random.choice(other_results)
            text = f"{random_pass[self.args.title_key]} {random_pass[self.args.text_key]}"
            distractors.add(text)
            
        return list(distractors)
    
    def get_random_distractor_passage_results(self, current_idx, count):
        total_instances = len(self.test_file)
        if total_instances <= 1:
            raise NotImplementedError()

        passages_per_instance = self.args.topk

        if (total_instances - 1) * passages_per_instance < count:
            raise NotImplementedError()

        # Create a possible passage locations as tuples (instance index, passage index)
        # while skipping the current instance index
        random_passage_results_loc = [
            (instance_idx, passage_result_idx) 
            for instance_idx in range(total_instances) for passage_result_idx in range(passages_per_instance)
            if instance_idx != current_idx
        ]

        # Select n = `count` unique random tuples 
        selected_passage_results_loc = random.sample(random_passage_results_loc, count)
        # Retrieve the random passages from the tuples
        passage_results = [
            self.test_file[instance_idx][self.args.firststage_result_key][passage_result_idx]
            for instance_idx, passage_result_idx in selected_passage_results_loc
        ]

        return passage_results
    
    def get_worst_from_top1000_passage_results(self, target_query_id, count, top_1000_dir="top1000"):
        assert count <= 1000

        if count <= 0:
            return []

        top100_path = Path(self.args.input_path)
        top100_name = top100_path.name
        top1000_path = top100_path.parent / top_1000_dir / top100_name

        with top1000_path.open(mode="r", encoding="utf-8") as file:
            for line in file:
                query_passage_instance = json.loads(line)
                query_id = query_passage_instance["qid"]
                if target_query_id == query_id:
                    first_stage_results = query_passage_instance["bm25_results"]
                    worst_results = first_stage_results[-count:]
                    return worst_results
        return None

    # Code based upon generate_ranklist() in MVP
    def get_ranking_scores(self, question, candidates):
        full_input_texts = self.make_listwise_text(question, candidates)
        input_tensors = self.make_input_tensors(full_input_texts)
        
        input_ids = input_tensors["input_ids"]
        attention_mask = input_tensors["attention_mask"]

        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.args.max_gen_length,
                return_dict=False,
            )
            if outputs.logits.dim() == 3:
                logits = outputs.logits.squeeze(1).to(input_ids.device)
            else:
                logits = outputs.logits.to(input_ids.device)
            
            topk_logits = logits.topk(logits.size(-1), dim=1)
            topk_indices = topk_logits.indices.tolist()
        
        topk_logits = topk_logits.values.tolist()

        return topk_indices, topk_logits

    def rerank_instance_evaluation(self, instance, ranked_indices, passages: list[dict]):
        reranked_items = []
        for i, passage_idx in enumerate(ranked_indices):
            passage_idx = int(passage_idx)
            template  = passages[passage_idx]
            template['orig_'+self.args.score_key] = template[self.args.score_key]
            template[self.args.score_key] = 100000 - i
            reranked_items.append(template)

        instance[self.args.firststage_result_key] = reranked_items
        data = [instance]
        ndcg_k, scores = run_direct_rerank_eval(data, k=self.args.retrieve_k, combined=True)
        return ndcg_k, scores

    def rerank_dataset_evaluation(
        self,
        dataset_ranked_indices: list[list],
        dataset_passages: list[list[dict]]
    ):
        reranked_instances = []
        for instance, ranked_indices, passages in zip(
            self.test_file, dataset_ranked_indices, dataset_passages
        ):
            reranked_items = []
            for i, passage_idx in enumerate(ranked_indices):
                passage_idx = int(passage_idx)
                template  = passages[passage_idx]
                template['orig_'+self.args.score_key] = template[self.args.score_key]
                template[self.args.score_key] = 100000 - i                
                reranked_items.append(template)
            
            instance[self.args.firststage_result_key] = reranked_items
            reranked_instances.append(instance)

        ndcg_k, scores = run_direct_rerank_eval(reranked_instances, k=self.args.retrieve_k, combined=True)
        return ndcg_k, scores

    def run_anchor_noise_test(self, instance_idx: int):
        center_padded_print("Starting Anchor Robustness Test", pad_token="-")
        print(f"Instance: {instance_idx}")

        noise_type = NoisePassages(self.args.noise)
        match noise_type:
            case NoisePassages.NONE:
                noise_mode = f"No noise. Rerank retrieved {self.args.retrieve_k}"
            case NoisePassages.JUNK:
                noise_mode = "Random Junk Noise"
            case NoisePassages.RANDOM:
                noise_mode = "Real Distractor Noise"
            case NoisePassages.WORST1000:
                noise_mode = "The worst passage from the top1000 first stage ranking"
            case _:
                raise ValueError("Unknown noise type")
        print(f"Mode: {noise_mode} ({noise_type})")

        # We test on the first query (Index 0) of our test file
        TARGET_IDX = instance_idx
        instance = self.test_file[TARGET_IDX]
        qid = instance["qid"]
        question = instance[self.args.question_text_key]
        bm25_results = instance[self.args.firststage_result_key]
        qrels = instance["qrels"]
        
        # Format ALL original candidates
        original_txt = [f"{x[self.args.title_key]} {x[self.args.text_key]}".strip() for x in bm25_results]
        original_pid = [x["pid"] for x in bm25_results]
        
        # First we get the ranking of the top 100 bm25, as if we run normal inference
        print(f"\nPhase 1: Running Reference Inference on all {len(original_txt)} candidates...")
        ranked_indices, ranked_scores = self.get_ranking_scores(question, original_txt)
        ranked_indices = ranked_indices[0]
        ranked_scores = ranked_scores[0]
        ranked_pid = np.array(original_pid)[ranked_indices]
        ranked_qrels = [qrels.get(pid, 0) for pid in ranked_pid]
        
        # Now we get the top K favorites of this original inference, for example the top 10 ranked documents
        k = self.args.retrieve_k
        top_k_indices = ranked_indices[:k]
        
        print(f"Model's Top {k} Favorites (Indices): {top_k_indices}")
        
        # Get the text of the top k
        # top_passages = [original_txt[i] for i in top_k_indices]
        top_passage_results = [bm25_results[i] for i in top_k_indices]

        top_scores_ref = ranked_scores[:k]

        # TEST (THE ACTUAL ROBUSTNESS TEST!)
        # We always rank a total of 100 docs, so we get the noise of what we still need
        noise_count = 100 - k
        if noise_count < 0: noise_count = 0

        match noise_type:
            case NoisePassages.NONE:
                noise_passage_results = []
            case NoisePassages.JUNK:
                print(f"Generating {noise_count} random junk strings...")
                noise_passage_results = self.generate_junk_passage_results(count=noise_count)
            case NoisePassages.RANDOM:
                print(f"Gathering {noise_count} irrelevant passages from other queries...")
                noise_passage_results = self.get_random_distractor_passage_results(TARGET_IDX, count=noise_count)
            case NoisePassages.WORST1000:
                print(f"Gathering {noise_count} worst passages from top1000...")
                noise_passage_results = self.get_worst_from_top1000_passage_results(qid, count=noise_count)
            case _:
                raise ValueError("Unknown noise type")
            
        # Create the list of query-passages to do inference on, which has the top k be the top-k we found and the rest noise
        # Placing them at the top should be no issue, since the paper claims non-positional bias in the model
        noisy_bm25_results = top_passage_results + noise_passage_results

        noisy_txt = [f"{x[self.args.title_key]} {x[self.args.text_key]}".strip() for x in noisy_bm25_results]
        noisy_pid = [x["pid"] for x in noisy_bm25_results]

        print(f"Running Stress Inference on [{k} Top passagas + {noise_count} Noise]...")
        ranked_stress_indices, ranked_stress_scores = self.get_ranking_scores(question, noisy_txt)
        ranked_stress_indices = ranked_stress_indices[0]
        ranked_stress_scores = ranked_stress_scores[0]
        ranked_stress_pid = np.array(noisy_pid)[ranked_stress_indices]
        ranked_stress_qrels = [qrels.get(pid, 0) for pid in ranked_stress_pid]

        # We get the top 10, from this noisey test
        top_scores_stress_test = ranked_stress_scores[:k]
        
        # The idea here is, that we put our top k at the top of the inference list, implying that it still should be there
        # After the ranking
        top_k_stress_test_order = ranked_stress_indices[:k]
        
        center_padded_print("RESULTS", pad_token="-")
        print(f"Query: {question}")
        print(f"Reference Order: {list(range(k))}")
        print(f"Noisey Test Order:    {top_k_stress_test_order}")
        
        if np.array_equal(top_k_stress_test_order, np.arange(k)):
            print("\nCONCLUSION: PASS. Perfect Consistency.")
            print("The model kept its Top K in the exact same order despite the noise.")
        else:
            print("\nCONCLUSION: FAIL. Ranking Issues Detected.")
            print("The relative order of the Top K changed when the background noise changed.")
            
        for position,(normal_pid, normal_qrel, stress_pid, stress_qrel) in enumerate(zip(ranked_pid[:k], ranked_qrels[:k], ranked_stress_pid[:k], ranked_stress_qrels[:k])):
            print(f"  -> Position #{position+1}: Document {normal_pid} (qrel {normal_qrel}), but is now Document {stress_pid} (qrel {stress_qrel})")
        

        survivor_count = sum(1 for idx in top_k_stress_test_order if idx < k)
        center_padded_print("SURVIVAL RATE", pad_token="-")

        print(f"Survivors in Top {k}: {survivor_count}/{k}")
        if survivor_count < k:
            print(f"CRITICAL FAIL: {k - survivor_count} relevant documents were pushed out of the Top {k} by the noise!")
        else:
            print(f"PASS: All original VIPs remained in the Top {k} (even if shuffled).")

        center_padded_print("SCORE DRIFT (Candidate #1)", pad_token="-")
        print(f"Original Score: {top_scores_ref[0]:.4f}")
        print(f"Noisey Score: {top_scores_stress_test[0]:.4f}")
        print(f"Delta: {abs(top_scores_ref[0] - top_scores_stress_test[0]):.4f}")

        center_padded_print("PERFORMANCE", pad_token="-")
        print("Original ranking")
        ndcg_k_, scores_ = self.rerank_instance_evaluation(instance, ranked_indices, bm25_results)
        print("Ranking with added noise")
        ndcg_k_, scores_ = self.rerank_instance_evaluation(instance, ranked_stress_indices, noisy_bm25_results)
    
    def run_anchor_noise_test_on_dataset(self):
        center_padded_print("Starting Anchor Robustness Test on full dataset", pad_token="-")

        noise_type = NoisePassages(self.args.noise)
        match noise_type:
            case NoisePassages.NONE:
                noise_mode = f"No noise. Rerank retrieved {self.args.retrieve_k}"
            case NoisePassages.JUNK:
                noise_mode = "Random Junk Noise"
            case NoisePassages.RANDOM:
                noise_mode = "Real Distractor Noise"
            case NoisePassages.WORST1000:
                noise_mode = "The worst passage from the top1000 first stage ranking"
            case _:
                raise ValueError("Unknown noise type")
        print(f"Mode: {noise_mode} ({noise_type})")

        ranked_indices_list = []
        bm25_results_list = []
        ranked_stress_indices_list = []
        noisy_bm25_results_list = []
        
        for instance_idx, instance in enumerate(tqdm(self.test_file)):
            question = instance[self.args.question_text_key]
            bm25_results = instance[self.args.firststage_result_key]
            bm25_results_list.append(bm25_results)

            qrels = instance["qrels"]
            qid = instance["qid"]
            
            # Format ALL original candidates
            original_txt = [f"{x[self.args.title_key]} {x[self.args.text_key]}".strip() for x in bm25_results]
            original_pid = [x["pid"] for x in bm25_results]
        
            # First we get the ranking of the top 100 bm25, as if we run normal inference
            ranked_indices, ranked_scores = self.get_ranking_scores(question, original_txt)
            ranked_indices = ranked_indices[0]
            ranked_indices_list.append(ranked_indices)

            ranked_scores = ranked_scores[0]
            ranked_pid = np.array(original_pid)[ranked_indices]
            ranked_qrels = [qrels.get(pid, 0) for pid in ranked_pid]
            
            # Now we get the top K favorites of this original inference, for example the top 10 ranked documents
            k = self.args.retrieve_k
            top_k_indices = ranked_indices[:k]
        
            # Get the text of the top k
            # top_passages = [original_txt[i] for i in top_k_indices]
            top_passage_results = [bm25_results[i] for i in top_k_indices]

            top_scores_ref = ranked_scores[:k]

            # TEST (THE ACTUAL ROBUSTNESS TEST!)
            # We always rank a total of 100 docs, so we get the noise of what we still need
            noise_count = 100 - k
            if noise_count < 0: noise_count = 0

            match noise_type:
                case NoisePassages.NONE:
                    noise_passage_results = []
                case NoisePassages.JUNK:
                    noise_passage_results = self.generate_junk_passage_results(count=noise_count)
                case NoisePassages.RANDOM:
                    noise_passage_results = self.get_random_distractor_passage_results(instance_idx, count=noise_count)
                case NoisePassages.WORST1000:
                    noise_passage_results = self.get_worst_from_top1000_passage_results(qid, count=noise_count)
                case _:
                    raise ValueError("Unknown noise type")
            
            # Create the list of query-passages to do inference on, which has the top k be the top-k we found and the rest noise
            # Placing them at the top should be no issue, since the paper claims non-positional bias in the model
            noisy_bm25_results = top_passage_results + noise_passage_results
            noisy_bm25_results_list.append(noisy_bm25_results)

            noisy_txt = [f"{x[self.args.title_key]} {x[self.args.text_key]}".strip() for x in noisy_bm25_results]
            noisy_pid = [x["pid"] for x in noisy_bm25_results]

            assert len(noisy_txt) in [k, 100]

            ranked_stress_indices, ranked_stress_scores = self.get_ranking_scores(question, noisy_txt)
            ranked_stress_indices = ranked_stress_indices[0]


            ranked_stress_indices_list.append(ranked_stress_indices)

            ranked_stress_scores = ranked_stress_scores[0]
            ranked_stress_pid = np.array(noisy_pid)[ranked_stress_indices]
            ranked_stress_qrels = [qrels.get(pid, 0) for pid in ranked_stress_pid]

        center_padded_print("PERFORMANCE", pad_token="-")
        print("Original ranking")
        # ndcg_k_, scores_ = self.rerank_instance_evaluation(instance, ranked_indices, bm25_results)
        ndcg_k, scores = self.rerank_dataset_evaluation(ranked_indices_list, bm25_results_list)
        print("Ranking with added noise")
        # ndcg_k_, scores_ = self.rerank_instance_evaluation(instance, ranked_stress_indices, noisy_bm25_results)
        ndcg_k, scores = self.rerank_dataset_evaluation(ranked_stress_indices_list, noisy_bm25_results_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='bulbna/MVP-base', type=str)
    parser.add_argument('--input_path', type=str, default='./eval_data/dl19.jsonl')
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--n_special_tokens', default=1, type=int)
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--padding', default='max_length', type=str)
    parser.add_argument('--max_gen_length', default=10, type=int)
    parser.add_argument('--measure_flops', action='store_true')

    parser.add_argument('--instance_idx', type=int, default=None)
    parser.add_argument(
        '--noise',
        type=str,
        choices=[noise.value for noise in NoisePassages],
        default=NoisePassages.NONE,
        help="The type of noisy passages that is added to the top k ranking of the first reranking"
    )
    parser.add_argument('--retrieve_k', default=10, type=int, help="Number of Top Candidates to evaluate this robustness test with")
    parser.add_argument("--aggregation_strategy", default=AggregationStrategy.MEAN, type=AggregationStrategy, choices=list(AggregationStrategy))
    parser.add_argument("--single_view_index", type=int)
    
    parser.add_argument('--question_text_key', default='q_text')
    parser.add_argument('--firststage_result_key', default='bm25_results')
    parser.add_argument('--score_key', default='bm25_score', type=str)
    parser.add_argument('--title_key', default='title')
    parser.add_argument('--text_key', default='text')

    args = parser.parse_args()
    
    tester = RobustnessTester(args)
    
    if args.instance_idx is not None:
        tester.run_anchor_noise_test(args.instance_idx)
    else:
        tester.run_anchor_noise_test_on_dataset()


# --- Starting Anchor Robustness Test ---
# Mode: Real Distractor Noise

# Phase 1: Running Reference Inference on all 100 candidates...
# Model's Top 10 Favorites (Indices): [ 1  2 44 31 28 52 21 38 24 55]
# Gathering 90 irrelevant passages from other queries...
# Running Stress Inference on [10 Top passagas + 90 Noise]...

# --- RESULTS ---
# Query: how long is life cycle of flea
# Reference Order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Noisey Test Order:    [8 7 5 2 3 9 4 6 0 1]

# CONCLUSION: FAIL. Ranking Issues Detected.
# The relative order of the Top K changed when the background noise changed.
#   -> Rank #1 was TOP #0, but is now TOP #8
#   -> Rank #2 was TOP #1, but is now TOP #7
#   -> Rank #3 was TOP #2, but is now TOP #5
#   -> Rank #4 was TOP #3, but is now TOP #2
#   -> Rank #5 was TOP #4, but is now TOP #3
#   -> Rank #6 was TOP #5, but is now TOP #9
#   -> Rank #7 was TOP #6, but is now TOP #4
#   -> Rank #8 was TOP #7, but is now TOP #6
#   -> Rank #9 was TOP #8, but is now TOP #0
#   -> Rank #10 was TOP #9, but is now TOP #1

# --- SURVIVAL RATE ---
# Survivors in Top 10: 10/10
# PASS: All original VIPs remained in the Top 10 (even if shuffled).

# --- SCORE DRIFT (Candidate #1) ---
# Original Score: 2.8773
# Noisey Score: 0.4167
# Delta: 2.4606

# --- Starting Anchor Robustness Test ---
# Mode: Real Distractor Noise

# Phase 1: Running Reference Inference on all 100 candidates...
# Model's Top 10 Favorites (Indices): [44 13  6 12 65 90 76  2 10 43]
# Gathering 90 irrelevant passages from other queries...
# Running Stress Inference on [10 Top passagas + 90 Noise]...

# --- RESULTS ---
# Query: are naturalization records public information
# Reference Order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Noisey Test Order:    [1 2 6 4 3 0 7 5 9 8]

# CONCLUSION: FAIL. Ranking Issues Detected.
# The relative order of the Top K changed when the background noise changed.
#   -> Rank #1 was TOP #0, but is now TOP #1
#   -> Rank #2 was TOP #1, but is now TOP #2
#   -> Rank #3 was TOP #2, but is now TOP #6
#   -> Rank #4 was TOP #3, but is now TOP #4
#   -> Rank #5 was TOP #4, but is now TOP #3
#   -> Rank #6 was TOP #5, but is now TOP #0
#   -> Rank #7 was TOP #6, but is now TOP #7
#   -> Rank #8 was TOP #7, but is now TOP #5
#   -> Rank #9 was TOP #8, but is now TOP #9
#   -> Rank #10 was TOP #9, but is now TOP #8

# --- SURVIVAL RATE ---
# Survivors in Top 10: 10/10
# PASS: All original VIPs remained in the Top 10 (even if shuffled).

# --- SCORE DRIFT (Candidate #1) ---
# Original Score: 0.9296
# Noisey Score: 0.5092
# Delta: 0.4205


# Notes to these findings
# The fact that 10/10 survived proves that the MVP model works.
# It successfully encoded the "meaning" of the relevant passages.
# It successfully encoded the "meaninglessness" of the noise.
# Even with a corrupted Anchor Vector, the mathematical similarity of a "Real Passage" was still higher than "Random Junk."
# Conclusion: The model is safe against Spam Injection (junk won't suddenly rank #1).
# The Bad News: The Decoder is Confused
# While the set of Top 10 remained the same, the order of the top 10 was altered.
# Rank 1 (The Best Answer) dropped to Rank 9.
# Rank 10 (The Worst Answer) rose to Rank 1.

# Why is this an issue?
# The entire purpose of a Reranker is to fix the ordering.
# A cheap retriever (BM25) can already find the Top 10ish documents.
# We pay the expensive computational cost of MVP specifically to know which one is #1.
# If MVP shuffles the Top 10 randomly based on background noise, it is failing its primary job.


# An additional test that comes form this would be to do the following
# Rank the top 100 bm25 scores to get a top 10
# Now rank this top 10 on its own (so the anchor vectors will be based on solely this top 10)
# if the order of the top 10 changes, then there clearly is issues with the model implying the anchor vector becomes "diluted"
# The more passages you add, the more diluted the anchor vector becomes, messing with the top 10 score.
# This was done in anchor_dilution_test.py, results are there