import torch
import argparse
import random
import string
import numpy as np
from evaluation import MVPEvaluator

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
# 

class RobustnessTester(MVPEvaluator):
    def generate_junk_context(self, count=90, length=50):
        junk_list = []
        chars = string.ascii_letters + string.digits
        for _ in range(count):
            noise = ''.join(random.choices(chars, k=length))
            junk_list.append(noise)
        return junk_list

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

    # Code based upon generate_ranklist() in MVP
    def get_ranking_scores(self, question, candidates):
        full_input_texts = self.make_listwise_text(question, candidates)
        input_tensors = self.make_input_tensors(full_input_texts)
        
        with torch.no_grad():
            outputs = self.model(**input_tensors)

        # Get the raw scores
        all_scores = outputs.logits.flatten().cpu().numpy()

        # get the ranked raw scores
        ranked_indices = np.argsort(-all_scores)
        
        return ranked_indices, all_scores

    def run_anchor_noise_test(self):
        print(f"--- Starting Anchor Robustness Test ---")
        print(f"Mode: {'Random Junk Noise' if self.args.use_junk else 'Real Distractor Noise'}")
        
        # We test on the first query (Index 0) of our test file
        TARGET_IDX = 0
        instance = self.test_file[TARGET_IDX]
        question = instance[self.args.question_text_key]
        bm25_results = instance[self.args.firststage_result_key]
        
        # Format ALL original candidates
        original_txt = [f"{x[self.args.title_key]} {x[self.args.text_key]}" for x in bm25_results]
        
        # First we get the ranking of the top 100 bm25, as if we run normal inference
        print(f"\nPhase 1: Running Reference Inference on all {len(original_txt)} candidates...")
        ranked_indices, original_scores = self.get_ranking_scores(question, original_txt)
        
        # Now we get the top K favorites of this original inference, for example the top 10 ranked documents
        k = self.args.retrieve_k
        top_k_indices = ranked_indices[:k]
        
        print(f"Model's Top {k} Favorites (Indices): {top_k_indices}")
        
        # Get the text of the top k
        top_passages = [original_txt[i] for i in top_k_indices]
        top_scores_ref = [original_scores[i] for i in top_k_indices]

        # TEST (THE ACTUAL ROBUSTNESS TEST!)
        # We always rank a total of 100 docs, so we get the noise of what we still need
        noise_count = 100 - k
        if noise_count < 0: noise_count = 0
        
        if self.args.use_junk:
            print(f"Generating {noise_count} random junk strings...")
            noise_txt = self.generate_junk_context(count=noise_count)
        else:
            print(f"Gathering {noise_count} irrelevant passages from other queries...")
            noise_txt = self.get_random_distractors(TARGET_IDX, count=noise_count)
            
        # Create the list of query-passages to do inference on, which has the top k be the top-k we found and the rest noise
        # Placing them at the top should be no issue, since the paper claims non-positional bias in the model
        test_list = top_passages + noise_txt
        
        print(f"Running Stress Inference on [{k} Top passagas + {noise_count} Noise]...")
        stress_indices, stress_scores = self.get_ranking_scores(question, test_list)

        # We get the top 10, from this noisey test
        top_scores_stress_test = stress_scores[:k]
        
        # The idea here is, that we put our top k at the top of the inference list, implying that it still should be there
        # After the ranking
        top_k_stress_test_order = stress_indices[:k]
        
        print(f"\n--- RESULTS ---")
        print(f"Query: {question}")
        print(f"Reference Order: {list(range(k))}")
        print(f"Noisey Test Order:    {top_k_stress_test_order}")
        
        if np.array_equal(top_k_stress_test_order, np.arange(k)):
            print("\nCONCLUSION: PASS. Perfect Consistency.")
            print("The model kept its Top K in the exact same order despite the noise.")
        else:
            print("\nCONCLUSION: FAIL. Ranking Issues Detected.")
            print("The relative order of the Top K changed when the background noise changed.")
            
            for rank, idx in enumerate(top_k_stress_test_order):
                if rank != idx:
                    print(f"  -> Rank #{rank+1} was TOP #{rank}, but is now TOP #{idx}")
        

        survivor_count = sum(1 for idx in top_k_stress_test_order if idx < k)
        print(f"\n--- SURVIVAL RATE ---")
        print(f"Survivors in Top {k}: {survivor_count}/{k}")
        if survivor_count < k:
            print(f"CRITICAL FAIL: {k - survivor_count} relevant documents were pushed out of the Top {k} by the noise!")
        else:
            print(f"PASS: All original VIPs remained in the Top {k} (even if shuffled).")

        print(f"\n--- SCORE DRIFT (Candidate #1) ---")
        print(f"Original Score: {top_scores_ref[0]:.4f}")
        print(f"Noisey Score: {top_scores_stress_test[0]:.4f}")
        print(f"Delta: {abs(top_scores_ref[0] - top_scores_stress_test[0]):.4f}")

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

    parser.add_argument('--use_junk', action='store_true')
    parser.add_argument('--retrieve_k', default=10, type=int, help="Number of Top Candidates to evaluate this robustness test with")
    
    parser.add_argument('--question_text_key', default='q_text')
    parser.add_argument('--firststage_result_key', default='bm25_results')
    parser.add_argument('--title_key', default='title')
    parser.add_argument('--text_key', default='text')

    args = parser.parse_args()
    
    tester = RobustnessTester(args)
    tester.run_anchor_noise_test()


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