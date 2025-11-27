import torch
import random
import string
import argparse
from evaluation import MVPEvaluator

# My idea behind this test:
# Usually, a reranker model looks at the group pointwise, implying it looks at passage A in isolation
# However, MPV does not. It looks at it list wise, so the entire group, and it asks what the average is or ideal topic looks like
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

class RobustnessTester(MVPEvaluator):
    def generate_junk_context(self, count=99, length=50):
        junk_list = []
        chars = string.ascii_letters + string.digits
        for _ in range(count):
            noise = ''.join(random.choices(chars, k=length))
            junk_list.append(noise)
        return junk_list
    
    # Code based upon generate_ranklist() in MVP
    def get_target_score(self, question, target_passage, noise_passages):
        ctxs = [target_passage] + noise_passages
        full_input_texts = self.make_listwise_text(question, ctxs)
        input_tensors = self.make_input_tensors(full_input_texts)
        # access the internal model output to get raw scores, not just ranks
        with torch.no_grad():
            outputs = self.model(**input_tensors)

        # We grab the score assigned to the first passage in the list (target/index 0)
        raw_score = outputs.logits.flatten()[0].item()

        return raw_score

    def run_anchor_noise_test(self):
        print(f"--- Starting Anchor Robustness Test ---")
        
        # Load the instance from the dataset
        instance = self.test_file[0]
        # This is the question aka the query, based on the input path
        question = instance[self.args.question_text_key]
        
        # Here I assume the first result in the dataset is the Target (Relevant), but this should be double checked
        # So i assume based on the question, we retrieved bm25 results, where the top 1 is #1 for bm25
        bm25_results = instance[self.args.firststage_result_key]
        target_passage = f"{bm25_results[0][self.args.title_key]} {bm25_results[0][self.args.text_key]}"
        
        # CLEAN INFERENCE
        original_neighbors = [
            f"{x[self.args.title_key]} {x[self.args.text_key]}" 
            for x in bm25_results[1:100] # Take the leftover 99, so we get 99 bm25 closely related neighbors
        ]
        score_clean = self.get_target_score(question, target_passage, original_neighbors)
        
        # DIRTY INFERENCE
        junk_neighbors = self.generate_junk_context(count=99)
        score_dirty = self.get_target_score(question, target_passage, junk_neighbors)

        # Print all resutls
        print(f"\nQuery: {question}")
        print(f"Target Passage Start: {target_passage[:50]}...")
        print(f"-"*30)
        print(f"Score in Context A (Real Neighbors):   {score_clean:.4f}")
        print(f"Score in Context B (Junk Neighbors):   {score_dirty:.4f}")
        print(f"-"*30)
        
        delta = abs(score_clean - score_dirty)
        print(f"Stability Delta: {delta:.4f}")
        
        if delta > 0.1: # I assume its between 0-1 here, again not sure?
            print("CONCLUSION: FAIL. The model is not robust to noise.")
        else:
            print("CONCLUSION: PASS. The model scored the passage consistently.")

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
    
    # Dummy args for compatibility
    parser.add_argument('--question_text_key', default='q_text')
    parser.add_argument('--firststage_result_key', default='bm25_results')
    parser.add_argument('--title_key', default='title')
    parser.add_argument('--text_key', default='text')

    args = parser.parse_args()
    
    tester = RobustnessTester(args)
    tester.run_anchor_noise_test()