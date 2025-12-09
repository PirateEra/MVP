import torch
import argparse
import numpy as np
from evaluation import MVPEvaluator

class RobustnessTester(MVPEvaluator):
    
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

    def run_isolation_test(self):
        print(f"--- Starting Dilution/Isolation Test ---")
        
        # We test on the first query (Index 0) of our test file
        TARGET_IDX = 0
        instance = self.test_file[TARGET_IDX]
        question = instance[self.args.question_text_key]
        bm25_results = instance[self.args.firststage_result_key]
        
        # Limit to topk args if necessary, usually 100
        initial_candidates = bm25_results[:self.args.topk]
        original_txt = [f"{x[self.args.title_key]} {x[self.args.text_key]}" for x in initial_candidates]
        
        print(f"\nPhase 1: Ranking specific context (Top {len(original_txt)} candidates)...")
        
        # Rank the full list (0 to 99)
        p1_indices, p1_scores = self.get_ranking_scores(question, original_txt)
        
        # Extract the winners (The Top K, e.g., Top 10)
        k = self.args.retrieve_k
        top_k_indices = p1_indices[:k] 
        
        print(f"Top {k} Winners identified (Indices): {top_k_indices}")
        
        # Get the actual text content of these winners to form a new, isolated batch
        winner_texts = [original_txt[i] for i in top_k_indices]
        
        # Phase 2: Isolation Test
        # We rank ONLY these 10. In a listwise world, removing the 90 "losers" might change how the model views the "winners".
        print(f"\nPhase 2: Ranking ISOLATED context (Only the Top {k})...")
        p2_indices, p2_scores = self.get_ranking_scores(question, winner_texts)
        
        print(f"\n--- RESULTS ---")
        print(f"Query: {question}")
        
        # Ideal order is simply 0 to k-1, because we constructed winner_texts in the order of the Phase 1 ranking
        ideal_order = np.arange(k)
        
        print(f"Phase 1 Order (Implicit): {ideal_order}")
        print(f"Phase 2 Order (Isolated): {p2_indices}")
        
        if np.array_equal(p2_indices, ideal_order):
            print("\nCONCLUSION: PASS (Consistent).")
            print("Removing the bottom 90 candidates did NOT change the relative order of the Top 10.")
        else:
            print("\nCONCLUSION: FAIL (Dilution/Context Effect Detected).")
            print("When observed in isolation, the model changed its mind about the order.")
            
            for rank, original_idx in enumerate(p2_indices):
                if rank != original_idx:
                    print(f" -> The item that was #{original_idx + 1} in the large batch moved to rank #{rank + 1} in isolation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='bulbna/MVP-base', type=str)
    parser.add_argument('--input_path', type=str, default='./eval_data/dl19.jsonl')
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--retrieve_k', default=10, type=int)
    
    # Dummy args
    parser.add_argument('--n_special_tokens', default=1, type=int)
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--padding', default='max_length', type=str)
    parser.add_argument('--max_gen_length', default=10, type=int)
    parser.add_argument('--measure_flops', action='store_true')
    parser.add_argument('--question_text_key', default='q_text')
    parser.add_argument('--firststage_result_key', default='bm25_results')
    parser.add_argument('--title_key', default='title')
    parser.add_argument('--text_key', default='text')

    args = parser.parse_args()
    
    tester = RobustnessTester(args)
    tester.run_isolation_test()


# --- Starting Dilution/Isolation Test ---

# Phase 1: Ranking specific context (Top 100 candidates)...
# Top 10 Winners identified (Indices): [ 1  2 44 31 28 52 21 38 24 55]

# Phase 2: Ranking ISOLATED context (Only the Top 10)...

# --- RESULTS ---
# Query: how long is life cycle of flea
# Phase 1 Order (Implicit): [0 1 2 3 4 5 6 7 8 9]
# Phase 2 Order (Isolated): [1 0 2 3 4 6 5 7 9 8]

# CONCLUSION: FAIL (Dilution/Context Effect Detected).
# When observed in isolation, the model changed its mind about the order.
#  -> The item that was #2 in the large batch moved to rank #1 in isolation.
#  -> The item that was #1 in the large batch moved to rank #2 in isolation.
#  -> The item that was #7 in the large batch moved to rank #6 in isolation.
#  -> The item that was #6 in the large batch moved to rank #7 in isolation.
#  -> The item that was #10 in the large batch moved to rank #9 in isolation.
#  -> The item that was #9 in the large batch moved to rank #10 in isolation.

# --- Starting Dilution/Isolation Test ---

# Phase 1: Ranking specific context (Top 100 candidates)...
# Top 10 Winners identified (Indices): [44 13  6 12 65 90 76  2 10 43]

# Phase 2: Ranking ISOLATED context (Only the Top 10)...

# --- RESULTS ---
# Query: are naturalization records public information
# Phase 1 Order (Implicit): [0 1 2 3 4 5 6 7 8 9]
# Phase 2 Order (Isolated): [0 1 2 3 4 5 6 7 8 9]

# CONCLUSION: PASS (Consistent).
# Removing the bottom 90 candidates did NOT change the relative order of the Top 10.
# Conclusion: the "noise" candidates act as distractors that flip the relationship between the top candidates 
# (#1 vs #2), proving the model's ranking of the winners is fragile. Therefore the relative preferences are inconsistent.