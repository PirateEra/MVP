import json
import random
import copy
import argparse

def main():
    parser = argparse.ArgumentParser(description='shuffle and rev order of the jsonl files')
    parser.add_argument("--input", type=str, help="path to jsonl input file")
    parser.add_argument("--output_rev", type=str, help="jsonl output file name where the passages are reversed")
    parser.add_argument("--output_shuffle", type=str, help="jsonl output file name where the passages are shuffled")

    args = parser.parse_args()
    
    # open jsonl file and reverse/shuffle the passages of each question
    with open(args.input, 'r') as jsonl:
        data = [json.loads(line) for line in jsonl]
        # make copies of the original data 
        data_copy_rev = copy.deepcopy(data)
        data_copy_shuffle = copy.deepcopy(data)
        # loop through all questions
        for question in range(len(data)): 
            # reverse the order of the passages of one question
            bm25_res = data[question]['bm25_results']
            reversed_bm25_res = bm25_res[::-1] 
            data_copy_rev[question]['bm25_results'] = reversed_bm25_res
            # randomly shuffle the order of the passages of one question
            copy_bm25_res = copy.deepcopy(bm25_res)
            random.shuffle(copy_bm25_res)
            data_copy_shuffle[question]['bm25_results'] = copy_bm25_res

    # make a new jsonl file with the reversed candidates 
    with open(args.output_rev, 'w') as outfile:
        for line in data_copy_rev:
            json.dump(line, outfile)
            outfile.write('\n')

    # make a new jsonl file with the randomly shuffled candidates 
    with open(args.output_shuffle, 'w') as outfile:
        for line in data_copy_shuffle:
            json.dump(line, outfile)
            outfile.write('\n')


if __name__ == "__main__":
    main()
