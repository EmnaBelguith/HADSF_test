#!/usr/bin/env python3
# compute_adr.py
# Updated: skip entries with empty 'sentence' and only compute ADR on records with non-empty aspects.

import json
import argparse

def load_vocab(vocab_path):
    """
    Load the established vocabulary A* from a text file.
    Each line in the file should contain one aspect.
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def compute_adr(data_path, vocab):
    """
    Compute ADR = (1/|S|) * sum_{records} (num_aspects_not_in_vocab / total_aspects_in_record).
    Only processes records where 'sentence' is non-empty.
    """
    total_rate = 0.0
    count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            triples = record.get('sentence', [])
            if not triples:
                # Skip entries with empty 'sentence'
                continue
            s_size = len(triples)
            # Count aspects not in A*
            num_hallucin = sum(1 for a, _, _ in triples if a not in vocab)
            total_rate += num_hallucin / s_size
            count += 1

    if count == 0:
        return None, 0
    return total_rate / count, count

def main():
    parser = argparse.ArgumentParser(description='Compute Aspect Deviation Rate (ADR).')
    parser.add_argument('--data_path',  required=True,
                        help='Path to the JSONL data file.')
    parser.add_argument('--vocab_path', required=True,
                        help='Path to the vocabulary file (one aspect per line).')
    args = parser.parse_args()

    vocab = load_vocab(args.vocab_path)
    adr, count = compute_adr(args.data_path, vocab)

    if count == 0:
        print('No records with extracted aspects to process.')
    else:
        print(f'ADR: {adr:.6f}')
        print(f'Processed {count} records (empty sentences were skipped).')

if __name__ == '__main__':
    main()
