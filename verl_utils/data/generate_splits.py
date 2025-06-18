"""
Preprocess the MATH-Hard dataset to parquet format
"""

import os
from datasets import load_dataset
import argparse

train_data_path = 'data/train/MATH_Hard.jsonl'
val_data_path = 'data/train/MATH_val.jsonl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/train')
    args = parser.parse_args()

    train_dataset = load_dataset("json", data_files={"train": train_data_path}, split="train")
    val_dataset = load_dataset("json", data_files={"val": val_data_path}, split="val")

    def process_fn_train(example, idx):
        data = {
            "data_source": train_data_path,
            "prompt": example['messages'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['answer']
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': example['answer'],
                "question": example['problem'],
            }
        }
        return data

    def process_fn_test(example, idx):
        data = {
            "data_source": val_data_path,
            "prompt": example['messages'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['answer']
            },
            "extra_info": {
                'split': 'test',
                'index': idx,
                'answer': example['answer'],
                "question": example['problem'],
            }
        }
        return data

    train_dataset = train_dataset.map(function=process_fn_train, with_indices=True)
    test_dataset = val_dataset.map(function=process_fn_test, with_indices=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
