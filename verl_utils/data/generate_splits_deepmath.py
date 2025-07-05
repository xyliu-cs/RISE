"""
Preprocess the deepmath dataset to parquet format
"""

import os
import datasets
import argparse

train_source = 'xiaoyuanliu/DeepMath-10K'
train_split = 'train'
val_source="xiaoyuanliu/math-gen-critique"
val_split="math_val"

my_system_prompt = 'Please reason step by step, and put your final answer within \\boxed{}.'


def format_messages(question, system_prompt=my_system_prompt):
    if system_prompt:
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    else:
        message = [
            {"role": "user", "content": question}
        ]

    return message



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/deepmath')
    parser.add_argument('--add_message', action='store_true', help='Whether to add message column to the dataset')
    args = parser.parse_args()

    train_dataset = datasets.load_dataset(train_source, split=train_split)
    val_dataset = datasets.load_dataset(val_source, split=val_split)

    if args.add_message:
        train_dataset = train_dataset.map(
            lambda x: {'messages': format_messages(x['question'], my_system_prompt)},
            desc='Formatting messages for train dataset'
        )

    def process_fn_train(example, idx):
        data = {
            "data_source": train_source,
            "prompt": example['messages'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['final_answer']
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': example['final_answer'],
                "question": example['question'],
            }
        }
        return data

    def process_fn_test(example, idx):
        data = {
            "data_source": val_source,
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
    # preview the first few entries
    print('-'* 50)
    print("Train dataset sample:")
    print(train_dataset[5])
    print('-'* 50)
    print("Test dataset sample:")
    print(test_dataset[5])
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))

