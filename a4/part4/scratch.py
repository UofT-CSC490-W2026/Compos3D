from datasets import load_dataset
from pprint import pprint

ds = load_dataset("openai/gsm8k", "main")

print(ds)
print()

example = ds["train"][0]

print("Top-level example keys:")
print(example.keys())
print()

print("Question type:", type(example["question"]))
print("Answer type:", type(example["answer"]))
print()

print("Question:")
print(example["question"])
print()

print("Answer:")
print(example["answer"])
print()

print("Full example:")
pprint(example)