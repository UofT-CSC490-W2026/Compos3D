from datasets import load_dataset
from tasks.common import Task


class MetaMathQA(Task):
    def __init__(self, stop=None, **kwargs):
        super().__init__(**kwargs)
        ds = load_dataset("meta-math/MetaMathQA", split="train")
        ds = ds.filter(
            lambda row: bool(row["query"].strip()) and bool(row["response"].strip())
        )
        ds = ds.shuffle(seed=42)
        if stop is not None:
            ds = ds.select(range(min(stop, len(ds))))
        self.ds = ds
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        return {
            "messages": [
                {"role": "user", "content": row["query"].strip()},
                {"role": "assistant", "content": row["response"].strip()},
            ]
        }
