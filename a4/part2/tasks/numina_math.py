from datasets import load_dataset
from tasks.common import Task


class NuminaMathCoT(Task):
    def __init__(self, stop=100_000, **kwargs):
        super().__init__(**kwargs)
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        ds = ds.filter(
            lambda row: (
                len(row["messages"]) == 2
                and row["messages"][0]["role"] == "user"
                and row["messages"][1]["role"] == "assistant"
                and bool(row["messages"][0]["content"].strip())
                and bool(row["messages"][1]["content"].strip())
            )
        )
        ds = ds.shuffle(seed=42)
        if stop is not None:
            ds = ds.select(range(min(stop, len(ds))))
        self.ds = ds
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.ds[index]["messages"]}
