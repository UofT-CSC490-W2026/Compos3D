from datasets import load_dataset
from tasks.common import Task


class UltraChat200k(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train_sft", "test_sft"], (
            f"UltraChat200k split must be 'train_sft' or 'test_sft', got '{split}'"
        )
        self.ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split).shuffle(
            seed=42
        )
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]

        assert len(messages) >= 1, (
            "UltraChat200k: conversation must have at least 1 message"
        )

        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages

        assert len(rest_messages) >= 2, (
            f"UltraChat200k: expected ≥2 non-system messages, got {len(rest_messages)}"
        )
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, (
                f"UltraChat200k: message {i} has role '{message['role']}' "
                f"but expected '{expected_role}'"
            )
            assert isinstance(message["content"], str), (
                f"UltraChat200k: message {i} content must be a string"
            )

        return {"messages": messages}
