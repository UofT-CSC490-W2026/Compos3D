import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from compos3d.config import CriticConfig
from compos3d.evaluation.critic import CriticUnavailableError
from compos3d.paper.judges import (
    _extract_json_payload,
    BedrockPairwiseJudge,
    _load_open_clip_components,
    clip_directional_similarity,
)


@pytest.mark.unit
def test_extract_json_payload():
    assert _extract_json_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json_payload('  {"a": 1}  ') == {"a": 1}

    with pytest.raises(ValueError, match="Could not locate JSON object"):
        _extract_json_payload("not json")

    with pytest.raises(ValueError, match="Could not locate JSON object"):
        _extract_json_payload("}{")


@pytest.mark.unit
def test_bedrock_pairwise_judge_image_format():
    judge = BedrockPairwiseJudge(CriticConfig())
    assert judge._image_format(Path("img.jpg")) == "jpeg"
    assert judge._image_format(Path("img.jpeg")) == "jpeg"
    assert judge._image_format(Path("img.png")) == "png"
    assert judge._image_format(Path("img.webp")) == "webp"
    assert judge._image_format(Path("img.gif")) == "png"


@pytest.mark.unit
def test_bedrock_pairwise_judge_run_json_prompt(monkeypatch, tmp_path):
    judge = BedrockPairwiseJudge(CriticConfig())

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": '{"success": true}'}]}}
    }
    judge.client = mock_client

    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake_image_data")

    res = judge._run_json_prompt(prompt_text="test prompt", image_paths=[img_path])
    assert res == {"success": True}

    # Test exceptions
    mock_client.converse.side_effect = Exception("AWS login required")
    with pytest.raises(
        CriticUnavailableError, match="credentials are unavailable or expired"
    ):
        judge._run_json_prompt(prompt_text="test prompt", image_paths=[])

    mock_client.converse.side_effect = Exception("Other error")
    with pytest.raises(CriticUnavailableError, match="Bedrock judge request failed"):
        judge._run_json_prompt(prompt_text="test prompt", image_paths=[])


@pytest.mark.unit
def test_bedrock_pairwise_judge_generation_pair(monkeypatch):
    judge = BedrockPairwiseJudge(CriticConfig())
    monkeypatch.setattr(
        judge, "_run_json_prompt", lambda **kwargs: {"layout_coherence": "A"}
    )

    res = judge.judge_generation_pair(
        prompt="prompt",
        candidate_a_paths=[],
        candidate_b_paths=[],
        method_a="A",
        method_b="B",
    )
    assert res == {"layout_coherence": "A"}


@pytest.mark.unit
def test_bedrock_pairwise_judge_edit_pair(monkeypatch):
    judge = BedrockPairwiseJudge(CriticConfig())
    monkeypatch.setattr(
        judge, "_run_json_prompt", lambda **kwargs: {"edit_success": 1.0}
    )

    res = judge.judge_edit_pair(
        base_prompt="base",
        edited_prompt="edit",
        edit_instruction="do it",
        before_paths=[],
        after_paths=[],
    )
    assert res == {"edit_success": 1.0}


@pytest.mark.unit
def test_load_open_clip_components_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "open_clip", None)
    import compos3d.paper.judges as mod

    mod._load_open_clip_components.cache_clear()
    assert mod._load_open_clip_components() is None


@pytest.mark.unit
def test_clip_directional_similarity_missing_clip(monkeypatch):
    import compos3d.paper.judges as mod

    mod._load_open_clip_components.cache_clear()
    monkeypatch.setattr(mod, "_load_open_clip_components", lambda: None)

    res = clip_directional_similarity(
        before_image_paths=[Path("a")],
        after_image_paths=[Path("b")],
        before_text="a",
        after_text="b",
    )
    assert res is None


@pytest.mark.unit
def test_clip_directional_similarity(monkeypatch, tmp_path):
    import compos3d.paper.judges as mod

    mod._load_open_clip_components.cache_clear()

    class DummyModel:
        def eval(self):
            pass

        def to(self, device):
            pass

        def encode_image(self, batch):
            class DummyTensor:
                def __init__(self, val):
                    self.val = val

                def norm(self, **kwargs):
                    return DummyTensor(1.0)

                def mean(self, **kwargs):
                    return self

                def __truediv__(self, other):
                    return self

                def __sub__(self, other):
                    return DummyTensor(self.val - other.val)

                def item(self):
                    return float(self.val)

                def sum(self):
                    return self

                def __mul__(self, other):
                    return DummyTensor(self.val * other.val)

            return DummyTensor(1.0)

        def encode_text(self, tokenized):
            class DummyTensor:
                def __init__(self, val):
                    self.val = val

                def norm(self, **kwargs):
                    return DummyTensor(1.0)

                def __truediv__(self, other):
                    return self

                def __getitem__(self, idx):
                    return DummyTensor(float(idx))

                def __sub__(self, other):
                    return DummyTensor(self.val - other.val)

                def item(self):
                    return float(self.val)

                def __mul__(self, other):
                    return DummyTensor(self.val * other.val)

                def sum(self):
                    return self

            return DummyTensor(1.0)

    class DummyTorch:
        cuda = type("cuda", (), {"is_available": staticmethod(lambda: False)})()

        class no_grad:
            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        def stack(self, tensors):
            class DummyBatch:
                def to(self, device):
                    return self

            return DummyBatch()

    class DummyTokenizer:
        def __init__(self, val):
            pass

        def to(self, device):
            return self

    monkeypatch.setattr(
        mod,
        "_load_open_clip_components",
        lambda: (
            DummyModel(),
            lambda x: x,
            lambda x: DummyTokenizer(x),
            DummyTorch(),
            "cpu",
        ),
    )

    # We need to test the logic where image directions are computed
    # Create fake images
    from PIL import Image

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (10, 10)).save(a)
    Image.new("RGB", (10, 10)).save(b)

    res = mod.clip_directional_similarity(
        before_image_paths=[a], after_image_paths=[b], before_text="a", after_text="b"
    )
    assert res == 0.0

    # Missing 179, 183, 198: empty lists or non-existent files
    res2 = mod.clip_directional_similarity(
        before_image_paths=[tmp_path / "non_existent.png"],
        after_image_paths=[b],
        before_text="a",
        after_text="b",
    )
    assert res2 is None

    # Missing 206: norm is 0.0
    class DummyModel2(DummyModel):
        def encode_image(self, batch):
            class DummyTensor2:
                def __init__(self, val=0):
                    self.val = val

                def norm(self, **kwargs):
                    return DummyTensor2(0.0)

                def mean(self, **kwargs):
                    return self

                def __truediv__(self, other):
                    return self

                def __sub__(self, other):
                    return DummyTensor2(0.0)

                def item(self):
                    return float(self.val)

            return DummyTensor2(1.0)

    monkeypatch.setattr(
        mod,
        "_load_open_clip_components",
        lambda: (
            DummyModel2(),
            lambda x: x,
            lambda x: DummyTokenizer(x),
            DummyTorch(),
            "cpu",
        ),
    )
    assert (
        mod.clip_directional_similarity(
            before_image_paths=[a],
            after_image_paths=[b],
            before_text="a",
            after_text="b",
        )
        is None
    )


@pytest.mark.unit
def test_load_open_clip_components_success(monkeypatch):
    import sys

    class MockOpenClip:
        def create_model_and_transforms(self, *args, **kwargs):
            return "model", "transforms", "preprocess"

        def get_tokenizer(self, *args, **kwargs):
            return "tokenizer"

    monkeypatch.setitem(sys.modules, "open_clip", MockOpenClip())

    class MockTorch:
        cuda = type("cuda", (), {"is_available": staticmethod(lambda: False)})()

    monkeypatch.setitem(sys.modules, "torch", MockTorch())

    import compos3d.paper.judges as mod

    mod._load_open_clip_components.cache_clear()

    # Patch the returned mock model so it has eval and to
    mock_oc = MockOpenClip()

    class MockModel:
        def eval(self):
            pass

        def to(self, device):
            pass

    mock_oc.create_model_and_transforms = lambda *a, **kw: (MockModel(), None, None)
    monkeypatch.setitem(sys.modules, "open_clip", mock_oc)
    mod._load_open_clip_components.cache_clear()

    res = mod._load_open_clip_components()
    assert res is not None
    assert len(res) == 5
