from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from compos3d.config import load_experiment_config
from compos3d.hypothesis.engine import run_vertical_inference
from compos3d.llm.bedrock import BedrockChatClient

DEFAULT_BANK = str(PROJECT_ROOT / "artifacts/training/claude_qwen/hypothesis_bank.json")
DEFAULT_CONFIG = str(PROJECT_ROOT / "train_configs/compos3d.json")
DEMO_OUT = PROJECT_ROOT / "artifacts" / "demo"
STRATEGIES = ["filter_and_weight", "joint_top_k"]

EXAMPLES = [
    "A cozy dining room with a round table, four chairs, a rug, and a lamp",
    "A modern dining room with a long rectangular table, six chairs, and pendant lamps",
    "A rustic dining room with a wooden table, mismatched chairs, and a floor lamp",
    "A minimalist dining room with a glass table, two chairs, and a window",
    "A dining room with a dining table, several chairs, three lamps, and a rug",
]

_captured_calls: list[dict] = []
_original_converse = BedrockChatClient.converse


def _intercepting_converse(self, *, messages, **kwargs):
    result = _original_converse(self, messages=messages, **kwargs)
    try:
        user_text = messages[0]["content"][0]["text"]
        raw_text = result["output"]["message"]["content"][0]["text"]
        _captured_calls.append({"prompt": user_text, "response": raw_text})
    except Exception:
        pass
    return result


BedrockChatClient.converse = _intercepting_converse


def _heuristic_config_path(config_path: str) -> Path:
    cfg = load_experiment_config(Path(config_path) if config_path.strip() else None)
    cfg.critic.mode = "heuristic"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="compos3d_demo_cfg_", delete=False
    )
    tmp.write(cfg.model_dump_json())
    tmp.close()
    return Path(tmp.name)


def respond(message, history, bank_path, config_path, strategy, render):
    if not message.strip():
        yield history, gr.update(), gr.update()
        return

    run_id = f"run_{int(time.time() * 1000)}"
    out_dir = DEMO_OUT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    history = list(history) + [{"role": "user", "content": message}]
    yield history, gr.update(), gr.update()

    _captured_calls.clear()
    patched_config = _heuristic_config_path(config_path)
    try:
        run_vertical_inference(
            bank_path=Path(bank_path),
            prompt=message.strip(),
            output_dir=out_dir,
            config_path=patched_config,
            inference_strategy=strategy,
            render_scene=render,
            render_view_samples=100,
            render_video_samples=100,
        )
    except Exception as exc:
        history = history + [{"role": "assistant", "content": str(exc)}]
        yield history, [], None
        return
    finally:
        patched_config.unlink(missing_ok=True)

    for call in _captured_calls:
        history = history + [
            {"role": "assistant", "content": call["prompt"]},
            {"role": "assistant", "content": call["response"]},
        ]

    sp_path = out_dir / "scene_program.json"
    sp = json.loads(sp_path.read_text()) if sp_path.exists() else {}
    history = history + [{"role": "assistant", "content": json.dumps(sp, indent=2)}]

    images: list[str] = []
    views_dir = out_dir / "scene" / "views"
    if views_dir.exists():
        images = sorted(str(p) for p in views_dir.glob("view_*.png") if p.exists())

    video_path = out_dir / "scene" / "video.mp4"
    video = str(video_path) if video_path.exists() else None

    yield history, images, video


with gr.Blocks(title="Compos3D") as demo:
    gr.Markdown("# Compos3D — 3D Scene Generation")
    gr.Markdown(
        "Describe a room to generate a 3D scene. Send follow-up messages to edit or refine."
    )

    with gr.Accordion("Settings", open=False):
        bank_path = gr.Textbox(label="Hypothesis bank", value=DEFAULT_BANK)
        config_path = gr.Textbox(label="Config path", value=DEFAULT_CONFIG)
        strategy = gr.Dropdown(
            choices=STRATEGIES, value="filter_and_weight", label="Inference strategy"
        )
        render = gr.Checkbox(label="Render scene with Blender", value=True)

    chatbot = gr.Chatbot(
        label="Generation",
        height=560,
        render_markdown=False,
        placeholder="Describe a room and hit Send.",
    )

    with gr.Row():
        prompt_box = gr.Textbox(label="Prompt", lines=1, scale=8, submit_btn=False)
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear", variant="secondary", size="sm")

    gr.Examples(examples=EXAMPLES, inputs=prompt_box, label="Example prompts")

    with gr.Row():
        images_out = gr.Gallery(label="Rendered views", columns=4, scale=3)
        video_out = gr.Video(label="Orbital video", scale=1)

    send_btn.click(
        fn=respond,
        inputs=[prompt_box, chatbot, bank_path, config_path, strategy, render],
        outputs=[chatbot, images_out, video_out],
    ).then(fn=lambda: "", outputs=prompt_box)

    prompt_box.submit(
        fn=respond,
        inputs=[prompt_box, chatbot, bank_path, config_path, strategy, render],
        outputs=[chatbot, images_out, video_out],
    ).then(fn=lambda: "", outputs=prompt_box)

    clear_btn.click(fn=lambda: ([], [], None), outputs=[chatbot, images_out, video_out])


if __name__ == "__main__":
    demo.launch()
