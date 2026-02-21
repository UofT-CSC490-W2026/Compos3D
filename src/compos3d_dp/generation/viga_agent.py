"""
VIGA-style Agent Loop for 3D Scene Generation

Implements the Generator → Execute → Verify → Iterate loop from VIGA.

Generator: LLM writes Blender Python code
Execute: Run code in Blender, render scene
Verify: Critic evaluates renders
Iterate: Feed back and improve

Based on: external/VIGA/agents/ and external/VIGA/runners/
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime, timezone

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from compos3d_dp.generation.blender_executor import (
    BlenderExecutor,
)
from compos3d_dp.generation.critic import SceneCritic, SceneCritiqueScore
from compos3d_dp.generation.hypogenic_ucb import HypothesisBank, Hypothesis


@dataclass
class AgentIteration:
    """Single iteration in the agent loop"""

    iteration: int
    thought: str
    code: str
    execution_result: Optional[Dict] = None
    critique_score: Optional[Dict] = None
    hypotheses_used: List[str] = None

    def __post_init__(self):
        if self.hypotheses_used is None:
            self.hypotheses_used = []


@dataclass
class AgentRun:
    """Complete agent run with all iterations"""

    run_id: str
    prompt: str
    iterations: List[AgentIteration]
    final_score: float
    success: bool
    created_at: str
    blend_file: Optional[str] = None


class VIGAAgent:
    """
    VIGA-style agent for iterative 3D scene generation.

    Usage:
        agent = VIGAAgent(
            openai_api_key="sk-...",
            blender_command="blender"
        )

        run = agent.generate_scene(
            prompt="A cozy living room with warm lighting",
            max_iterations=5
        )

        print(f"Final score: {run.final_score}")
    """

    def __init__(
        self,
        openai_api_key: str,
        blender_command: str = "blender",
        output_dir: str | Path = "output/viga_runs",
        use_hypothesis_bank: bool = True,
        model: str = "gpt-4",
    ):
        """
        Initialize VIGA agent.

        Args:
            openai_api_key: OpenAI API key
            blender_command: Path to Blender executable
            output_dir: Directory for outputs
            use_hypothesis_bank: Use HypoGeniC UCB bandit
            model: OpenAI model to use
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Run: pip install openai")

        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model

        self.executor = BlenderExecutor(
            blender_command=blender_command,
            output_dir=Path(output_dir) / "blender_runs",
        )

        self.critic = SceneCritic()

        self.use_hypothesis_bank = use_hypothesis_bank
        if use_hypothesis_bank:
            self.hypothesis_bank = HypothesisBank()
        else:
            self.hypothesis_bank = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_count = 0

    def generate_scene(
        self,
        prompt: str,
        initial_blend_file: Optional[Path] = None,
        max_iterations: int = 5,
        target_score: float = 0.75,
    ) -> AgentRun:
        """
        Generate a 3D scene through iterative refinement.

        Args:
            prompt: Natural language description of desired scene
            initial_blend_file: Starting .blend file (creates empty if None)
            max_iterations: Maximum refinement iterations
            target_score: Stop if score exceeds this

        Returns:
            AgentRun with all iterations and results
        """
        self.run_count += 1
        run_id = f"run_{self.run_count:04d}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"   Prompt: {prompt}")
        print(f"   Max iterations: {max_iterations}")

        iterations: List[AgentIteration] = []
        current_blend_file = initial_blend_file
        conversation_history = []

        for i in range(max_iterations):
            # Select hypotheses if using UCB
            hypotheses = []
            if self.hypothesis_bank:
                hypotheses = self.hypothesis_bank.select_top_k(k=3)
                print("   Selected hypotheses:")
                for h in hypotheses:
                    print(f"      - {h.text}")

            # Generate code
            thought, code = self._generate_code(
                prompt=prompt,
                conversation_history=conversation_history,
                hypotheses=hypotheses,
            )

            print(f"   Thought: {thought[:100]}...")

            # Execute code
            exec_result = self.executor.execute(
                code=code,
                blend_file=current_blend_file,
                render=True,
            )

            if not exec_result.success:
                conversation_history.append(
                    {
                        "role": "user",
                        "content": f"Error: {exec_result.error_message}\nPlease fix the code.",
                    }
                )

                iterations.append(
                    AgentIteration(
                        iteration=i + 1,
                        thought=thought,
                        code=code,
                        execution_result={
                            "success": False,
                            "error": exec_result.error_message,
                        },
                        hypotheses_used=[h.hypothesis_id for h in hypotheses],
                    )
                )
                continue

            # Update blend file for next iteration
            current_blend_file = exec_result.blend_file

            # Evaluate renders
            if exec_result.rendered_images:
                critique = self.critic.evaluate(
                    image_path=exec_result.rendered_images[0],
                    prompt=prompt,
                )

                print(f"      Quality: {critique.visual_quality:.3f}")
                print(f"      Physics: {critique.physical_plausibility:.3f}")
                print(f"      Prompt: {critique.prompt_adherence:.3f}")

                # Update hypothesis bank
                if self.hypothesis_bank:
                    for h in hypotheses:
                        self.hypothesis_bank.update_hypothesis(
                            h.hypothesis_id,
                            reward=critique.overall,
                            success=critique.overall > 0.6,
                        )

                # Store iteration
                iterations.append(
                    AgentIteration(
                        iteration=i + 1,
                        thought=thought,
                        code=code,
                        execution_result={
                            "success": True,
                            "num_images": len(exec_result.rendered_images),
                        },
                        critique_score=asdict(critique),
                        hypotheses_used=[h.hypothesis_id for h in hypotheses],
                    )
                )

                # Check if target reached
                if critique.overall >= target_score:
                    break

                # Add feedback to conversation
                feedback = self._generate_feedback(critique, prompt)
                conversation_history.append({"role": "user", "content": feedback})

            else:
                print("   ⚠️  No renders produced")
                iterations.append(
                    AgentIteration(
                        iteration=i + 1,
                        thought=thought,
                        code=code,
                        execution_result={"success": True, "num_images": 0},
                        hypotheses_used=[h.hypothesis_id for h in hypotheses],
                    )
                )

        # Final score
        final_score = 0.0
        if iterations and iterations[-1].critique_score:
            final_score = iterations[-1].critique_score.get("overall", 0.0)

        run = AgentRun(
            run_id=run_id,
            prompt=prompt,
            iterations=iterations,
            final_score=final_score,
            success=final_score >= target_score,
            created_at=datetime.now(timezone.utc).isoformat(),
            blend_file=str(current_blend_file) if current_blend_file else None,
        )

        # Save run
        self._save_run(run, run_dir)

        print(f"   Final score: {final_score:.3f}")
        print(f"   Iterations: {len(iterations)}")
        print(f"   Output: {run_dir}")

        return run

    def _generate_code(
        self,
        prompt: str,
        conversation_history: List[Dict],
        hypotheses: List[Hypothesis],
    ) -> Tuple[str, str]:
        """Generate Blender Python code using LLM"""

        system_prompt = """You are an expert Blender Python programmer. 
Generate Blender Python code to create 3D scenes based on natural language descriptions.

Your code should:
1. Clear the default scene
2. Create objects, materials, lighting
3. Set up camera(s)
4. Follow best practices for realistic scenes

Format your response as:
THOUGHT: <your reasoning>
CODE:
```python
<your code>
```
"""

        # Add hypotheses to system prompt
        if hypotheses:
            hyp_text = "\n".join([f"- {h.text}" for h in hypotheses])
            system_prompt += f"\n\nConsider these principles:\n{hyp_text}"

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        messages.extend(conversation_history)

        # Add current prompt
        messages.append(
            {"role": "user", "content": f"Create a Blender scene: {prompt}"}
        )

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )

        content = response.choices[0].message.content

        # Parse thought and code
        thought, code = self._parse_response(content)

        # Add to history
        conversation_history.append({"role": "assistant", "content": content})

        return thought, code

    def _parse_response(self, content: str) -> Tuple[str, str]:
        """Parse LLM response into thought and code"""
        lines = content.split("\n")

        thought = ""
        code = ""
        in_code = False

        for line in lines:
            if line.startswith("THOUGHT:"):
                thought = line.replace("THOUGHT:", "").strip()
            elif line.startswith("```python"):
                in_code = True
            elif line.startswith("```") and in_code:
                in_code = False
            elif in_code:
                code += line + "\n"

        if not code:
            # Fallback: treat everything after CODE: as code
            if "CODE:" in content:
                code = content.split("CODE:")[-1].strip()

        return thought, code

    def _generate_feedback(self, critique: SceneCritiqueScore, prompt: str) -> str:
        """Generate feedback for the agent based on critique"""
        feedback = f"Current scene score: {critique.overall:.2f}\n\n"

        if critique.prompt_adherence < 0.6:
            feedback += (
                f"Scene doesn't match prompt well ({critique.prompt_adherence:.2f}). "
            )
            feedback += f"Make sure to include key elements from: {prompt}\n"

        if critique.visual_quality < 0.6:
            feedback += f"Visual quality is low ({critique.visual_quality:.2f}). "
            feedback += "Improve lighting, materials, or camera angle.\n"

        if critique.physical_plausibility < 0.6:
            feedback += f"Scene looks physically unrealistic ({critique.physical_plausibility:.2f}). "
            feedback += "Check object positions, physics, and scale.\n"

        if critique.overall < 0.7:
            feedback += "\nPlease improve the scene."
        else:
            feedback += "\nScene is good! Try minor refinements."

        return feedback

    def _save_run(self, run: AgentRun, run_dir: Path):
        """Save run metadata to JSON"""
        metadata = asdict(run)

        metadata_file = run_dir / "run_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    import os

    # Test the VIGA agent
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        exit(1)

    print("🧪 Testing VIGA Agent...")

    agent = VIGAAgent(
        openai_api_key=api_key,
        blender_command="blender",
        output_dir="output/test_viga_agent",
    )

    run = agent.generate_scene(
        prompt="A simple scene with a red cube on a wooden table",
        max_iterations=3,
    )

    print(f"   Final score: {run.final_score:.3f}")
    print(f"   Success: {run.success}")
