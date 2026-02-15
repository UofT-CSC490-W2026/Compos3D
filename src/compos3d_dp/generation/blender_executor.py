"""
Blender Executor - adapted from VIGA

Executes Blender Python code in headless mode and captures renders.
Simpler version of VIGA's MCP executor for our pipeline.

Based on: external/VIGA/tools/blender/exec.py
"""

from __future__ import annotations
import subprocess
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BlenderExecutionResult:
    """Result from executing Blender code"""

    success: bool
    rendered_images: List[Path]
    stdout: str
    stderr: str
    blend_file: Optional[Path] = None
    error_message: Optional[str] = None


class BlenderExecutor:
    """
    Executes Blender Python scripts and renders scenes.

    Usage:
        executor = BlenderExecutor(
            blender_command="blender",
            output_dir="output/blender_runs"
        )

        code = '''
        import bpy
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        '''

        result = executor.execute(
            code=code,
            blend_file="scene.blend"
        )
    """

    def __init__(
        self,
        blender_command: str = "blender",
        output_dir: str | Path = "output/blender_runs",
        gpu_devices: Optional[str] = None,
    ):
        """
        Initialize Blender executor.

        Args:
            blender_command: Path to Blender executable
            output_dir: Directory to store outputs
            gpu_devices: Comma-separated GPU IDs (e.g., "0,1")
        """
        self.blender_command = blender_command
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_devices = gpu_devices

        self.run_count = 0

    def execute(
        self,
        code: str,
        blend_file: Optional[str | Path] = None,
        render: bool = True,
        resolution: Tuple[int, int] = (512, 512),
        save_blend: bool = True,
    ) -> BlenderExecutionResult:
        """
        Execute Blender Python code.

        Args:
            code: Python code to execute in Blender
            blend_file: Input .blend file (creates empty if None)
            render: Whether to render the scene
            resolution: Image resolution (width, height)
            save_blend: Save the resulting .blend file

        Returns:
            BlenderExecutionResult with success status and outputs
        """
        self.run_count += 1
        run_dir = self.output_dir / f"run_{self.run_count:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save code to file
        code_file = run_dir / "script.py"
        code_file.write_text(self._prepare_code(code, render, resolution))

        # Prepare blend file
        if blend_file is None:
            blend_file = run_dir / "input.blend"
            self._create_empty_blend(blend_file)
        else:
            blend_file = Path(blend_file)
            if not blend_file.exists():
                raise FileNotFoundError(f"Blend file not found: {blend_file}")

        # Output paths
        render_dir = run_dir / "renders"
        render_dir.mkdir(exist_ok=True)
        output_blend = run_dir / "output.blend" if save_blend else None

        # Execute Blender
        success, images, stdout, stderr = self._execute_blender(
            blend_file=blend_file,
            script_file=code_file,
            render_dir=render_dir,
            output_blend=output_blend,
        )

        if not success:
            return BlenderExecutionResult(
                success=False,
                rendered_images=[],
                stdout=stdout,
                stderr=stderr,
                error_message=f"Blender execution failed:\n{stderr}",
            )

        return BlenderExecutionResult(
            success=True,
            rendered_images=images,
            stdout=stdout,
            stderr=stderr,
            blend_file=output_blend if save_blend else None,
        )

    def _prepare_code(
        self, user_code: str, render: bool, resolution: Tuple[int, int]
    ) -> str:
        """Wrap user code with rendering logic"""
        # Strip markdown code fences if present
        if user_code.startswith("```python") and user_code.endswith("```"):
            user_code = user_code[len("```python") : -len("```")].strip()
        elif user_code.startswith("```") and user_code.endswith("```"):
            user_code = user_code[3:-3].strip()

        wrapper = f"""
import bpy
import sys
from pathlib import Path

# User code
{user_code}

# Render if requested
{
            ""
            if not render
            else f'''
# Setup render settings
scene = bpy.context.scene
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}
scene.render.resolution_percentage = 100

# Render from all cameras
render_dir = Path(bpy.context.scene.render.filepath).parent
for i, obj in enumerate(bpy.data.objects):
    if obj.type == 'CAMERA':
        scene.camera = obj
        output_path = str(render_dir / f"camera_{{i}}.png")
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered from camera {{i}}: {{output_path}}")

# If no cameras, use default view
if scene.camera is None:
    print("WARNING: No camera found in scene")
'''
        }

"""
        return wrapper

    def _create_empty_blend(self, path: Path):
        """Create an empty .blend file"""
        cmd = [
            self.blender_command,
            "--background",
            "--python-expr",
            f"import bpy; bpy.ops.wm.save_as_mainfile(filepath='{path}')",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except Exception as e:
            raise RuntimeError(f"Failed to create empty blend file: {e}")

    def _execute_blender(
        self,
        blend_file: Path,
        script_file: Path,
        render_dir: Path,
        output_blend: Optional[Path],
    ) -> Tuple[bool, List[Path], str, str]:
        """Execute Blender in background mode"""
        cmd = [
            self.blender_command,
            "--background",
            str(blend_file),
            "--python",
            str(script_file),
        ]

        if output_blend:
            cmd.extend(
                [
                    "--render-output",
                    str(render_dir / "render_"),
                    "--",
                    "--save-blend",
                    str(output_blend),
                ]
            )

        # Set environment
        env = os.environ.copy()
        if self.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
        env["AL_LIB_LOGLEVEL"] = "0"  # Suppress audio errors

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,  # 5 minutes max
            )

            # Find rendered images
            images = sorted([p for p in render_dir.glob("*.png")])

            # Save .blend if requested
            if output_blend and proc.returncode == 0:
                save_cmd = [
                    self.blender_command,
                    "--background",
                    str(blend_file),
                    "--python",
                    str(script_file),
                    "--python-expr",
                    f"import bpy; bpy.ops.wm.save_as_mainfile(filepath='{output_blend}')",
                ]
                subprocess.run(save_cmd, capture_output=True, timeout=60)

            return (proc.returncode == 0, images, proc.stdout, proc.stderr)

        except subprocess.TimeoutExpired:
            return (False, [], "", "Blender execution timed out")
        except Exception as e:
            return (False, [], "", str(e))

    def extract_scene_info(self, blend_file: Path) -> Dict:
        """Extract scene metadata (objects, materials, cameras)"""
        extract_script = """
import bpy
import json

scene_info = {
    "objects": [],
    "cameras": [],
    "lights": [],
    "materials": [],
}

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        scene_info["objects"].append({
            "name": obj.name,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale),
        })
    elif obj.type == 'CAMERA':
        scene_info["cameras"].append({
            "name": obj.name,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
        })
    elif obj.type == 'LIGHT':
        scene_info["lights"].append({
            "name": obj.name,
            "type": obj.data.type,
            "energy": obj.data.energy,
        })

for mat in bpy.data.materials:
    scene_info["materials"].append(mat.name)

print(json.dumps(scene_info))
"""

        result = self.execute(
            code=extract_script,
            blend_file=blend_file,
            render=False,
        )

        if result.success:
            try:
                # Parse JSON from stdout
                for line in result.stdout.split("\n"):
                    if line.strip().startswith("{"):
                        return json.loads(line)
            except Exception:
                pass

        return {}


if __name__ == "__main__":
    # Test the executor
    print("🧪 Testing Blender executor...")

    executor = BlenderExecutor(output_dir="output/test_blender")

    test_code = """
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Add cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Add camera
bpy.ops.object.camera_add(location=(5, -5, 5))
cam = bpy.context.object
cam.rotation_euler = (1.1, 0, 0.8)
bpy.context.scene.camera = cam

# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
"""

    result = executor.execute(test_code)

    if result.success:
        print(f"Rendered {len(result.rendered_images)} images")
        print(f"Blend file: {result.blend_file}")
    else:
        print(f"Failed: {result.error_message}")
