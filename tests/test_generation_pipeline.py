"""Unit tests for generation pipeline"""

import pytest
from pathlib import Path


@pytest.mark.generation
@pytest.mark.unit
def test_generator_initialization(test_env):
    """Test Generator can be initialized"""
    from compos3d_dp.inference.generator import Compos3DGenerator

    generator = Compos3DGenerator(env=test_env)

    assert generator.env == test_env
    assert generator.cfg is not None
    assert generator.store is not None
    assert generator.blender is not None
    assert generator.critic is not None


@pytest.mark.generation
@pytest.mark.unit
def test_generator_blender_code_generation(test_env):
    """Test Generator can generate Blender code (stub)"""
    from compos3d_dp.inference.generator import Compos3DGenerator

    generator = Compos3DGenerator(env=test_env)

    code = generator.generate_blender_code(prompt="a red cube")

    assert code is not None
    assert len(code) > 0
    assert "import bpy" in code


@pytest.mark.generation
@pytest.mark.integration
def test_generator_scene_evaluation(test_env):
    """Test Generator can evaluate scenes"""
    from compos3d_dp.inference.generator import Compos3DGenerator

    # First generate a simple render
    from compos3d_dp.generation.blender_executor import BlenderExecutor

    blender = BlenderExecutor()

    simple_code = """
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
bpy.ops.object.camera_add(location=(5, -5, 5))
camera = bpy.context.active_object
camera.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = camera
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
"""

    output_dir = "output/test_gen_eval"
    blender.output_dir = Path(output_dir)
    exec_result = blender.execute(code=simple_code, render=True)

    if exec_result.success and exec_result.rendered_images:
        generator = Compos3DGenerator(env=test_env)
        scores = generator.evaluate_scene(
            render_path=exec_result.rendered_images[0],
            prompt="a cube",
        )

        assert "quality" in scores
        assert "prompt_adherence" in scores
        assert all(0 <= v <= 1 for v in scores.values())


@pytest.mark.generation
@pytest.mark.unit
def test_generator_api(test_env):
    """Test Generator exposes correct API for future implementation"""
    from compos3d_dp.inference.generator import Compos3DGenerator

    generator = Compos3DGenerator(env=test_env)

    # Check methods exist
    assert hasattr(generator, "load_model")
    assert hasattr(generator, "generate_blender_code")
    assert hasattr(generator, "evaluate_scene")
    assert hasattr(generator, "generate")

    # Check methods are callable
    assert callable(generator.load_model)
    assert callable(generator.generate_blender_code)
    assert callable(generator.evaluate_scene)
    assert callable(generator.generate)


@pytest.mark.generation
@pytest.mark.integration
def test_generation_end_to_end(test_env):
    """Test full generation pipeline"""
    from compos3d_dp.inference.generator import generate_scene

    result = generate_scene(
        prompt="a simple cube",
        env=test_env,
        checkpoint_path=None,
    )

    assert "success" in result
    assert "generation_id" in result

    if result["success"]:
        assert "blend_file" in result
        assert "render" in result
        assert "s3_blend" in result
        assert "s3_render" in result
        assert "scores" in result
