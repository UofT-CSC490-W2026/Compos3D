"""Test actual Blender rendering with real .png outputs"""

from compos3d_dp.generation.blender_executor import BlenderExecutor
from compos3d_dp.datasets.blenderbench import BlenderBenchDataset
from pathlib import Path
import shutil


# Load BlenderBench
dataset = BlenderBenchDataset(cache_dir="data/blenderbench")
dataset.download()

# Get a scene
instance = dataset.get_instance("level1/camera1")
blend_file = dataset.download_blend_file(instance)

print(f"\n📦 Test Scene: {instance.instance_id}")
print(f"   Task: {instance.task_description}")
print(f"   Blend file: {blend_file}")
print(f"   Exists: {blend_file.exists()}")

# Initialize Blender executor
output_dir = Path("output/blender_test_renders")
output_dir.mkdir(parents=True, exist_ok=True)

executor = BlenderExecutor(
    blender_command="blender",
    output_dir=str(output_dir),
)

print("\nBlender executor initialized")

# Test 1: Execute start code and render
print("\n" + "=" * 80)
print("TEST 1: Render with START code")

result = executor.execute(
    code=instance.start_code,
    blend_file=blend_file,
    render=True,
    resolution=(512, 512),
)

if result.success:
    print("Execution successful")
    print(f"   Rendered images: {len(result.rendered_images)}")
    for img in result.rendered_images:
        print(f"   📸 {img}")
        print(f"      Size: {img.stat().st_size / 1024:.1f} KB")
        print(f"      Exists: {img.exists()}")
else:
    print(f"Execution failed: {result.error_message}")
    print(f"   stderr: {result.stderr[-500:]}")

# Test 2: Execute goal code and render
print("\n" + "=" * 80)
print("TEST 2: Render with GOAL code")

result2 = executor.execute(
    code=instance.goal_code,
    blend_file=blend_file,
    render=True,
    resolution=(512, 512),
)

if result2.success:
    print("Execution successful")
    print(f"   Rendered images: {len(result2.rendered_images)}")
    for img in result2.rendered_images:
        print(f"   📸 {img}")
        print(f"      Size: {img.stat().st_size / 1024:.1f} KB")
else:
    print(f"❌ Execution failed: {result2.error_message}")

# Test 3: Simple custom scene
print("\n" + "=" * 80)
print("TEST 3: Custom scene from scratch")

custom_code = """
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Add cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1))
cube = bpy.context.active_object
cube.scale = (1, 1, 1)

# Add ground plane
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))

# Add camera
bpy.ops.object.camera_add(location=(7, -7, 5))
cam = bpy.context.active_object
cam.rotation_euler = (1.1, 0, 0.8)
bpy.context.scene.camera = cam

# Add sun light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.active_object
light.data.energy = 3.0

# Set up rendering
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32  # Low for speed
scene.render.resolution_x = 512
scene.render.resolution_y = 512
"""

result3 = executor.execute(
    code=custom_code,
    blend_file=None,  # Create from scratch
    render=True,
    resolution=(512, 512),
)

if result3.success:
    print("Execution successful")
    print(f"   Rendered images: {len(result3.rendered_images)}")
    for img in result3.rendered_images:
        print(f"   📸 {img}")
        print(f"      Size: {img.stat().st_size / 1024:.1f} KB")

        # Copy to easy-to-find location for viewing
        dest = Path("output/test_render_cube.png")
        if img.exists():
            shutil.copy(img, dest)
            print(f"   📋 Copied to: {dest}")
else:
    print(f"❌ Execution failed: {result3.error_message}")
    print(f"   stderr: {result3.stderr[-1000:]}")

# Summary
print("\n" + "=" * 80)

total_renders = 0
if result.success:
    total_renders += len(result.rendered_images)
if result2.success:
    total_renders += len(result2.rendered_images)
if result3.success:
    total_renders += len(result3.rendered_images)

print(f"Total renders produced: {total_renders}")
print(f"Output directory: {output_dir}")

# List all generated PNGs
all_pngs = list(output_dir.rglob("*.png"))
print(f"\n📸 All rendered images ({len(all_pngs)} total):")
for png in all_pngs[:10]:  # Show first 10
    print(f"   {png.relative_to(output_dir)}")
if len(all_pngs) > 10:
    print(f"   ... and {len(all_pngs) - 10} more")

print("\n" + "=" * 80)
if total_renders > 0:
    print(f"✅ All {total_renders} renders completed successfully")
else:
    print("❌ No renders completed")
