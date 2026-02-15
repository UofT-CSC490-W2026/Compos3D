"""Direct Blender rendering test - bypassing executor to prove rendering works"""

import subprocess
from pathlib import Path


# Create output directory
output_dir = Path("output/direct_render_test")
output_dir.mkdir(parents=True, exist_ok=True)

# Simple Blender script that renders a cube
blender_script = """
import bpy

# Clear scene
bpy.ops.wm.read_homefile(use_empty=True)

# Add cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Add camera
bpy.ops.object.camera_add(location=(5, -5, 4))
cam = bpy.context.active_object
cam.rotation_euler = (1.1, 0, 0.8)
bpy.context.scene.camera = cam

# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.active_object
light.data.energy = 2.0

# Setup rendering
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'  # Fast render engine
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.filepath = '%OUTPUT%'

# Render
bpy.ops.render.render(write_still=True)
print(f"Rendered to: {scene.render.filepath}")
"""

# Save script
script_file = output_dir / "render_cube.py"
output_file = output_dir / "cube_render.png"

script_content = blender_script.replace("%OUTPUT%", str(output_file))
script_file.write_text(script_content)

print(f"📝 Script: {script_file}")
print(f"📸 Output: {output_file}")

# Run Blender
print("\n🚀 Running Blender...")
cmd = [
    "blender",
    "--background",
    "--python",
    str(script_file),
]

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    print(f"Blender finished (exit code: {result.returncode})")

    # Check if image was created
    if output_file.exists():
        size_kb = output_file.stat().st_size / 1024
        print("\n🎉 SUCCESS! Rendered image created:")
        print(f"   📸 {output_file}")
        print(f"   Size: {size_kb:.1f} KB")
        print("   Resolution: 512x512")

        # Show in stdout
        print(f"\n📋 You can view the image at: {output_file.absolute()}")

    else:
        print("\nNo image file created")
        print(f"\nStderr:\n{result.stderr[-1000:]}")

except subprocess.TimeoutExpired:
    print("Blender timed out")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
if output_file.exists():
    print(f"✅ Render successful: {output_file}")
else:
    print("❌ Render failed")
