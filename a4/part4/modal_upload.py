import modal
from pathlib import Path

# LOCAL_META = Path(r"C:\Users\minhc\Downloads\sft-20260313T045819Z-1-002\sft\meta_000966.json")
# LOCAL_PT = Path(r"C:\Users\minhc\Downloads\model_000966.pt")

# # destination inside the volume
# DEST_DIR = "nanochat_cache/chatsft_checkpoints/a4/d20_sft_orig"

# vol = modal.Volume.from_name("nanochat-vol")

# with vol.batch_upload(force=True) as batch:
#     batch.put_file(LOCAL_PT, f"{DEST_DIR}/{LOCAL_PT.name}")
#     batch.put_file(LOCAL_META, f"{DEST_DIR}/{LOCAL_META.name}")

# print("Upload complete.")

import modal

vol = modal.Volume.from_name("nanochat-vol")

old_files = [
    "nanochat_cache/chatsft_checkpoints/a4/d20_sft_orig/model_000966-001.pt",
]

for path in old_files:
    try:
        vol.remove_file(path)
        print(f"Removed: {path}")
    except Exception as e:
        print(f"Could not remove {path}: {e}")

vol.commit()
print("Done.")