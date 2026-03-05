"""
vol_clean.py — Delete one or more directories from the nanochat Modal volume.

Usage
-----
# Preview what would be deleted (dry-run, default):
    python part3/vol_clean.py nanochat_cache/base_checkpoints
    python part3/vol_clean.py nanochat_cache/base_checkpoints nanochat_cache/chatsft_checkpoints

# Actually delete:
    python part3/vol_clean.py --delete nanochat_cache/base_checkpoints
    python part3/vol_clean.py --delete nanochat_cache/base_checkpoints nanochat_cache/chatsft_checkpoints

# Common targets:
#   nanochat_cache/base_checkpoints       pretrain checkpoints
#   nanochat_cache/chatsft_checkpoints    SFT checkpoints
#   nanochat_cache/chatrl_checkpoints     RL checkpoints
#   nanochat_cache/report                 markdown report files
#   nanochat_cache/base_data_climbmix     downloaded data shards
#   nanochat_cache/eval_bundle            eval benchmark bundle
"""

import argparse
import subprocess
import sys

VOLUME = "nanochat-vol"


def modal_ls(path: str) -> list[str]:
    """Return a list of entries under `path` in the volume (empty if not found)."""
    result = subprocess.run(
        ["modal", "volume", "ls", VOLUME, f"/{path}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def modal_rm(path: str) -> int:
    """Recursively delete `path` from the volume. Returns the exit code."""
    print(f"  Deleting /{path} ...")
    result = subprocess.run(
        ["modal", "volume", "rm", "-r", VOLUME, f"/{path}"],
        text=True
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Clean up directories inside the nanochat Modal volume."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Volume-relative path(s) to delete, e.g. nanochat_cache/base_checkpoints",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Actually perform the deletion (without this flag it's a dry-run)",
    )
    args = parser.parse_args()

    if not args.delete:
        print("=== DRY RUN — pass --delete to actually remove files ===\n")

    errors = 0
    for path in args.paths:
        # Normalise: strip leading slash so we can safely prepend one for modal
        path = path.lstrip("/")

        print(f"[{'DELETE' if args.delete else 'DRY-RUN'}] /{path}")
        entries = modal_ls(path)
        if not entries:
            print(f"  (path not found or already empty, skipping)\n")
            continue

        print(f"  Contents ({len(entries)} entries):")
        for e in entries:
            print(f"    {e}")

        if args.delete:
            rc = modal_rm(path)
            if rc != 0:
                print(f"  ERROR: modal volume rm exited with code {rc}")
                errors += 1
            else:
                print(f"  Done.\n")
        else:
            print(f"  (skipped — dry run)\n")

    if errors:
        print(f"\n{errors} error(s) occurred.")
        sys.exit(1)
    elif args.delete:
        print("\nAll deletions complete.")
    else:
        print("\nDry run complete. Re-run with --delete to apply.")


if __name__ == "__main__":
    main()
