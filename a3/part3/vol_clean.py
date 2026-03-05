import argparse
import subprocess
import sys

VOLUME = "nanochat-vol"


def modal_ls(path: str) -> list[str]:
    result = subprocess.run(
        ["modal", "volume", "ls", VOLUME, f"/{path}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def modal_rm(path: str) -> int:
    print(f"  Deleting /{path} ...")
    result = subprocess.run(
        ["modal", "volume", "rm", "-r", VOLUME, f"/{path}"], text=True
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
