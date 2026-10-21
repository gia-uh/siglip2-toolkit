#!/usr/bin/env python3
"""List the toolkit model cache and the SigLIP 2 catalog.

This example never downloads anything. Use the CLI for download/import/use:

    siglip2 models --hf
    siglip2 models download google/siglip2-base-patch16-naflex
    siglip2 models use google/siglip2-base-patch16-naflex
"""

from __future__ import annotations

import argparse

from siglip2_toolkit import models


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", help="Toolkit home (default: ~/.local/share/siglip2-toolkit)")
    parser.add_argument("--catalog", action="store_true", help="Also print the SigLIP 2 catalog")
    args = parser.parse_args()

    entries = models.list_models(home=args.home)
    current = models.current_identifier(home=args.home)
    print(f"toolkit home: {models.toolkit_home(args.home)}")
    print(f"current: {current or '—'}")
    if not entries:
        print("\nNo models in the cache.")
    else:
        print("\ncached models:")
        for entry in entries:
            marker = "*" if entry.identifier == current else " "
            print(f" {marker} {entry.identifier}  {human_bytes(entry.size_bytes)}  {entry.source}")
            print(f"     {entry.path}")

    if args.catalog:
        print("\nSigLIP 2 catalog (informational; nothing is downloaded):")
        for item in models.CATALOG:
            print(f"  {item.id:45} {item.params:>5}  patch {item.patch:>2}  {item.approx_size}")

    print("\nDownload/import with:")
    print("  siglip2 models download google/siglip2-base-patch16-naflex")
    print("  siglip2 models import /path/to/model --name my-model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
