#!/usr/bin/env python3
"""Call a running REST server (text embeddings).

Terminal 1:

    pip install "siglip2-toolkit[server]"
    siglip2 serve --host 127.0.0.1 --port 8000

Terminal 2:

    uv run python examples/06_rest_client.py --text "a cat on a sofa"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--text", action="append", help="Text to embed; repeat for several")
    args = parser.parse_args()

    texts = args.text or ["a cat on a sofa", "a rocket on the launch pad"]
    payload = json.dumps({"texts": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/embed/text",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"error: HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:
        print(f"error: cannot reach {args.url}: {exc.reason}", file=sys.stderr)
        print("Start the server with `siglip2 serve --host 127.0.0.1 --port 8000`.", file=sys.stderr)
        return 2

    print(f"model: {data['model']}")
    print(f"dimensions: {data['dimensions']}")
    print(f"vectors: {len(data['vectors'])} × {len(data['vectors'][0]) if data['vectors'] else 0}")
    for text, vector in zip(texts, data["vectors"]):
        print(f"  {text!r}: first values {vector[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
