#!/usr/bin/env python3
"""Probe a llama.cpp server to inspect /props, /v1/models, and related endpoints."""

import argparse
import json
import urllib.request
import urllib.error


def fetch_json(url: str) -> dict | list | None:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"  ERROR: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  ERROR: invalid JSON: {e}")
        return None


def post_json(url: str, body: dict) -> dict | None:
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"  ERROR: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  ERROR: invalid JSON: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Probe a llama.cpp server")
    parser.add_argument(
        "--port", type=int, default=1337, help="Server port (default: 1337)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"

    # 1. Basic endpoints
    for endpoint in ["/props", "/health"]:
        url = f"{base}{endpoint}"
        print(f"\n--- GET {url} ---")
        data = fetch_json(url)
        if data is not None:
            print(json.dumps(data, indent=2))

    # 2. /v1/models — this is where Go code reads n_embd from meta
    print(f"\n--- GET {base}/v1/models ---")
    models_data = fetch_json(f"{base}/v1/models")
    model_ids = []
    if models_data is not None:
        print(json.dumps(models_data, indent=2))
        for m in models_data.get("data", []):
            model_ids.append(m.get("id", ""))
            meta = m.get("meta")
            if meta:
                print(f"  -> {m['id']}: n_embd={meta.get('n_embd', 'MISSING')}")
            else:
                print(f"  -> {m['id']}: meta field is ABSENT")

    # 3. /props?model=<name> for one discovered model (router mode)
    url = f"{base}/props?model=qwen3.5-9b"
    print(f"\n--- GET {url} ---")
    data = fetch_json(url)
    if data is not None:
        print(json.dumps(data, indent=2))
    # 4. Try a dummy embedding with one model
    print(f"\n--- POST {base}/v1/embeddings (model=qwen-embeddings) ---")
    result = post_json(
        f"{base}/v1/embeddings", {"input": "hello", "model": "qwen-embeddings"}
    )
    if result is not None:
        if "data" in result and len(result["data"]) > 0:
            emb = result["data"][0].get("embedding", [])
            print(f"  embedding dimension: {len(emb)}")
            print(f"  first 5 values: {emb[:5]}")
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
