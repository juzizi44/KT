#!/usr/bin/env python3
"""Quick connectivity check for the configured OpenAI endpoint."""

import json
import os
import sys
from openai import OpenAI


def main() -> int:
    # Hard-coded per user request; adjust if you want to test other keys or endpoints.
    os.environ["OPENAI_API_KEY"] = "***REMOVED***"
    # Bianxie exposes an OpenAI-compatible API that still expects the /v1 prefix.
    os.environ["OPENAI_BASE_URL"] = "https://api.bianxie.ai/v1"

    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a connectivity test."},
                {"role": "user", "content": "Reply with the word OK if you received this."},
            ],
            max_tokens=10,
            temperature=0,
        )
        print("Request succeeded. Raw response:")
        if hasattr(resp, "model_dump"):
            print(json.dumps(resp.model_dump(), indent=2))
        else:
            # Some proxies may already return plain text or dicts.
            print(f"(type={type(resp)}) {resp}")
        return 0
    except Exception as exc:  # pragma: no cover - debugging utility
        print(f"Request failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
