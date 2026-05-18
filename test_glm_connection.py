#!/usr/bin/env python3
"""
快速验证 GLM API 连接是否正常。
使用 httpx 直接调用，不依赖 zhipuai SDK（该包在此环境 import 时会挂起）。
"""
import sys
import os
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LLM_factory.GLM import ZHIPU_API_KEY, ZHIPU_BASE_URL


def chat(model: str, content: str) -> str:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{ZHIPU_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {ZHIPU_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 16,
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def test_connection():
    print(f"[CONFIG] BASE_URL : {ZHIPU_BASE_URL}")
    print(f"[CONFIG] API_KEY  : {ZHIPU_API_KEY[:8]}...{ZHIPU_API_KEY[-4:]}")
    print()

    for model in ["glm-4.7"]:
        print(f"[TEST] {model} ...")
        try:
            reply = chat(model, "你好，请只回复 OK")
            print(f"[PASS] 回复: {repr(reply)}")
        except Exception as e:
            print(f"[FAIL] {model} 失败: {e}")
            return False
        print()

    print("[ALL PASS] API KEY 与 BASE URL 均正常，可以运行实验。")
    return True


if __name__ == "__main__":
    ok = test_connection()
    sys.exit(0 if ok else 1)
