"""
Quick Claude API Test
"""

import os
import httpx

key = os.getenv("ANTHROPIC_API_KEY", "")

if not key:
    print("ERROR: ANTHROPIC_API_KEY not set")
    print("\nSet it with:")
    print('  $env:ANTHROPIC_API_KEY = "sk-ant-..."')
    exit(1)

print("Testing Claude API...")

response = httpx.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    },
    json={
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Reply: OK"}]
    },
    timeout=30
)

if response.status_code == 200:
    text = response.json()["content"][0]["text"]
    print(f"✓ Claude works! Response: {text}")
else:
    print(f"✗ Error {response.status_code}: {response.text[:200]}")
