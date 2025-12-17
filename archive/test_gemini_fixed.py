"""
test_gemini_fixed.py - Test gemini-2.0-flash (stable)
"""

import os
import httpx
import json

key = os.getenv("GOOGLE_API_KEY", "")
if not key:
    print("ERROR: GOOGLE_API_KEY not set")
    exit(1)

# Test gemini-2.0-flash (stable, FREE)
model = "gemini-2.0-flash"
print(f"Testing {model}...")

url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

with httpx.Client(timeout=30) as client:
    response = client.post(
        url,
        params={"key": key},
        json={
            "contents": [{"parts": [{"text": "Reply with exactly one word: OK"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 10}
        }
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Full response:\n{json.dumps(data, indent=2)[:500]}")
        
        # Try different response formats
        try:
            if "candidates" in data:
                candidate = data["candidates"][0]
                if "content" in candidate:
                    content = candidate["content"]
                    if "parts" in content:
                        text = content["parts"][0].get("text", "")
                    else:
                        text = str(content)
                else:
                    text = str(candidate)
            else:
                text = str(data)
            
            print(f"\n✓ GEMINI WORKS!")
            print(f"  Model: {model}")
            print(f"  Response: {text.strip()}")
        except Exception as e:
            print(f"Parse error: {e}")
            print(f"Raw: {data}")
    else:
        print(f"Error: {response.text[:300]}")
