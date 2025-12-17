"""
test_apis.py - Quick API connectivity test for Gemini + DeepSeek

Usage:
  1. Set environment variables:
     $env:GOOGLE_API_KEY = "your-key"
     $env:DEEPSEEK_API_KEY = "your-key"
  
  2. Run:
     python test_apis.py
"""

import os
import httpx

def test_gemini():
    """Test Gemini API"""
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        return False, "GOOGLE_API_KEY not set"
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                url,
                params={"key": key},
                json={
                    "contents": [{"parts": [{"text": "Reply with exactly: GEMINI_OK"}]}],
                    "generationConfig": {"temperature": 0, "maxOutputTokens": 20}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return True, text.strip()
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
    
    except Exception as e:
        return False, str(e)


def test_deepseek():
    """Test DeepSeek API"""
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return False, "DEEPSEEK_API_KEY not set"
    
    url = "https://api.deepseek.com/v1/chat/completions"
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "Reply with exactly: DEEPSEEK_OK"}],
                    "temperature": 0,
                    "max_tokens": 20
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                return True, text.strip()
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
    
    except Exception as e:
        return False, str(e)


def test_claude():
    """Test Claude API (optional)"""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return None, "ANTHROPIC_API_KEY not set (optional)"
    
    url = "https://api.anthropic.com/v1/messages"
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                url,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 20,
                    "messages": [{"role": "user", "content": "Reply with exactly: CLAUDE_OK"}]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                text = data["content"][0]["text"]
                return True, text.strip()
            elif response.status_code == 429:
                return True, "Rate limited (but key is valid)"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
    
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("=" * 50)
    print("GTI API CONNECTIVITY TEST")
    print("=" * 50)
    
    # Gemini
    print("\n[1/3] Testing Gemini (gemini-1.5-flash)...")
    ok, msg = test_gemini()
    if ok:
        print(f"  ✓ Gemini: {msg}")
    else:
        print(f"  ✗ Gemini: {msg}")
    
    # DeepSeek
    print("\n[2/3] Testing DeepSeek (deepseek-chat)...")
    ok, msg = test_deepseek()
    if ok:
        print(f"  ✓ DeepSeek: {msg}")
    else:
        print(f"  ✗ DeepSeek: {msg}")
    
    # Claude (optional)
    print("\n[3/3] Testing Claude (optional)...")
    ok, msg = test_claude()
    if ok is None:
        print(f"  ⊘ Claude: {msg}")
    elif ok:
        print(f"  ✓ Claude: {msg}")
    else:
        print(f"  ✗ Claude: {msg}")
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)
