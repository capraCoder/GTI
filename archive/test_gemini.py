"""
test_gemini_models.py - Find correct Gemini model name
"""

import os
import httpx

key = os.getenv("GOOGLE_API_KEY", "")
if not key:
    print("ERROR: GOOGLE_API_KEY not set")
    exit(1)

# List available models
print("Fetching available Gemini models...\n")

url = "https://generativelanguage.googleapis.com/v1beta/models"
with httpx.Client(timeout=30) as client:
    response = client.get(url, params={"key": key})
    
    if response.status_code == 200:
        data = response.json()
        models = data.get("models", [])
        
        print(f"Found {len(models)} models:\n")
        
        # Filter for flash/pro models that support generateContent
        flash_models = []
        for m in models:
            name = m.get("name", "")
            methods = m.get("supportedGenerationMethods", [])
            
            if "generateContent" in methods:
                short_name = name.replace("models/", "")
                if "flash" in short_name.lower() or "pro" in short_name.lower():
                    flash_models.append(short_name)
                    print(f"  ✓ {short_name}")
        
        print(f"\n--- Recommended models for GTI ---")
        for m in flash_models[:5]:
            print(f"  • {m}")
        
        # Test the first flash model
        if flash_models:
            test_model = flash_models[0]
            print(f"\n--- Testing {test_model} ---")
            
            test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{test_model}:generateContent"
            test_response = client.post(
                test_url,
                params={"key": key},
                json={
                    "contents": [{"parts": [{"text": "Say OK"}]}],
                    "generationConfig": {"temperature": 0, "maxOutputTokens": 10}
                }
            )
            
            if test_response.status_code == 200:
                result = test_response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"  ✓ Response: {text.strip()}")
                print(f"\n✓ USE THIS MODEL: {test_model}")
            else:
                print(f"  ✗ Error: {test_response.status_code}")
    else:
        print(f"Error listing models: {response.status_code}")
        print(response.text[:500])
