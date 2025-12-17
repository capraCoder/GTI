"""
GTI Pipeline v1.0
End-to-end: Raw text → LLM structuring → Classification
"""

import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gti_classifier import classify_case, process_file

# Try to import anthropic, provide helpful error if missing
try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Run: pip install anthropic")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml package not installed")
    print("Run: pip install pyyaml")
    sys.exit(1)


def load_prompt() -> str:
    """Load the structurer system prompt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "structurer_prompt.md"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_yaml_from_response(response: str) -> str:
    """Extract YAML content from LLM response."""
    # Try to find YAML in code blocks first
    yaml_match = re.search(r'```ya?ml\s*\n(.*?)```', response, re.DOTALL)
    if yaml_match:
        return yaml_match.group(1).strip()

    # Try generic code block
    code_match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Return raw response if no code blocks
    return response.strip()


def structure_case(raw_text: str) -> dict:
    """Use Claude to structure raw case text into YAML format."""
    client = anthropic.Anthropic()

    system_prompt = load_prompt()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Structure this game theory scenario:\n\n{raw_text}"
            }
        ]
    )

    response_text = message.content[0].text
    yaml_content = extract_yaml_from_response(response_text)

    return yaml.safe_load(yaml_content)


def generate_case_id() -> str:
    """Generate a unique case ID."""
    now = datetime.now()
    return f"GTI-{now.year}-{now.strftime('%m%d%H%M%S')}"


def run_pipeline(input_source: str, case_id: str = None, save: bool = True) -> dict:
    """
    Run the full GTI pipeline.

    Args:
        input_source: Either a file path or raw text content
        case_id: Optional case ID (generated if not provided)
        save: Whether to save output files (default True)

    Returns:
        Dict with case_id, title, game_type, confidence, reasoning, and case data
    """
    base_dir = Path(__file__).parent.parent

    # Determine if input is a file path or raw text
    input_path = Path(input_source)
    if input_path.exists() and input_path.is_file():
        print(f"[1/3] Extracting structure via LLM...")
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        source_file = str(input_path)
    else:
        # Treat as raw text
        print(f"[1/3] Extracting structure via LLM...")
        raw_text = input_source
        source_file = None

    # Structure via LLM
    try:
        structured = structure_case(raw_text)
    except Exception as e:
        return {
            'case_id': case_id or 'ERROR',
            'error': str(e),
            'game_type': 'ERROR',
            'confidence': 0,
            'reasoning': f"LLM structuring failed: {e}"
        }

    # Ensure case_id exists
    if case_id:
        structured['case_id'] = case_id
    elif not structured.get('case_id') or structured['case_id'].startswith('GTI-YYYY'):
        structured['case_id'] = generate_case_id()

    title = structured.get('title', 'Unknown')
    print(f"      Title: {title}")

    # Save structured YAML
    if save:
        output_filename = f"{structured['case_id']}.yaml"
        output_path = base_dir / "data" / "structured" / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(structured, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"[2/3] Saved: {output_path}")
    else:
        print(f"[2/3] Structured (not saved)")

    # Classify
    print(f"[3/3] Classifying...")
    result = classify_case(structured)

    # Build result
    result_output = {
        'case_id': structured['case_id'],
        'title': title,
        'source_file': source_file,
        'game_type': result.game_type.value,
        'confidence': result.confidence,
        'reasoning': result.reasoning,
        'case': structured
    }

    # Save result
    if save:
        result_path = base_dir / "data" / "output" / f"{structured['case_id']}_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            # Don't include full case in saved JSON
            save_output = {k: v for k, v in result_output.items() if k != 'case'}
            json.dump(save_output, f, indent=2)

    return result_output


def main():
    if len(sys.argv) < 2:
        print("Usage: python gti_pipeline.py <raw_case.txt>")
        print("\nExample:")
        print("  python gti_pipeline.py data/raw/advertising_war.txt")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    result = run_pipeline(input_path)

    print()
    print(f"RESULT: {result['game_type']}")
    print(f"Confidence: {result['confidence']}%")


if __name__ == "__main__":
    main()
