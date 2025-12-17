"""
GTI Batch Processor v1.0
Process all raw cases in a folder automatically
"""

import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from gti_pipeline import run_pipeline


def process_batch(input_dir: Path, output_summary: bool = True) -> List[Dict]:
    """
    Process all .txt files in input_dir through the GTI pipeline.

    Returns list of results.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        return []

    # Find all .txt files
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return []

    print(f"\n{'='*60}")
    print(f"GTI Batch Processor")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Files found: {len(txt_files)}")
    print(f"{'='*60}\n")

    results = []
    success = 0
    failed = 0

    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] Processing: {txt_file.name}")
        print("-" * 40)

        try:
            case_id = f"GTI-{txt_file.stem.upper().replace(' ', '-')}"

            result = run_pipeline(str(txt_file), case_id=case_id, save=True)

            if "error" not in result:
                results.append({
                    "file": txt_file.name,
                    "case_id": result["case_id"],
                    "title": result.get("title", ""),
                    "game_type": result["game_type"],
                    "confidence": result["confidence"],
                    "reasoning": result["reasoning"],
                    "status": "success"
                })
                success += 1
            else:
                results.append({
                    "file": txt_file.name,
                    "case_id": case_id,
                    "title": "",
                    "game_type": "ERROR",
                    "confidence": 0,
                    "reasoning": result["error"],
                    "status": "failed"
                })
                failed += 1

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "file": txt_file.name,
                "case_id": txt_file.stem,
                "title": "",
                "game_type": "ERROR",
                "confidence": 0,
                "reasoning": str(e),
                "status": "failed"
            })
            failed += 1

    # Generate summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total:   {len(txt_files)}")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"{'='*60}\n")

    if output_summary and results:
        save_summary(results, input_dir.parent)

    return results


def save_summary(results: List[Dict], base_dir: Path):
    """Save batch results to CSV and JSON summaries."""
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save CSV
    csv_path = output_dir / f"batch_summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "case_id", "title", "game_type", "confidence", "reasoning", "status"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV summary: {csv_path}")

    # Save JSON
    json_path = output_dir / f"batch_summary_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "total": len(results),
            "success": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results
        }, f, indent=2)
    print(f"JSON summary: {json_path}")

    # Print results table
    print(f"\n{'='*80}")
    print(f"{'File':<25} {'Game Type':<20} {'Conf':<6} {'Status':<8}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['file']:<25} {r['game_type']:<20} {r['confidence']:<6} {r['status']:<8}")
    print(f"{'='*80}\n")


def main():
    if len(sys.argv) < 2:
        # Default to data/raw
        base = Path(__file__).parent.parent
        input_dir = base / "data" / "raw"
    else:
        input_dir = Path(sys.argv[1])

    process_batch(input_dir)


if __name__ == "__main__":
    main()
