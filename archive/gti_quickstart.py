"""
GTI Quick Start - games2p2k Validation

This script is pre-configured for your directory structure:
D:\CaprazliIndex\GTI\games2p2k_data\osfstorage\human data\games2p2k_main.csv

Usage:
    python gti_quickstart.py --stats       # View dataset statistics (no API)
    python gti_quickstart.py --validate 50 # Validate on 50 games
    python gti_quickstart.py --corpus      # Generate full labeled corpus
"""

import subprocess
import sys
from pathlib import Path

# Configure paths for your setup
DATA_PATH = Path("games2p2k_data/osfstorage/human data/games2p2k_main.csv")
VALIDATION_SCRIPT = "gti_validate_games2p2k.py"

def main():
    # Check if data file exists
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        print("\nExpected structure:")
        print("  GTI/")
        print("    games2p2k_data/")
        print("      osfstorage/")
        print("        human data/")
        print("          games2p2k_main.csv  <-- This file")
        return
    
    print(f"Found data: {DATA_PATH} ({DATA_PATH.stat().st_size / 1024:.1f} KB)")
    
    # Get command line args
    args = sys.argv[1:] if len(sys.argv) > 1 else ["--stats"]
    
    # Build command
    cmd = [
        sys.executable,
        VALIDATION_SCRIPT,
        "--data", str(DATA_PATH),
    ] + args
    
    print(f"Running: {' '.join(cmd)}\n")
    
    # Execute
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
