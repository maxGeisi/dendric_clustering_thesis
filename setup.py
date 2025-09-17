#!/usr/bin/env python3
"""
Setup script for Dendric Clustering Analysis
This script helps users set up their environment and verify the installation.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_conda():
    """Check if conda is available."""
    try:
        subprocess.run("conda --version", shell=True, check=True, capture_output=True)
        print("✅ Conda is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ Conda is not available. Please install Anaconda or Miniconda first.")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("DENDRIC CLUSTERING ANALYSIS - SETUP SCRIPT")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the refactored directory")
        print("   cd dendric_clustering/refactored")
        print("   python setup.py")
        sys.exit(1)
    
    # Check conda
    if not check_conda():
        sys.exit(1)
    
    print("\n📋 SETUP INSTRUCTIONS:")
    print("1. Create a conda environment:")
    print("   conda create -n dendric python=3.9")
    print("   conda activate dendric")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Verify installation:")
    print("   python -c \"import navis, pandas, numpy, plotly; print('Installation successful!')\"")
    
    print("\n📁 DIRECTORY STRUCTURE:")
    print("✅ Source code and configuration files are ready")
    print("📝 Place your data files in:")
    print("   - data/microns/extracted_swc/SWC_Skel/ (for SWC files)")
    print("   - data/microns/extracted_swc/syn/ (for synapse files)")
    print("   - data/iascone/extracted_swc/ (for IASCONE dataset)")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Set up your conda environment (see instructions above)")
    print("2. Add your data files to the appropriate directories")
    print("3. Edit config/default.yaml to set your target neuron")
    print("4. Run: jupyter notebook")
    print("5. Open: main/thesis_script_main.ipynb")
    
    print("\n" + "=" * 60)
    print("Setup instructions complete! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()
