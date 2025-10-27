#!/usr/bin/env python3
"""
Quick ML training and testing commands for the enhanced email system.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n🔄 {description}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"✅ Success!")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ Error (exit code {result.returncode})")
            if result.stderr:
                print(result.stderr)
                
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    """Main command interface."""
    
    if len(sys.argv) < 2:
        print("🚀 Enhanced Email System - ML Training Commands")
        print("=" * 50)
        print()
        print("Available commands:")
        print("  train    - Train enhanced ML model with combined dataset")
        print("  test     - Test the enhanced hybrid system (requires Ollama)")
        print("  quick    - Quick training and testing (requires Ollama)")
        print("  api      - Start enhanced API server")
        print()
        print("Usage: python ml_commands.py <command>")
        print()
        print("Note: 'test' and 'quick' require Ollama to be running.")
        print("      Use 'train' for ML-only training without LLM.")
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        run_command("uv run python train_enhanced_model.py", "Training Enhanced ML Model")
        
    elif command == "test":
        run_command("uv run python test_enhanced_model.py", "Testing Enhanced Hybrid System")
        
    elif command == "quick":
        print("🚀 Quick Enhanced System Setup")
        print("=" * 40)
        
        run_command("uv run python train_enhanced_model.py", "Step 1: Training Enhanced ML Model")
        run_command("uv run python test_enhanced_model.py", "Step 2: Testing Hybrid System")
        
        print("\n🎉 Quick setup complete!")
        print("✅ Your enhanced system is ready!")
        
    elif command == "api":
        print("🌐 Starting Enhanced API Server...")
        print("   Visit: http://localhost:8000/docs")
        print("   Press Ctrl+C to stop")
        subprocess.run("uv run python -m src.mail_agents.api", shell=True)
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Available: train, test, quick, api")

if __name__ == "__main__":
    main()