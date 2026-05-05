#!/usr/bin/env python3
"""
SmartAttend — Command-line Model Trainer
Usage: python scripts/train_model.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_engine import FaceEngine
from core.trainer     import ModelTrainer

def progress(cur, total, msg):
    bar_len = 30
    filled  = int(bar_len * cur / total)
    bar     = '█' * filled + '░' * (bar_len - filled)
    print(f"\r[{bar}] {cur}/{total}  {msg[:50]:<50}", end='', flush=True)

if __name__ == "__main__":
    print("SmartAttend — Face Recognition Model Trainer")
    print("=" * 50)
    engine  = FaceEngine()
    trainer = ModelTrainer(engine)
    result  = trainer.train(progress_callback=progress)
    print()  # newline after progress bar
    print("=" * 50)
    if result["success"]:
        stats = result["stats"]
        print(f"✓  {result['message']}")
        print(f"   Images processed : {stats['encoded']}/{stats['total_images']}")
        print(f"   Failed            : {stats['failed']}")
        print(f"   Quality skips     : {stats.get('skipped_quality', 0)}")
        print(f"   Duplicate skips   : {stats.get('skipped_duplicate', 0)}")
        print(f"   Cap skips         : {stats.get('skipped_cap', 0)}")
        print(f"   Time              : {stats['elapsed_seconds']}s")
    else:
        print(f"✗  {result['message']}")
        sys.exit(1)
