#!/usr/bin/env python3
"""
Model warmup script to keep FAL AI models hot and reduce cold start times.
Run this periodically to maintain warm instances.
"""

import fal_client
import time
import dotenv
import threading
from concurrent.futures import ThreadPoolExecutor
dotenv.load_dotenv()

def warmup_model():
    """Send a lightweight request to keep the model warm."""
    try:
        print("🔥 Warming up LTX Video model...")
        start_time = time.time()
        
        # Use minimal parameters for fastest warmup
        result = fal_client.subscribe(
            "fal-ai/ltxv-13b-098-distilled/image-to-video",
            arguments={
                "prompt": "warmup",
                "image_url": "https://storage.googleapis.com/falserverless/example_inputs/ltxv-image-input.jpg",
                "num_frames": 5,  # Minimal frames for fastest processing
                "frame_rate": 5,
                "resolution": "480p",
                "enable_safety_checker": False
            },
            with_logs=False,  # Disable logs for cleaner output
        )
        
        warmup_time = time.time() - start_time
        print(f"✅ Model warmed up in {warmup_time:.1f}s")
        return warmup_time
        
    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        return None

def continuous_warmup(interval_minutes=15):
    """Keep the model warm with periodic requests."""
    print(f"🔄 Starting continuous warmup every {interval_minutes} minutes")
    
    while True:
        warmup_model()
        print(f"⏰ Next warmup in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)

def parallel_warmup_test():
    """Test parallel warmup requests to see if multiple instances help."""
    print("🧪 Testing parallel warmup requests...")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(warmup_model) for _ in range(3)]
        times = [f.result() for f in futures if f.result() is not None]
    
    if times:
        print(f"📊 Parallel warmup times: {times}")
        print(f"📈 Average: {sum(times)/len(times):.1f}s")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FAL AI Model Warmup Utility")
    parser.add_argument("--continuous", "-c", action="store_true", 
                       help="Run continuous warmup")
    parser.add_argument("--interval", "-i", type=int, default=15,
                       help="Warmup interval in minutes (default: 15)")
    parser.add_argument("--test-parallel", "-p", action="store_true",
                       help="Test parallel warmup requests")
    
    args = parser.parse_args()
    
    if args.test_parallel:
        parallel_warmup_test()
    elif args.continuous:
        continuous_warmup(args.interval)
    else:
        warmup_model()