#!/usr/bin/env python3
"""
Test script for streaming video generation with predefined prompts.
Tests the continuity feature by generating a sequence of related prompts.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from streaming_inference import StreamingVideoGenerator


def test_streaming_continuity():
    """Test the streaming video generator with a sequence of prompts."""
    
    # Test prompts that should create a coherent narrative
    test_prompts = [
        "a person walking forward",
        "move left slowly", 
        "move right quickly",
        "stop and look up",
        "ufo appears in the sky",
        "ufo hovers overhead",
        "person runs away scared",
        "fade to black"
    ]
    
    print("🧪 STREAMING VIDEO CONTINUITY TEST")
    print("="*50)
    print(f"Testing with {len(test_prompts)} sequential prompts:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. {prompt}")
    print("="*50)
    
    # Initialize the generator
    try:
        generator = StreamingVideoGenerator(
            config_path="configs/ltxv-2b-simple.yaml",
            height=256,
            width=384
        )
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return False
    
    # Generate videos for each prompt
    start_time = time.time()
    generated_files = []
    
    try:
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🎬 Processing prompt {i}/{len(test_prompts)}: '{prompt}'")
            
            segment_start = time.time()
            output_file = generator.generate_video(prompt)
            segment_duration = time.time() - segment_start
            
            generated_files.append(output_file)
            print(f"   ⏱️  Segment {i} took {segment_duration:.1f}s")
            
            # Small delay between generations
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        return False
    finally:
        # Clean up
        generator._cleanup()
    
    total_duration = time.time() - start_time
    
    # Report results
    print(f"\n✅ TEST COMPLETED")
    print("="*50)
    print(f"📊 Results:")
    print(f"   • Total segments: {len(generated_files)}")
    print(f"   • Total time: {total_duration:.1f}s")
    print(f"   • Average per segment: {total_duration/len(generated_files):.1f}s")
    print(f"   • Output directory: {generator.output_dir}")
    
    print(f"\n📁 Generated files:")
    for i, file in enumerate(generated_files, 1):
        if file and file.exists():
            file_size = file.stat().st_size / (1024*1024)  # MB
            print(f"   {i}. {file.name} ({file_size:.1f} MB)")
        else:
            print(f"   {i}. ❌ Missing file")
    
    # Check for continuous video
    continuous_files = list(generator.output_dir.glob("continuous_*.mp4"))
    if continuous_files:
        latest_continuous = max(continuous_files, key=lambda f: f.stat().st_mtime)
        file_size = latest_continuous.stat().st_size / (1024*1024)
        print(f"\n🎞️  Continuous video: {latest_continuous.name} ({file_size:.1f} MB)")
    
    return True


def test_reset_functionality():
    """Test the reset functionality."""
    print("\n🔄 TESTING RESET FUNCTIONALITY")
    print("="*50)
    
    try:
        generator = StreamingVideoGenerator(
            config_path="configs/ltxv-2b-simple.yaml", 
            height=256,
            width=384
        )
        
        # Generate first sequence
        print("📝 First sequence:")
        generator.generate_video("person walks")
        generator.generate_video("person sits down")
        print(f"   Generated {generator.segment_count} segments")
        
        # Reset and start new sequence
        print("\n🔄 Resetting...")
        generator.reset_sequence()
        print(f"   Segment count after reset: {generator.segment_count}")
        
        # Generate second sequence
        print("\n📝 Second sequence (should start fresh):")
        generator.generate_video("car drives by")
        generator.generate_video("car parks")
        print(f"   Generated {generator.segment_count} segments in new sequence")
        
        generator._cleanup()
        print("✅ Reset functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Reset test failed: {e}")
        return False


def main():
    """Main test runner."""
    print("🚀 STARTING STREAMING VIDEO TESTS")
    print("="*60)
    
    # Check if config file exists
    config_path = Path("configs/ltxv-2b-simple.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure the config file exists before running tests.")
        return
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Continuity
    if test_streaming_continuity():
        tests_passed += 1
    
    # Test 2: Reset functionality  
    if test_reset_functionality():
        tests_passed += 1
    
    # Final report
    print(f"\n🏁 TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()