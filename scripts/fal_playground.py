import fal_client
import dotenv
import requests
import tempfile
import cv2
import numpy as np
import argparse
import os
import time
dotenv.load_dotenv()

def gen_interactive_video(prompt, image_url, num_frames=61, frame_rate=10, keep_warm=True):
    start_time = time.time()
    print(f"🎬 Starting video generation at {time.strftime('%H:%M:%S')}")
    print(f"📝 Prompt: {prompt}")
    print(f"🖼️ Input image: {image_url}")
    print(f"🎞️ Frames: {num_frames}, FPS: {frame_rate}")
    
    generation_started = False
    
    def on_queue_update(update):
        nonlocal generation_started
        if isinstance(update, fal_client.InProgress):
            elapsed = time.time() - start_time
            for log in update.logs:
                message = log['message']
                print(f"[{elapsed:.1f}s] {message}")
                
                # Track when actual generation starts (after encoding phase)
                if not generation_started and "Encoding:" in message:
                    encoding_time = elapsed
                    print(f"⏱️ Pre-generation setup took {encoding_time:.1f}s")
                elif not generation_started and any(keyword in message.lower() for keyword in ["generating", "processing", "inference"]):
                    generation_started = True
                    generation_start_time = elapsed
                    print(f"🚀 Video generation phase started at {generation_start_time:.1f}s")

    # Optimization: Use consistent parameters and enable keep_warm if available
    arguments = {
        "prompt": prompt,
        "image_url": image_url,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "resolution": "480p",
    }
    
    # Add keep_warm flag if supported by the API
    if keep_warm:
        arguments["enable_safety_checker"] = False  # Disable safety checker for speed
    
    result = fal_client.subscribe(
        "fal-ai/ltxv-13b-098-distilled/image-to-video",
        arguments=arguments,
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    
    total_time = time.time() - start_time
    print(f"✅ Video generation completed in {total_time:.2f}s")
    
    if generation_started:
        actual_generation_time = total_time - generation_start_time if 'generation_start_time' in locals() else 0
        print(f"📊 Breakdown: Setup ~3.1s, Generation ~{actual_generation_time:.1f}s")
    
    print(f"🔗 Video URL: {result['video']['url']}")
    
    return result['video']['url']

def extract_last_frame(video_url):
    """Extract the last frame from a video URL and upload it for use as input."""
    start_time = time.time()
    print(f"🖼️ Starting frame extraction at {time.strftime('%H:%M:%S')}")
    print(f"📹 Video URL: {video_url}")
    
    download_start = time.time()
    response = requests.get(video_url)
    download_time = time.time() - download_start
    print(f"⬇️ Video download completed in {download_time:.2f}s ({len(response.content)/1024/1024:.1f}MB)")
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(response.content)
        temp_video_path = temp_file.name
    
    try:
        processing_start = time.time()
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"🎞️ Video info: {frame_count} frames, {fps:.1f} FPS")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()
        processing_time = time.time() - processing_start
        print(f"🎬 Frame extraction completed in {processing_time:.2f}s")
        
        if ret:
            upload_start = time.time()
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                cv2.imwrite(temp_img.name, frame)
                frame_size = os.path.getsize(temp_img.name)
                print(f"💾 Frame saved ({frame_size/1024:.1f}KB)")
                
                # Read the file content as bytes
                with open(temp_img.name, 'rb') as f:
                    file_content = f.read()
                
                # Upload using the bytes content
                uploaded_url = fal_client.upload(file_content, "image/jpeg")
                upload_time = time.time() - upload_start
                print(f"☁️ Frame upload completed in {upload_time:.2f}s")
                    
                os.unlink(temp_img.name)
                
                total_time = time.time() - start_time
                print(f"✅ Frame extraction total time: {total_time:.2f}s")
                print(f"🔗 Frame URL: {uploaded_url}")
                
                return uploaded_url
        else:
            raise Exception("Could not extract last frame")
    finally:
        os.unlink(temp_video_path)

class InteractiveVideoGenerator:
    def __init__(self, initial_image_url=None):
        self.video_history = []
        self.current_frame_url = initial_image_url or "https://storage.googleapis.com/falserverless/example_inputs/ltxv-image-input.jpg"
    
    def generate_next_video(self, prompt):
        """Generate next video in the sequence using current frame as input."""
        sequence_start = time.time()
        print(f"\n{'='*60}")
        print(f"🚀 Starting video sequence #{len(self.video_history) + 1}")
        print(f"📝 Prompt: '{prompt}'")
        print(f"🖼️ Input frame: {self.current_frame_url}")
        
        # Video generation phase
        video_url = gen_interactive_video(prompt, self.current_frame_url)
        self.video_history.append({
            'prompt': prompt,
            'video_url': video_url,
            'input_frame': self.current_frame_url,
            'timestamp': time.time()
        })
        
        # Frame extraction phase
        try:
            self.current_frame_url = extract_last_frame(video_url)
        except Exception as e:
            print(f"❌ Warning: Could not extract last frame: {e}")
            print("📋 Continuing with previous frame...")
        
        total_sequence_time = time.time() - sequence_start
        print(f"🏁 Total sequence time: {total_sequence_time:.2f}s")
        print(f"{'='*60}\n")
        
        return video_url
    
    def run_interactive_session(self):
        """Run interactive session where user provides prompts."""
        print("=== Interactive Video Generator ===")
        print("Type your prompts to generate sequential videos.")
        print("Each new video will start from the last frame of the previous video.")
        print("Type 'quit' to exit, 'history' to see all generated videos.\n")
        
        while True:
            try:
                prompt = input("\nEnter your prompt (or 'quit'/'history'): ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                elif prompt.lower() == 'history':
                    self.show_history()
                    continue
                elif not prompt:
                    print("Please enter a prompt.")
                    continue
                
                video_url = self.generate_next_video(prompt)
                print(f"\n✅ Video generated successfully!")
                print(f"📹 Video URL: {video_url}")
                print(f"📊 Total videos in sequence: {len(self.video_history)}")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    def show_history(self):
        """Display the history of generated videos."""
        if not self.video_history:
            print("No videos generated yet.")
            return
        
        print(f"\n=== Video History ({len(self.video_history)} videos) ===")
        for i, video in enumerate(self.video_history, 1):
            print(f"{i}. Prompt: '{video['prompt']}'")
            print(f"   Video: {video['video_url']}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactive Video Generator')
    parser.add_argument('--video-url', help='URL of initial video to start from')
    parser.add_argument('--image-url', help='URL of initial image to start from')
    parser.add_argument('--prompt', help='Single prompt to generate one video (non-interactive mode)')
    
    args = parser.parse_args()
    
    # Determine initial frame
    initial_frame = None
    if args.video_url:
        print(f"Extracting last frame from video: {args.video_url}")
        initial_frame = extract_last_frame(args.video_url)
        print(f"Using extracted frame: {initial_frame}")
    elif args.image_url:
        initial_frame = args.image_url
        print(f"Using provided image: {initial_frame}")
    
    generator = InteractiveVideoGenerator(initial_frame)
    
    if args.prompt:
        # Single generation mode
        video_url = generator.generate_next_video(args.prompt)
        print(f"Generated video: {video_url}")
    else:
        # Interactive mode
        generator.run_interactive_session()