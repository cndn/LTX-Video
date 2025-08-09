import fal_client as fal
import asyncio
import argparse
import sys


async def extend_video(video_url, prompt, extension_seconds=2, fps=16, resolution="720p", expand_directions=None):
    """
    Extend a video by X seconds using fal.ai's Wan VACE 14B outpainting API.
    
    Args:
        video_url (str): URL to the source video file
        prompt (str): Text description guiding video generation
        extension_seconds (float): Duration to extend the video by (default: 2 seconds)
        fps (int): Frames per second (5-30, default: 16)
        resolution (str): Output resolution - "480p", "580p", or "720p" (default: "720p")
        expand_directions (dict): Dictionary specifying expansion directions
                                 e.g., {"left": True, "right": True, "top": False, "bottom": False}
    
    Returns:
        dict: API response containing the extended video URL
    """
    if expand_directions is None:
        expand_directions = {"left": True, "right": True, "top": False, "bottom": False}
    
    # Calculate number of frames based on extension duration
    additional_frames = int(extension_seconds * fps)
    # Base frames (minimum) plus additional frames for extension
    num_frames = 81 + additional_frames
    # Ensure we don't exceed the maximum of 241 frames
    num_frames = min(num_frames, 241)
    
    try:
        arguments = {
            "prompt": prompt,
            "video_url": video_url,
            "expand_left": expand_directions.get("left", False),
            "expand_right": expand_directions.get("right", False), 
            "expand_top": expand_directions.get("top", False),
            "expand_bottom": expand_directions.get("bottom", False),
            "expand_ratio": 0.25,
            "num_frames": num_frames,
            "frames_per_second": fps,
            "resolution": resolution
        }
        result = await fal.subscribe("fal-ai/wan-vace-14b/outpainting", arguments)
        
        return result
    except Exception as e:
        print(f"Error extending video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extend videos using fal.ai API")
    parser.add_argument("--video-url", required=True, help="URL to the source video file")
    parser.add_argument("--prompt", required=True, help="Text description guiding video generation")
    parser.add_argument("--extension-seconds", type=float, default=2.0, 
                       help="Duration to extend the video by in seconds (default: 2.0)")
    parser.add_argument("--fps", type=int, default=16, choices=range(5, 31),
                       help="Frames per second (5-30, default: 16)")
    parser.add_argument("--resolution", default="720p", choices=["480p", "580p", "720p"],
                       help="Output resolution (default: 720p)")
    parser.add_argument("--expand-left", action="store_true", help="Expand video to the left")
    parser.add_argument("--expand-right", action="store_true", help="Expand video to the right")
    parser.add_argument("--expand-top", action="store_true", help="Expand video to the top")
    parser.add_argument("--expand-bottom", action="store_true", help="Expand video to the bottom")
    
    args = parser.parse_args()
    
    expand_directions = {
        "left": args.expand_left,
        "right": args.expand_right,
        "top": args.expand_top,
        "bottom": args.expand_bottom
    }
    
    # If no expansion directions specified, default to temporal extension
    if not any(expand_directions.values()):
        expand_directions["right"] = True
    
    async def run_extension():
        result = await extend_video(
            video_url=args.video_url,
            prompt=args.prompt,
            extension_seconds=args.extension_seconds,
            fps=args.fps,
            resolution=args.resolution,
            expand_directions=expand_directions
        )
        
        if result:
            print(f"Video extension successful!")
            print(f"Result: {result}")
            if 'video' in result:
                print(f"Extended video URL: {result['video']['url']}")
        else:
            print("Video extension failed.")
            sys.exit(1)
    
    asyncio.run(run_extension())


if __name__ == "__main__":
    main()
