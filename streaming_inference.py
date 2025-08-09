#!/usr/bin/env python3
"""
Streaming video generation script.
Each text prompt generates a 1-second video (30 frames).
"""

import os
import sys
import yaml
import torch
import numpy as np
import imageio
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ltx_video.inference import create_ltx_video_pipeline, to_hwc3_uint8, get_unique_filename
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from huggingface_hub import hf_hub_download


class StreamingVideoGenerator:
    def __init__(self, config_path="configs/ltxv-2b-simple.yaml", height=256, width=384):
        self.config_path = config_path
        self.height = height
        self.width = width
        self.num_frames = 30  # 1 second at 30fps
        self.fps = 30
        self.pipeline = None
        self.output_dir = Path("streaming_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Store previous video state for continuation
        self.previous_frames = None
        self.all_generated_frames = []
        self.segment_count = 0
        
        # Set memory optimization for MPS
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        print("Initializing streaming video generator...")
        self._load_pipeline()
        
    def _load_pipeline(self):
        """Load the video generation pipeline once for reuse."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle checkpoint path - download from HF if not local
        ltxv_model_name_or_path = config.get("checkpoint_path", "ltxv-2b-0.9.8-distilled.safetensors")
        if not os.path.isfile(ltxv_model_name_or_path):
            print(f"Downloading checkpoint: {ltxv_model_name_or_path}")
            ltxv_model_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=ltxv_model_name_or_path,
                repo_type="model",
            )
        else:
            ltxv_model_path = ltxv_model_name_or_path
        
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=ltxv_model_path,
            precision=config.get("precision", "float16"),
            text_encoder_model_name_or_path=config.get("text_encoder_model_name_or_path"),
            sampler=config.get("sampler"),
            device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
            enhance_prompt=False,
            prompt_enhancer_image_caption_model_name_or_path=config.get("prompt_enhancer_image_caption_model_name_or_path"),
            prompt_enhancer_llm_model_name_or_path=config.get("prompt_enhancer_llm_model_name_or_path"),
        )
        
        self.config = config
        print("Pipeline loaded successfully!")
        
    def generate_video(self, prompt):
        """Generate a 1-second video from the given prompt, continuing from previous frame if available."""
        self.segment_count += 1
        print(f"\nGenerating video segment {self.segment_count} for: '{prompt}'")
        
        # Prepare pipeline config like the main inference script
        pipeline_config = self.config.copy()
        
        # Remove fields that shouldn't be passed to pipeline
        stg_mode = pipeline_config.get("stg_mode", "attention_values")
        if "stg_mode" in pipeline_config:
            del pipeline_config["stg_mode"]
        if "checkpoint_path" in pipeline_config:
            del pipeline_config["checkpoint_path"]
        if "text_encoder_model_name_or_path" in pipeline_config:
            del pipeline_config["text_encoder_model_name_or_path"]
        if "precision" in pipeline_config:
            del pipeline_config["precision"]
        if "sampler" in pipeline_config:
            del pipeline_config["sampler"]
        if "prompt_enhancement_words_threshold" in pipeline_config:
            del pipeline_config["prompt_enhancement_words_threshold"]
        if "prompt_enhancer_image_caption_model_name_or_path" in pipeline_config:
            del pipeline_config["prompt_enhancer_image_caption_model_name_or_path"]
        if "prompt_enhancer_llm_model_name_or_path" in pipeline_config:
            del pipeline_config["prompt_enhancer_llm_model_name_or_path"]
        if "stochastic_sampling" in pipeline_config:
            del pipeline_config["stochastic_sampling"]
        if "spatial_upscaler_model_path" in pipeline_config:
            del pipeline_config["spatial_upscaler_model_path"]
        if "downscale_factor" in pipeline_config:
            del pipeline_config["downscale_factor"]
        if "decode_timestep" in pipeline_config:
            del pipeline_config["decode_timestep"]
        if "decode_noise_scale" in pipeline_config:
            del pipeline_config["decode_noise_scale"]
        if "pipeline_type" in pipeline_config:
            del pipeline_config["pipeline_type"]
            
        # Set skip layer strategy
        if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
            skip_layer_strategy = SkipLayerStrategy.AttentionSkip
        elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
            skip_layer_strategy = SkipLayerStrategy.Residual
        elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
            skip_layer_strategy = SkipLayerStrategy.TransformerBlock
        else:
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        # Generate the video
        with torch.no_grad():
            # Use different seed for each segment to avoid identical generation
            generator = torch.Generator().manual_seed(42 + self.segment_count)
            
            # For continuation, we need to handle initial frames
            generate_kwargs = {
                **pipeline_config,
                "skip_layer_strategy": skip_layer_strategy,
                "prompt": prompt,
                "height": self.height,
                "width": self.width,
                "num_frames": self.num_frames,
                "frame_rate": self.fps,
                "generator": generator,
                "output_type": "pt",
                "is_video": True,
                "vae_per_channel_normalize": True,
            }
            
            # If we have previous frames, use them for initialization
            if self.previous_frames is not None:
                # Use the last few frames as initial frames for continuity
                generate_kwargs["initial_frames"] = self.previous_frames
            
            images = self.pipeline(**generate_kwargs).images
        
        # Convert and save video
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        
        # Store the last few frames for next iteration continuity
        # Keep the last 3-5 frames to provide context for the next generation
        num_context_frames = min(5, video_np.shape[0])
        self.previous_frames = torch.from_numpy(video_np[-num_context_frames:]).permute(0, 3, 1, 2).unsqueeze(0)
        
        # Add frames to our continuous sequence (skip first few frames if continuing to avoid duplication)
        if self.segment_count == 1:
            # First segment - keep all frames
            self.all_generated_frames.append(video_np)
        else:
            # Subsequent segments - skip first few frames to avoid duplication with previous end
            overlap_frames = min(3, video_np.shape[0] // 2)
            self.all_generated_frames.append(video_np[overlap_frames:])
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in prompt)[:50]
        output_filename = self.output_dir / f"segment_{self.segment_count:02d}_{timestamp}_{safe_prompt.replace(' ', '_')}.mp4"
        
        # Write video using robust method
        writer = imageio.get_writer(
            str(output_filename),
            fps=self.fps,
            codec="libx264",
            format="FFMPEG",
            ffmpeg_params=[
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-pix_fmt", "yuv420p",
            ],
        )
        
        with writer:
            for frame in video_np:
                writer.append_data(to_hwc3_uint8(frame))
        
        print(f"✓ Video segment saved: {output_filename}")
        
        # Also save the combined continuous video
        self._save_continuous_video()
        
        return output_filename
        
    def _save_continuous_video(self):
        """Save the continuous video combining all segments."""
        if not self.all_generated_frames:
            return
            
        # Concatenate all frames
        continuous_video = np.concatenate(self.all_generated_frames, axis=0)
        
        # Generate filename for continuous video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        continuous_filename = self.output_dir / f"continuous_{self.segment_count:02d}segments_{timestamp}.mp4"
        
        # Write continuous video
        writer = imageio.get_writer(
            str(continuous_filename),
            fps=self.fps,
            codec="libx264",
            format="FFMPEG",
            ffmpeg_params=[
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-pix_fmt", "yuv420p",
            ],
        )
        
        with writer:
            for frame in continuous_video:
                writer.append_data(to_hwc3_uint8(frame))
        
        print(f"✓ Continuous video saved: {continuous_filename} ({len(continuous_video)} frames)")
        return continuous_filename
        
    def reset_sequence(self):
        """Reset the video sequence to start fresh."""
        self.previous_frames = None
        self.all_generated_frames = []
        self.segment_count = 0
        print("🔄 Video sequence reset - next prompt will start fresh")
        
    def run_interactive(self):
        """Run interactive streaming mode."""
        print("\n" + "="*60)
        print("🎬 STREAMING VIDEO GENERATOR")
        print("="*60)
        print("• Each prompt generates a 1-second video (30 frames)")
        print("• Videos continue from the end of the previous prompt")
        print("• Type 'reset' to start a new sequence")
        print("• Type 'quit' or 'exit' to stop")
        print("• Press Ctrl+C to exit")
        print("="*60)
        
        try:
            while True:
                try:
                    prompt = input("\n📝 Enter prompt: ").strip()
                    
                    if not prompt:
                        continue
                        
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if prompt.lower() == 'reset':
                        self.reset_sequence()
                        continue
                    
                    start_time = datetime.now()
                    output_file = self.generate_video(prompt)
                    end_time = datetime.now()
                    
                    duration = (end_time - start_time).total_seconds()
                    print(f"⏱️  Generation time: {duration:.1f}s")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted by user. Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error generating video: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        
        self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.pipeline:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Clear MPS cache if available
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        print("🧹 Cleaned up resources")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming video generation")
    parser.add_argument("--config", default="configs/ltxv-2b-simple.yaml", help="Pipeline config file")
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=384, help="Video width")
    parser.add_argument("--prompt", help="Single prompt mode (non-interactive)")
    
    args = parser.parse_args()
    
    try:
        generator = StreamingVideoGenerator(
            config_path=args.config,
            height=args.height,
            width=args.width
        )
        
        if args.prompt:
            # Single prompt mode
            generator.generate_video(args.prompt)
        else:
            # Interactive streaming mode
            generator.run_interactive()
            
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()