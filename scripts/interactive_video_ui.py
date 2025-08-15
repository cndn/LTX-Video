import streamlit as st
import requests
import tempfile
import os
import cv2
import numpy as np
from fal_playground import InteractiveVideoGenerator, extract_last_frame, gen_interactive_video
import asyncio
import time
import json
import fal_client

st.set_page_config(
    page_title="Interactive Video Generator", 
    page_icon="🎬",
    layout="wide"
)

def initialize_session_state():
    if 'generator' not in st.session_state:
        st.session_state.generator = None  # Start with no generator
    if 'generating' not in st.session_state:
        st.session_state.generating = False
    if 'generation_progress' not in st.session_state:
        st.session_state.generation_progress = ""

def display_video_player(video_url, caption=""):
    """Display video player with controls"""
    if video_url:
        st.video(video_url)
        if caption:
            st.caption(caption)

def display_current_frame():
    """Display current frame being used as input"""
    if st.session_state.generator and st.session_state.generator.current_frame_url:
        st.image(
            st.session_state.generator.current_frame_url, 
            caption="Last Frame (Input for next video)",
            width=300
        )
    else:
        st.info("👆 Please select an initial image to start generating videos")

@st.fragment(run_every=1)
def check_generation_status():
    """Check and display generation status"""
    if st.session_state.generating:
        st.info("🔄 Generating video... Please wait.")
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.1)
    
    if st.session_state.generation_progress:
        if "Error" in st.session_state.generation_progress:
            st.error(st.session_state.generation_progress)
        elif "successfully" in st.session_state.generation_progress:
            st.success(st.session_state.generation_progress)

def generate_video_sync(prompt):
    """Generate video synchronously"""
    try:
        st.session_state.generating = True
        st.session_state.generation_progress = "Starting generation..."
        
        # Store previous frame URL for debugging
        prev_frame = st.session_state.generator.current_frame_url
        
        with st.spinner("Generating video..."):
            video_url = st.session_state.generator.generate_next_video(prompt)
        
        # Check if frame was actually updated
        new_frame = st.session_state.generator.current_frame_url
        if new_frame != prev_frame:
            st.session_state.generation_progress = f"Video generated successfully! Frame updated from previous."
        else:
            st.session_state.generation_progress = "Video generated successfully! (Frame extraction may have failed - using previous frame)"
        
        st.session_state.generating = False
        return video_url
        
    except Exception as e:
        st.session_state.generation_progress = f"Error: {str(e)}"
        st.session_state.generating = False
        return None

def main():
    st.title("🎬 Interactive Video Generator")
    st.markdown("Generate sequential videos where each new video starts from the last frame of the previous one.")
    
    initialize_session_state()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Initial setup
        st.subheader("Initial Setup")
        initial_type = st.radio("Start from:", ["Default Image", "Custom Image", "Video URL"])
        
        if initial_type == "Default Image":
            if st.button("Use Default Astronaut Image"):
                st.session_state.generator = InteractiveVideoGenerator()
                st.success("Default image set!")
                st.rerun()
        
        # Model warmup
        st.subheader("🔥 Performance")
        if st.button("Warm Up Model"):
            with st.spinner("Warming up model..."):
                try:
                    # Quick warmup request
                    from fal_playground import gen_interactive_video
                    start = time.time()
                    gen_interactive_video(
                        "warmup", 
                        "https://storage.googleapis.com/falserverless/example_inputs/ltxv-image-input.jpg",
                        num_frames=5,
                        frame_rate=5
                    )
                    warmup_time = time.time() - start
                    st.success(f"Model warmed up in {warmup_time:.1f}s")
                except Exception as e:
                    st.error(f"Warmup failed: {e}")
        
        elif initial_type == "Custom Image":
            # Option 1: Image URL
            image_url = st.text_input("Image URL:")
            if st.button("Set Initial Image") and image_url:
                st.session_state.generator = InteractiveVideoGenerator(image_url)
                st.success("Initial image set!")
            
            st.write("**OR**")
            
            # Option 2: File Upload
            uploaded_file = st.file_uploader(
                "Upload your own image",
                type=['png', 'jpg', 'jpeg', 'webp'],
                help="Upload an image to use as the starting frame"
            )
            
            if uploaded_file is not None:
                if st.button("Upload & Set as Initial Frame"):
                    try:
                        with st.spinner("Uploading image..."):
                            # Read the uploaded file content
                            file_content = uploaded_file.read()
                            
                            # Upload to FAL
                            uploaded_url = fal_client.upload(file_content, uploaded_file.type)
                            
                            # Set as initial frame
                            st.session_state.generator = InteractiveVideoGenerator(uploaded_url)
                            
                            st.success("Custom image uploaded and set as initial frame!")
                            
                            # Show preview
                            st.image(uploaded_url, caption="Your uploaded image", width=200)
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
                
                # Show preview of uploaded file
                if uploaded_file:
                    st.image(uploaded_file, caption="Preview (click button above to use)", width=200)
                
        elif initial_type == "Video URL":
            video_url = st.text_input("Video URL:")
            if st.button("Extract Last Frame") and video_url:
                with st.spinner("Extracting last frame..."):
                    try:
                        initial_frame = extract_last_frame(video_url)
                        st.session_state.generator = InteractiveVideoGenerator(initial_frame)
                        st.success("Last frame extracted and set as initial frame!")
                    except Exception as e:
                        st.error(f"Error extracting frame: {e}")
        
        st.divider()
        
        # Generation controls
        st.subheader("📊 Statistics")
        videos_count = len(st.session_state.generator.video_history) if st.session_state.generator else 0
        st.metric("Videos Generated", videos_count)
        
        if st.button("🗑️ Clear History"):
            if st.session_state.generator:
                st.session_state.generator = InteractiveVideoGenerator(
                    st.session_state.generator.current_frame_url
                )
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎥 Video Generation")
        
        # Prompt input
        prompt = st.text_input(
            "Enter your prompt:",
            placeholder="e.g., 'A bird flies into the scene'",
            disabled=st.session_state.generating
        )
        
        # Generate button
        if st.button("🎬 Generate Video", disabled=st.session_state.generating or not prompt or not st.session_state.generator):
            if not st.session_state.generator:
                st.error("Please select an initial image first!")
            else:
                video_url = generate_video_sync(prompt)
                if video_url:
                    st.rerun()
        
        # Progress and status display
        if st.session_state.generation_progress:
            if "Error" in st.session_state.generation_progress:
                st.error(st.session_state.generation_progress)
            elif "successfully" in st.session_state.generation_progress:
                st.success(st.session_state.generation_progress)
        
        # Display latest video
        if st.session_state.generator and st.session_state.generator.video_history:
            latest_video = st.session_state.generator.video_history[-1]
            st.subheader("🎦 Latest Generated Video")
            
            # Auto-play video using st.video with autoplay
            st.video(
                latest_video['video_url'],
                autoplay=True,
            )
            st.caption(f"Prompt: '{latest_video['prompt']}'")
    
    with col2:
        st.header("🖼️ Last Frame")
        display_current_frame()
        
        # Debug info and profiling
        with st.expander("🔧 Debug & Performance"):
            if st.session_state.generator:
                st.text(f"Current Frame URL: {st.session_state.generator.current_frame_url}")
                st.text(f"Videos Generated: {len(st.session_state.generator.video_history)}")
                
                if st.session_state.generator.video_history:
                    latest = st.session_state.generator.video_history[-1]
                    st.text(f"Latest Video: {latest['video_url']}")
                    
                    # Performance metrics
                    if len(st.session_state.generator.video_history) >= 2:
                        prev_timestamp = st.session_state.generator.video_history[-2]['timestamp']
                        current_timestamp = latest['timestamp']
                        interval = current_timestamp - prev_timestamp
                        st.metric("Time Between Videos", f"{interval:.1f}s")
                    
                    # Average generation time
                    if len(st.session_state.generator.video_history) > 1:
                        timestamps = [v['timestamp'] for v in st.session_state.generator.video_history]
                        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                        avg_time = sum(intervals) / len(intervals)
                        st.metric("Average Generation Time", f"{avg_time:.1f}s")
                    
                    if st.button("🔄 Force Extract Last Frame", key="force_extract"):
                        try:
                            with st.spinner("Extracting frame..."):
                                new_frame = extract_last_frame(latest['video_url'])
                                st.session_state.generator.current_frame_url = new_frame
                                st.success("Frame extracted successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Frame extraction failed: {e}")
            else:
                st.info("No generator initialized yet")
        
        st.header("📝 Quick Prompts")
        quick_prompts = [
            "the astronaut walks away",
            "A pikachu appears in the scene running to the astronaut",
            "The scene transitions to night",
        ]
        
        for i, quick_prompt in enumerate(quick_prompts):
            if st.button(
                quick_prompt, 
                key=f"quick_{i}",
                disabled=st.session_state.generating or not st.session_state.generator,
                use_container_width=True
            ):
                if not st.session_state.generator:
                    st.error("Please select an initial image first!")
                else:
                    video_url = generate_video_sync(quick_prompt)
                    if video_url:
                        st.rerun()
    
    # Video history section
    if st.session_state.generator and st.session_state.generator.video_history:
        st.header("📚 Video History")
        
        # Display videos in reverse order (newest first)
        for i, video in enumerate(reversed(st.session_state.generator.video_history)):
            video_num = len(st.session_state.generator.video_history) - i
            
            with st.expander(f"Video {video_num}: {video['prompt']}", expanded=(i == 0)):
                col_video, col_info = st.columns([3, 1])
                
                with col_video:
                    display_video_player(video['video_url'])
                
                with col_info:
                    st.write("**Prompt:**")
                    st.write(video['prompt'])
                    
                    st.write("**Input Frame:**")
                    st.image(video['input_frame'], width=150)
                    
                    if st.button(f"🔗 Copy URL", key=f"copy_{video_num}"):
                        st.code(video['video_url'])

if __name__ == "__main__":
    main()