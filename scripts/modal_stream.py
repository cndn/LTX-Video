# Usage: 
# modal deploy scripts/modal_stream.py
# local test: 
# echo -e "\n=== Testing Simple Offer ==="
# curl -s -X POST https://YOUR_MODAL_URL/offer \
#   -H "Content-Type: application/json" \
#   -d '{
#     "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n",
#     "type": "offer",
#     "prompt": "cat sleeping"
#   }' | jq .
import modal
from pathlib import Path
import string
import time

app = modal.App("ltxv-webrtc-fixed")

VOLUME_NAME = "ltxv-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")
MODEL_VOLUME_NAME = "ltx-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models")
def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # some OSes limit filenames to <256 chars
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name

# Fixed image with proper dependencies
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.02-py3")
    # Install system dependencies first
    .apt_install(
        "ffmpeg",
        "libavcodec-dev", 
        "libavformat-dev",
        "libavdevice-dev",
        "libavutil-dev",
        "libavfilter-dev",
        "libswscale-dev",
        "libswresample-dev",
        "pkg-config"
    )
    .pip_install(
        "fastapi==0.115.5",
        "uvicorn==0.30.6",
        "aiortc==1.8.0",  # Stable version with fewer direction issues
        "av==11.0.0",     # Pre-built wheel, no compilation needed
        "numpy==1.26.4",
        "diffusers==0.33.1",
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "torch",
        "torchvision",
        "huggingface-hub",
        "sentencepiece",
        "einops",
        "timm",
        "imageio[ffmpeg]",
        "hf_transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": str(MODEL_PATH)})
)

@app.function(
    image=image,
    gpu=modal.gpu.H100(),
    concurrency_limit=5,
    keep_warm=1,
    allow_concurrent_inputs=10,
    timeout=60 * 30,
    volumes={str(OUTPUTS_PATH): outputs, str(MODEL_PATH): model}, 
)
@modal.asgi_app()
def webrtc_app():
    import asyncio
    import json
    import logging
    import weakref
    from fastapi import FastAPI, Body, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import numpy as np
    import av
    import torch

    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaStreamError
    from aiortc import VideoStreamTrack

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Global state for model and task tracking
    state = {
        "pipe": None, 
        "loading": False,
        "active_tracks": weakref.WeakSet(),
        "active_tasks": set()
    }

    app = FastAPI(title="LTX-Video WebRTC on Modal")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def get_pipe():
        """Load the LTX-Video pipeline properly."""
        if state["pipe"] is not None:
            return state["pipe"]
            
        if state["loading"]:
            # Wait for loading to complete
            while state["loading"]:
                await asyncio.sleep(0.1)
            return state["pipe"]
            
        state["loading"] = True
        
        try:
            # Use the native LTX-Video pipeline from the codebase
            from diffusers import DiffusionPipeline
            
            logger.info("Loading LTX-Video pipeline...")
            
            # Load the 2B distilled model (faster for streaming)
            pipe = DiffusionPipeline.from_pretrained(
                "Lightricks/LTX-Video-0.9.8-13B-distilled",
                torch_dtype=torch.bfloat16,
            )
            pipe.to("cuda")
            
            state["pipe"] = pipe
            logger.info("Pipeline loaded successfully!")
            return pipe
            
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            # Fallback to a simpler approach
            state["pipe"] = None
            raise
        finally:
            state["loading"] = False

    class LTXVideoTrack(VideoStreamTrack):
        """WebRTC video track with proper cleanup."""
        kind = "video"

        def __init__(self, prompt: str, num_frames: int = 48, height: int = 480, width: int = 854, fps: int = 12):
            super().__init__()
            self.prompt = prompt
            self.num_frames = min(num_frames, 48)
            self.h = height
            self.w = width
            self.fps = fps
            self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
            self.generation_complete = False
            self._shutdown = False
            self._producer_task = None
            
            # Track this instance
            state["active_tracks"].add(self)
            
            # Start frame production
            self._start_production()
            
            logger.info(f"Created video track: {prompt} ({num_frames} frames, {fps}fps)")

        def _start_production(self):
            """Start the frame production task."""
            if self._producer_task is None or self._producer_task.done():
                self._producer_task = asyncio.create_task(self._produce_frames())
                state["active_tasks"].add(self._producer_task)
                
                # Clean up task when done
                def cleanup_task(task):
                    state["active_tasks"].discard(task)
                
                self._producer_task.add_done_callback(cleanup_task)

        async def _produce_frames(self):
            """Produce frames with proper cancellation handling."""
            try:
                logger.info(f"Starting video generation for: {self.prompt}")
                
                # First, generate some immediate test frames
                # await self._generate_test_frames()
                
                # if self._shutdown:
                #     return
                
                # Then try LTX generation
                try:
                    pipe = await get_pipe()
                    if not self._shutdown:
                        await self._generate_ltx_frames(pipe)
                except Exception as e:
                    logger.warning(f"LTX generation failed: {e}, using test frames")
                    if not self._shutdown:
                        await self._generate_more_test_frames()
                        
            except asyncio.CancelledError:
                logger.info("Frame production cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in video generation: {e}")
                if not self._shutdown:
                    await self._generate_error_frames()
            finally:
                self.generation_complete = True
                if not self._shutdown:
                    try:
                        await self.frame_queue.put(None)  # End of stream marker
                    except:
                        pass  # Queue might be closed
                logger.info("Frame production completed")

        async def _generate_test_frames(self):
            """Generate immediate test frames."""
            logger.info("Generating test frames...")
            for i in range(24):  # 3 seconds at 8fps
                if self._shutdown:
                    break
                    
                # Create animated test frame
                frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                
                # Add animation
                color_intensity = int(127 + 127 * np.sin(i * 0.3))
                frame[:, :, 0] = color_intensity  # Red
                frame[:, :, 1] = 255 - color_intensity  # Green
                frame[:, :, 2] = 128  # Blue
                
                try:
                    await asyncio.wait_for(self.frame_queue.put(frame), timeout=1.0)
                    await asyncio.sleep(1.0 / self.fps)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    break

        async def _generate_ltx_frames(self, pipe):
            """Generate LTX-Video frames."""
            logger.info("Generating LTX-Video frames...")
            
            with torch.no_grad():
                result = pipe(
                    prompt=self.prompt,
                    height=self.h,
                    width=self.w,
                    num_frames=self.num_frames,
                    num_inference_steps=8,
                    guidance_scale=3.0,
                    output_type="pil",
                )
            
            frames = result.frames[0] if hasattr(result, 'frames') else result.images
            logger.info(f"Generated {len(frames)} LTX frames")
            from diffusers.utils import export_to_video
            mp4_name = slugify(self.prompt)
            export_to_video(frames, Path(OUTPUTS_PATH) / mp4_name)
            outputs.commit()
            
            for frame in frames:
                if self._shutdown:
                    break
                    
                frame_np = np.array(frame.convert("RGB"))
                try:
                    await asyncio.wait_for(self.frame_queue.put(frame_np), timeout=1.0)
                    await asyncio.sleep(1.0 / self.fps)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    break

        async def _generate_more_test_frames(self):
            """Generate more test frames."""
            for i in range(60):
                if self._shutdown:
                    break
                    
                frame = np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8)
                try:
                    await asyncio.wait_for(self.frame_queue.put(frame), timeout=1.0)
                    await asyncio.sleep(1.0 / self.fps)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    break

        async def _generate_error_frames(self):
            """Generate error frames."""
            for _ in range(30):
                if self._shutdown:
                    break
                    
                frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                frame[:, :, 0] = 255  # Red error frames
                try:
                    await asyncio.wait_for(self.frame_queue.put(frame), timeout=1.0)
                    await asyncio.sleep(1.0 / self.fps)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    break

        async def recv(self):
            """Receive frame with proper error handling."""
            try:
                if self._shutdown:
                    raise MediaStreamError
                    
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=2.0)
                
                if frame is None or self._shutdown:
                    logger.info("End of stream reached")
                    raise MediaStreamError
                
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts, video_frame.time_base = await self.next_timestamp()
                return video_frame
                
            except asyncio.TimeoutError:
                # Generate a timeout frame
                if self._shutdown:
                    raise MediaStreamError
                    
                timeout_frame = np.full((self.h, self.w, 3), 64, dtype=np.uint8)
                video_frame = av.VideoFrame.from_ndarray(timeout_frame, format="rgb24")
                video_frame.pts, video_frame.time_base = await self.next_timestamp()
                return video_frame
            except Exception as e:
                logger.error(f"recv() error: {e}")
                raise MediaStreamError

        def stop(self):
            """Stop the track and cleanup tasks."""
            logger.info("Stopping video track")
            self._shutdown = True
            
            # Cancel the producer task
            if self._producer_task and not self._producer_task.done():
                self._producer_task.cancel()
            
            # Clear the queue
            try:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            except:
                pass

        def __del__(self):
            """Cleanup on deletion."""
            self.stop()

    @app.get("/")
    def root():
        return {"ok": True, "service": "ltxv-webrtc-fixed", "status": "ready"}

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        try:
            pipe_status = "loaded" if state["pipe"] is not None else "not_loaded"
            loading_status = "loading" if state["loading"] else "idle"
            active_tracks = len(state["active_tracks"])
            active_tasks = len(state["active_tasks"])
            
            return {
                "status": "healthy", 
                "pipeline": pipe_status, 
                "loading": loading_status,
                "active_tracks": active_tracks,
                "active_tasks": active_tasks
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @app.post("/offer")
    async def offer(
        sdp: str = Body(..., embed=True),
        type: str = Body(..., embed=True),
        prompt: str = Body("A corgi surfing on a neon wave", embed=True),
        num_frames: int = Body(48, embed=True),
        height: int = Body(384, embed=True),
        width: int = Body(672, embed=True),
        fps: int = Body(12, embed=True),
    ):
        """WebRTC offer with proper aiortc configuration."""
        pc = None
        track = None
        
        try:
            logger.info(f"Received offer for prompt: {prompt}")
            
            # FIXED: aiortc expects RTCConfiguration object, not dict
            from aiortc import RTCConfiguration, RTCIceServer
            
            config = RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                    RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
                    RTCIceServer(urls=["stun:stun2.l.google.com:19302"]),
                ]
            )
            pc = RTCPeerConnection(configuration=config)
            logger.info("RTCPeerConnection created with ICE servers")
            
            # Create video track
            track = LTXVideoTrack(
                prompt=prompt, 
                num_frames=min(num_frames, 48),
                height=height, 
                width=width, 
                fps=fps
            )
            
            # Add transceiver
            transceiver = pc.addTransceiver(track, direction="sendonly")
            logger.info(f"Added sendonly transceiver: {transceiver}")
            
            # Set remote description
            logger.info("Setting remote description...")
            offer_obj = RTCSessionDescription(sdp=sdp, type=type)
            await pc.setRemoteDescription(offer_obj)
            logger.info("Remote description set successfully")
            
            # Create answer
            logger.info("Creating answer...")
            answer = await pc.createAnswer()
            logger.info(f"Answer created: {len(answer.sdp)} chars")
            
            # Enhanced monkey patch for direction issues
            import aiortc.rtcpeerconnection as rtc_module
            original_and_direction = rtc_module.and_direction
            
            def safe_and_direction(a, b):
                logger.info(f"and_direction: a={a}, b={b}")
                if a is None or b is None:
                    logger.warning(f"None direction, defaulting to sendonly")
                    return "sendonly"
                try:
                    result = original_and_direction(a, b)
                    logger.info(f"and_direction result: {result}")
                    return result
                except Exception as e:
                    logger.warning(f"Direction error: {e}, defaulting to sendonly")
                    return "sendonly"
            
            rtc_module.and_direction = safe_and_direction
            
            try:
                await pc.setLocalDescription(answer)
                logger.info("Local description set successfully")
            finally:
                rtc_module.and_direction = original_and_direction
            
            # Enhanced monitoring
            @pc.on("connectionstatechange")
            async def on_connection_state():
                logger.info(f"Connection state changed: {pc.connectionState}")
                
            @pc.on("iceconnectionstatechange")
            async def on_ice_state():
                logger.info(f"ICE connection state changed: {pc.iceConnectionState}")
                if pc.iceConnectionState == "failed":
                    logger.error("ICE connection failed")
                    if track:
                        track.stop()
            
            @pc.on("icegatheringstatechange")
            async def on_ice_gathering():
                logger.info(f"ICE gathering state: {pc.iceGatheringState}")

            logger.info("WebRTC offer processed successfully")
            return JSONResponse({
                "sdp": pc.localDescription.sdp, 
                "type": pc.localDescription.type,
                "status": "success"
            })
            
        except Exception as e:
            import traceback
            logger.error(f"Error in offer endpoint: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if track:
                track.stop()
            if pc:
                try:
                    await pc.close()
                except:
                    pass
                    
            raise HTTPException(status_code=500, detail=f"Failed to process offer: {str(e)}")
    # Cleanup function for graceful shutdown
    @app.on_event("shutdown")
    async def cleanup_on_shutdown():
        """Clean up all active tasks on shutdown."""
        logger.info("Cleaning up active tasks...")
        
        # Stop all active tracks
        for track in list(state["active_tracks"]):
            try:
                track.stop()
            except:
                pass
        
        # Cancel all active tasks
        for task in list(state["active_tasks"]):
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if state["active_tasks"]:
            await asyncio.gather(*state["active_tasks"], return_exceptions=True)
        
        logger.info("Cleanup completed")

    return app
