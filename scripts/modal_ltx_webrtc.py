import os
from pathlib import Path

import modal

from .modal_webrtc import ModalWebRtcPeer, ModalWebRtcSignalingServer

py_version = "3.12"
tensorrt_ld_path = f"/usr/local/lib/python{py_version}/site-packages/tensorrt_libs"

video_processing_image = (
    modal.Image.debian_slim(python_version=py_version)  # matching ld path
    # update locale as required by onnx
    .apt_install("locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",  # use sed to uncomment
        "locale-gen en_US.UTF-8",  # set locale
        "update-locale LANG=en_US.UTF-8",
    )
    .env({"LD_LIBRARY_PATH": tensorrt_ld_path, "LANG": "en_US.UTF-8"})
    # install system dependencies
    .apt_install("python3-opencv", "ffmpeg")
    # install Python dependencies
    .pip_install(
        "aiortc==1.11.0",
        "fastapi==0.115.12",
        "huggingface-hub[hf_xet]==0.30.2",
        "onnxruntime-gpu==1.21.0",
        "opencv-python==4.11.0.86",
        "tensorrt==10.9.0.34",
        "torch==2.7.0",
        "shortuuid==1.0.13",
    )
)

CACHE_VOLUME = modal.Volume.from_name("webrtc-yolo-cache", create_if_missing=True)
CACHE_PATH = Path("/cache")
cache = {CACHE_PATH: CACHE_VOLUME}

app = modal.App("example-webrtc-yolo")

