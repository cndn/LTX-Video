
import string
import time
from pathlib import Path
from typing import Optional

import modal
app = modal.App("example-ltx")
MINUTES = 60  # seconds
# image = (
#     modal.Image.debian_slim(python_version="3.12")
#     .pip_install(
#         "accelerate==1.6.0",
#         "diffusers==0.33.1",
#         "hf_transfer==0.1.9",
#         "imageio==2.37.0",
#         "imageio-ffmpeg==0.5.1",
#         "sentencepiece==0.2.0",
#         "torch==2.7.0",
#         "transformers==4.51.3",
#     )
#     .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
# )
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.02-py3")
    .pip_install(
        "accelerate==1.6.0",
        "diffusers==0.33.1",
        "hf_transfer==0.1.9",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.5.1",
        "sentencepiece==0.2.0",
        "transformers==4.51.3",
        ""
        # torch comes preinstalled inside this base image
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
VOLUME_NAME = "ltx-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")
MODEL_VOLUME_NAME = "ltx-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models")
image = image.env({"HF_HOME": str(MODEL_PATH)})


def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # some OSes limit filenames to <256 chars
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name

@app.cls(
    image=image,  # use our container Image
    volumes={OUTPUTS_PATH: outputs, MODEL_PATH: model},  # attach our Volumes
    gpu="B200",  # use a big, fast GPU
    timeout=10 * MINUTES,  # run inference for up to 10 minutes
    scaledown_window=15 * MINUTES,  # stay idle for 15 minutes before scaling down
    min_containers=1,
)
class LTX:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline

        self.pipe = DiffusionPipeline.from_pretrained(
            "Lightricks/LTX-Video-0.9.8-13B-distilled", torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

    @modal.method()
    def generate(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=8,
        num_frames=48,
        width=832,
        height=480,
    ):
        from diffusers.utils import export_to_video

        frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            width=width,
            height=height,
        ).frames[0]

        # save to disk using prompt as filename
        mp4_name = slugify(prompt)
        export_to_video(frames, Path(OUTPUTS_PATH) / mp4_name)
        outputs.commit()
        return mp4_name

sb_app = modal.App.lookup("sb_app", create_if_missing=True)
sb = modal.Sandbox.create(
    image=image,
    volumes={OUTPUTS_PATH: outputs, MODEL_PATH: model},
    workdir="/repo",
    app=sb_app,
)
process = sb.exec("modal", "run scripts/modal_ltx_inference.py --prompt 'astronauts walks on the moon and some random person appears'", timeout=300)