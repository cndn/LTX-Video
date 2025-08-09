# write_video_from_npy.py
import numpy as np
import imageio.v2 as imageio

IN_PATH = "test_tensor.npy"
OUT_PATH = "out.mp4"
FPS = 24

def to_T_H_W_C(arr: np.ndarray) -> np.ndarray:
    """Convert common video layouts to (T,H,W,C)."""
    if arr.ndim == 5:
        # (B, C, T, H, W)
        if arr.shape[1] in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 4, 1))  # -> (B,T,H,W,C)
            arr = arr.reshape(-1, *arr.shape[2:])     # flatten B,T -> T
        # (B, T, C, H, W)
        elif arr.shape[2] in (1, 3, 4):
            arr = np.transpose(arr, (0, 1, 3, 4, 2))  # -> (B,T,H,W,C)
            arr = arr.reshape(-1, *arr.shape[2:])
        # (B, T, H, W, C)
        elif arr.shape[-1] in (1, 3, 4):
            arr = arr.reshape(-1, *arr.shape[2:])
        else:
            raise ValueError(f"Don't know how to handle 5D shape {arr.shape}")
    elif arr.ndim == 4:
        # (T, H, W, C)
        if arr.shape[-1] in (1, 3, 4):
            pass
        # (T, C, H, W)
        elif arr.shape[1] in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))
        # (C, T, H, W)
        elif arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 3, 0))
        else:
            raise ValueError(f"Don't know how to handle 4D shape {arr.shape}")
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} (need 4D or 5D)")
    return arr

def to_hwc3_uint8(x: np.ndarray) -> np.ndarray:
    """Ensure (H,W,3) uint8 with sane range."""
    x = np.asarray(x)
    if x.ndim == 2:  # (H,W) -> (H,W,3)
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.concatenate([x, x, x], axis=2)
    elif x.ndim == 3 and x.shape[2] > 3:
        x = x[:, :, :3]

    x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)

    if np.issubdtype(x.dtype, np.floating):
        mn, mx = float(x.min()), float(x.max())
        if mn >= -1.1 and mx <= 1.1 and mn < 0:   # [-1,1]
            x = (x + 1.0) * 0.5
            x = np.clip(x, 0, 1) * 255.0
        elif mn >= 0.0 and mx <= 1.0:             # [0,1]
            x = np.clip(x, 0, 1) * 255.0
        else:
            x = np.clip(x, 0, 255.0)
        x = x.astype(np.uint8, copy=False)
    elif x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8, copy=False)

    return np.ascontiguousarray(x)

# --- Load and arrange ---
arr = np.load(IN_PATH, mmap_mode="r")
frames = to_T_H_W_C(arr)
print("frames:", frames.shape, frames.dtype)  # e.g., (17, 256, 384, 3)

# --- Write using FFmpeg backend (even dims + yuv420p) ---
writer = imageio.get_writer(
    OUT_PATH,
    fps=FPS,
    codec="libx264",
    format="FFMPEG",
    ffmpeg_params=[
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt", "yuv420p",
    ],
)

with writer:
    f0 = to_hwc3_uint8(frames[0])
    print("first frame stats -> min/max/mean:", f0.min(), f0.max(), float(f0.mean()))
    writer.append_data(f0)
    for i in range(1, frames.shape[0]):
        writer.append_data(to_hwc3_uint8(frames[i]))

print(f"Saved {OUT_PATH}")
