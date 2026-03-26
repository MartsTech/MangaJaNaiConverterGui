import argparse
import ctypes
import io
import json
import os
import platform
import re
import sys
import threading
import time
import math
from collections.abc import Callable
from collections import Counter
from io import BytesIO
from pathlib import Path
from queue import Queue
from multiprocessing import Queue as MPQueue, Process
from threading import Thread
from typing import Any, Literal
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
import pyvips
import rarfile
import torch
import torch.nn.functional as F
from chainner_ext import ResizeFilter, resize
from cv2.typing import MatLike
from PIL import Image, ImageCms, ImageFilter
from PIL.Image import Image as ImageType
from PIL.ImageCms import ImageCmsProfile
from rarfile import RarFile
from spandrel import ImageModelDescriptor, ModelDescriptor

sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))

from mangajanaitrt.trt_upscaler import TensorRTUpscaler
try:
    from mangajanaitrt.vram_monitor import MultiGPUVRAMMonitor
except ImportError:
    MultiGPUVRAMMonitor = None

import spandrel_custom
from nodes.impl.image_utils import normalize, to_uint8, to_uint16
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    MAX_TILE_SIZE,
    NO_TILING,
    TileSize,
)
from nodes.utils.utils import get_h_w_c
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node
from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
    upscale_image_node,
)
from progress_controller import ProgressController, ProgressToken

from api import (
    NodeContext,
    SettingsParser,
)

# Global lock to ensure thread-safe TensorRT Engine building
engine_build_lock = threading.Lock()


class _ExecutorNodeContext(NodeContext):
    def __init__(
        self, progress: ProgressToken, settings: SettingsParser, storage_dir: Path
    ) -> None:
        super().__init__()

        self.progress = progress
        self.__settings = settings
        self._storage_dir = storage_dir

        self.chain_cleanup_fns: set[Callable[[], None]] = set()
        self.node_cleanup_fns: set[Callable[[], None]] = set()

    @property
    def aborted(self) -> bool:
        return self.progress.aborted

    @property
    def paused(self) -> bool:
        time.sleep(0.001)
        return self.progress.paused

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

        # TODO: send progress event

    @property
    def settings(self) -> SettingsParser:
        """
        Returns the settings of the current node execution.
        """
        return self.__settings

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        if after == "chain":
            self.chain_cleanup_fns.add(fn)
        elif after == "node":
            self.node_cleanup_fns.add(fn)
        else:
            raise ValueError(f"Unknown cleanup type: {after}")


def get_tile_size(tile_size_str: str) -> TileSize:
    if tile_size_str == "Auto (Estimate)":
        return ESTIMATE
    elif tile_size_str == "Maximum":
        return MAX_TILE_SIZE
    elif tile_size_str == "No Tiling":
        return NO_TILING
    elif tile_size_str.isdecimal():
        return TileSize(int(tile_size_str))

    return ESTIMATE

# --- DYNAMIC SMART PADDING ---
def add_smart_padding(image: np.ndarray, d_pre: int, d_post: int, native_scale: int, force_bottom: bool = False, force_odd_w_pad: str = None) -> np.ndarray:
    """
    Calculates and applies exact edge padding so the canvas is perfectly 
    divisible by both the pre-upscale fraction and the post-upscale fraction.
    This operates on the raw uncompressed NumPy array in memory, making it 100% lossless.
    """
    h, w = image.shape[:2]

    def get_required_padding(dim):
        # Future-proof math: the maximum padding needed to satisfy both divisions
        # is strictly bounded by the product of the two divisors.
        max_search_space = max(50, (d_pre * d_post) + 1)
        for pad in range(max_search_space):
            new_dim = dim + pad
            if new_dim % d_pre == 0:
                up_dim = (new_dim // d_pre) * native_scale  # Dynamic based on actual model behavior
                if up_dim % d_post == 0:
                    return pad
        return 0

    pad_w = get_required_padding(w)
    pad_h = get_required_padding(h)

    if pad_w == 0 and pad_h == 0:
        return image

    def get_dominant_freq(edge_array):
        # edge_array shape: Grayscale [Length], Color [Length, Channels]
        if edge_array.ndim == 2:
            pixels = [tuple(p) for p in edge_array]
        else:
            pixels = edge_array.tolist()
        counts = Counter(pixels)
        return counts.most_common(1)[0][1] if counts else 0

    top_pad = bottom_pad = left_pad = right_pad = 0

    # Handle Width Padding
    if pad_w > 0:
        if pad_w % 2 == 0:
            left_pad = right_pad = pad_w // 2
        else:
            half = pad_w // 2
            # Webtoon Strict Override: Force odd pixel to the voted side for vertical alignment
            if force_odd_w_pad == "left":
                left_pad = half + 1
                right_pad = half
            elif force_odd_w_pad == "right":
                right_pad = half + 1
                left_pad = half
            else:
                # Standard dynamic behavior
                freq_l = get_dominant_freq(image[:, 0])
                freq_r = get_dominant_freq(image[:, -1])
                if freq_l > freq_r:
                    left_pad = half + 1
                    right_pad = half
                else:
                    right_pad = half + 1
                    left_pad = half

    # Handle Height Padding
    if pad_h > 0:
        if force_bottom:
            # Webtoon strict override: all vertical padding goes to the bottom
            bottom_pad = pad_h
            top_pad = 0
        else:
            if pad_h % 2 == 0:
                top_pad = bottom_pad = pad_h // 2
            else:
                freq_t = get_dominant_freq(image[0, :])
                freq_b = get_dominant_freq(image[-1, :])
                half = pad_h // 2
                if freq_t > freq_b:
                    top_pad = half + 1
                    bottom_pad = half
                else:
                    bottom_pad = half + 1
                    top_pad = half

    # Apply padding losslessly using BORDER_REPLICATE
    padded_image = cv2.copyMakeBorder(
        image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REPLICATE
    )
    
    print("\n" + "="*40, flush=True)
    print("--- SMART PADDING TRIGGERED ---", flush=True)
    print(f"Original Canvas: {w}x{h}", flush=True)
    print(f"Native Model Scale: {native_scale}x", flush=True)
    print(f"Target Fractions: Downscale-Before = 1/{d_pre}, Downscale-After = 1/{d_post}", flush=True)
    print(f"Padding Added:", flush=True)
    if pad_w > 0:
        w_override_text = f" [Webtoon Override: {force_odd_w_pad.capitalize()}]" if force_odd_w_pad else ""
        print(f"  -> Width:  +{pad_w}px (Left: {left_pad}px, Right: {right_pad}px){w_override_text}", flush=True)
    if pad_h > 0:
        h_override_text = " [Webtoon Override: Bottom Only]" if force_bottom else ""
        print(f"  -> Height: +{pad_h}px (Top: {top_pad}px, Bottom: {bottom_pad}px){h_override_text}", flush=True)
    print(f"New Canvas: {w+pad_w}x{h+pad_h}", flush=True)
    print("="*40 + "\n", flush=True)
    
    return padded_image


def wavelet_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: torch.Tensor, levels: int = 5):
    high_freq = torch.zeros_like(image)
    low_freq = image
    for i in range(levels):
        low_freq = wavelet_blur(image, radius=2**i)
        high_freq += image - low_freq
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(
    content_feat: torch.Tensor, style_feat: torch.Tensor, levels: int
) -> torch.Tensor:
    content_high_freq, _ = wavelet_decomposition(content_feat, levels=levels)
    _, style_low_freq = wavelet_decomposition(style_feat, levels=levels)
    return content_high_freq + style_low_freq


def apply_wavelet_color_fix(
    target_img: np.ndarray, source_img: np.ndarray, levels: int = 5
) -> np.ndarray:
    """
    Applies wavelet color fix using the source_img (original) as the color reference
    and target_img (upscaled) as the content reference.
    """
    device_idx = settings_parser.get_int("accelerator_device_index", 0)
    
    if torch.cuda.is_available():
        # DOCKER FIX: If PyTorch sees fewer GPUs than the requested index 
        # (due to CUDA_VISIBLE_DEVICES), map it back to 0 to prevent crashes.
        if device_idx >= torch.cuda.device_count():
            device_idx = 0
        device = torch.device(f"cuda:{device_idx}")
    else:
        device = torch.device("cpu")

    target_h, target_w, _ = get_h_w_c(target_img)

    # Resize source image (original) to match target image (upscaled)
    # Using Box filter is generally better for the 'color' reference pass to avoid ringing
    source_img_resized = resize(
        source_img, (target_w, target_h), ResizeFilter.Box, False
    )

    # Helper to convert Numpy HWC [0-1] to Tensor BCHW
    def to_tensor(img):
        # Assumes float32 0-1 input
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return t.to(device)

    target_tensor = to_tensor(target_img)
    source_tensor_resized = to_tensor(source_img_resized)

    with torch.no_grad():
        result_tensor = wavelet_reconstruction(
            target_tensor, source_tensor_resized, levels=levels
        )
        result_tensor = torch.clamp(result_tensor, 0, 1)

    # Convert back to Numpy HWC
    result_img = result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return result_img

# --- WAVELET COLOR FIX FUNCTIONS END ---

def upscale_alpha(alpha: np.ndarray, scale: int) -> np.ndarray:
    """Upscale alpha channel separately to ensure transparency is preserved"""
    h, w = alpha.shape
    img = pyvips.Image.new_from_memory(alpha.tobytes(), w, h, 1, "uchar")
    img = img.resize(scale, kernel="cubic")
    return np.ndarray(
        buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width]
    )

"""
lanczos downscale without color conversion, for pre-upscale
downscale and final color downscale
"""

def standard_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    h, w, _ = get_h_w_c(image)
    if (w, h) == new_size:
        return image
    
    # prevent upscaling
    if new_size[0] > w or new_size[1] > h:
        return image
        
    print(f"Applying standard_resize (Lanczos) to {new_size}", flush=True)
    
    new_image = image.astype(np.float32) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.Lanczos, False)
    new_image = (new_image * 255).round().astype(np.uint8)

    _, _, c = get_h_w_c(image)

    if c == 1 and new_image.ndim == 3:
        new_image = np.squeeze(new_image, axis=-1)

    return new_image


"""
final downscale for grayscale images only
"""


def dotgain20_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    h, w, _ = get_h_w_c(image)
    if (w, h) == new_size:
        return image
    
    # prevent upscaling
    if new_size[0] > w or new_size[1] > h:
        return image
        
    print(f"Applying dotgain20_resize to {new_size}", flush=True)
    
    shrink_factor = h / new_size[1]
    
    # NEW LOGIC: Check if the shrink factor is essentially a perfect integer (e.g., 2.0, 3.0)
    # Using a small tolerance (0.01) to account for floating point inaccuracies
    is_integer_shrink = abs(shrink_factor - round(shrink_factor)) < 0.01
    
    # Determine filter and blur dynamically
    if is_integer_shrink:
        blur_size = 0  # No blur needed for perfect integers
        chosen_filter = ResizeFilter.Box
        print(f"Exact integer downscale detected (~{round(shrink_factor)}x). Using Box filter with no blur.", flush=True)
    else:
        blur_size = (shrink_factor - 1) / 3.5
        if blur_size >= 0.1:
            blur_size = min(blur_size, 250)
        else:
            blur_size = 0
        chosen_filter = ResizeFilter.CubicCatrom
        print(f"Fractional downscale detected. Using Catrom filter with blur radius {blur_size:.2f}.", flush=True)

    pil_image = Image.fromarray(image, mode="L")
    
    if blur_size > 0:
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_size))
        
    pil_image = ImageCms.applyTransform(pil_image, dotgain20togamma1transform, False)

    new_image = np.array(pil_image)
    new_image = new_image.astype(np.float32) / 255.0
    
    # Apply the dynamically chosen filter
    new_image = resize(new_image, new_size, chosen_filter, False)
    
    new_image = (new_image * 255).round().astype(np.uint8)

    pil_image = Image.fromarray(new_image[:, :, 0], mode="L")
    pil_image = ImageCms.applyTransform(pil_image, gamma1todotgain20transform, False)
    return np.array(pil_image)


def image_resize(
    image: np.ndarray, new_size: tuple[int, int], is_grayscale: bool, force_standard: bool = False
) -> np.ndarray:
    if is_grayscale and not force_standard:
        return dotgain20_resize(image, new_size)
    if force_standard and is_grayscale:
        print("Forcing standard resize for grayscale image", flush=True)

    return standard_resize(image, new_size)


def get_system_codepage() -> Any:
    return None if not is_windows else ctypes.windll.kernel32.GetConsoleOutputCP()


def get_force_standard_resize(name: str) -> bool:
    lower_name = name.lower()
    return lower_name.endswith("(gray)") or lower_name.endswith("(gray4)") or lower_name.endswith("(gray-2048)") or lower_name.endswith("(gray4-2048)")


def enhance_contrast(image: np.ndarray) -> MatLike:
    image_p = Image.fromarray(image).convert("L")

    # Calculate the histogram
    hist = image_p.histogram()
    # print(hist)

    # Find the global maximum peak in the range 0-30 for the black level
    new_black_level = 0
    global_max_black = hist[0]

    for i in range(1, 31):
        if hist[i] > global_max_black:
            global_max_black = hist[i]
            new_black_level = i
        # elif hist[i] < global_max_black:
        #     break

    # Continue searching at 31 and later for the black level
    continuous_count = 0
    for i in range(31, 256):
        if hist[i] > global_max_black:
            continuous_count = 0
            global_max_black = hist[i]
            new_black_level = i
        elif hist[i] < global_max_black:
            continuous_count += 1
            if continuous_count > 1:
                break

    # Find the global maximum peak in the range 255-225 for the white level
    new_white_level = 255
    global_max_white = hist[255]

    for i in range(254, 239, -1):
        if hist[i] > global_max_white:
            global_max_white = hist[i]
            new_white_level = i
        # elif hist[i] < global_max_white:
        #     break

    # Continue searching at 224 and below for the white level
    continuous_count = 0
    for i in range(239, -1, -1):
        if hist[i] > global_max_white:
            continuous_count = 0
            global_max_white = hist[i]
            new_white_level = i
        elif hist[i] < global_max_white:
            continuous_count += 1
            if continuous_count > 1:
                break

    print(
        f"Auto adjusted levels: new black level = {new_black_level}; new white level = {new_white_level}",
        flush=True,
    )

    image_array = np.array(image_p).astype("float32")
    image_array = np.maximum(image_array - new_black_level, 0) / (
        new_white_level - new_black_level
    )
    return np.clip(image_array, 0, 1)


def _read_image(img_stream: bytes, filename: str) -> np.ndarray:
    return _read_vips(img_stream)


def _read_image_from_path(path: str) -> np.ndarray:
    return pyvips.Image.new_from_file(path, access="sequential", fail=True).icc_transform("srgb").numpy()


def _read_vips(img_stream: bytes) -> np.ndarray:
    return pyvips.Image.new_from_buffer(img_stream, "", access="sequential").icc_transform("srgb").numpy()


def cv_image_is_grayscale(image: np.ndarray, user_threshold: float) -> bool:
    _, _, c = get_h_w_c(image)

    if c == 1:
        return True

    b, g, r = cv2.split(image[:, :, :3])

    ignore_threshold = user_threshold

    # getting differences between (b,g), (r,g), (b,r) channel pixels
    r_g = cv2.subtract(cv2.absdiff(r, g), ignore_threshold)  # type: ignore
    r_b = cv2.subtract(cv2.absdiff(r, b), ignore_threshold)  # type: ignore
    g_b = cv2.subtract(cv2.absdiff(g, b), ignore_threshold)  # type: ignore

    # create masks to identify pure black and pure white pixels
    pure_black_mask = np.logical_and.reduce((r == 0, g == 0, b == 0))
    pure_white_mask = np.logical_and.reduce((r == 255, g == 255, b == 255))

    # combine masks to exclude both pure black and pure white pixels
    exclude_mask = np.logical_or(pure_black_mask, pure_white_mask)

    # exclude pure black and pure white pixels from diff_sum and image size calculation
    diff_sum = np.sum(np.where(exclude_mask, 0, r_g + r_b + g_b))
    size_without_black_and_white = np.sum(~exclude_mask) * 3

    # if the entire image is pure black or pure white, return False
    if size_without_black_and_white == 0:
        return True

    # finding ratio of diff_sum with respect to size of image without pure black and pure white pixels
    ratio = diff_sum / size_without_black_and_white

    return ratio <= user_threshold / 12


def convert_image_to_grayscale(image: np.ndarray) -> np.ndarray:
    channels = get_h_w_c(image)[2]
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return image


def get_chain_for_image(
    image: np.ndarray,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    grayscale_detection_threshold: int,
) -> tuple[dict[str, Any] | None, bool, int, int]:
    original_height, original_width, _ = get_h_w_c(image)

    if target_width != 0 and target_height != 0:
        target_scale = min(
            target_height / original_height, target_width / original_width
        )
    if target_height != 0:
        target_scale = target_height / original_height
    elif target_width != 0:
        target_scale = target_width / original_width

    assert target_scale is not None

    is_grayscale = cv_image_is_grayscale(image, grayscale_detection_threshold)

    for chain in chains:
        if should_chain_activate_for_image(
            original_width, original_height, is_grayscale, target_scale, chain
        ):
            print("Matched Chain:", chain, flush=True)
            return chain, is_grayscale, original_width, original_height

    # --- FIX: PURE WHITE/BLACK CHUNK FALLBACK ---
    # If an image is 100% pure white, the grayscale detector flags it as True.
    # If the user only selected a Color model, the chain match fails and the chunk is skipped.
    # Since grayscale/white images are perfectly fine to process through color models,
    # we fallback to pretending it is a color image so the dimensions don't break.
    if is_grayscale:
        for chain in chains:
            if should_chain_activate_for_image(
                original_width, original_height, False, target_scale, chain
            ):
                print("[Fallback] Pure white/grayscale chunk detected, but no Grayscale chain exists. Routing through Color chain.", flush=True)
                return chain, False, original_width, original_height

    return None, is_grayscale, original_width, original_height


def should_chain_activate_for_image(
    original_width: int,
    original_height: int,
    is_grayscale: bool,
    target_scale: float,
    chain: dict[str, Any],
) -> bool:
    min_width, min_height = (int(x) for x in chain["MinResolution"].split("x"))
    max_width, max_height = (int(x) for x in chain["MaxResolution"].split("x"))

    # resolution tests
    if min_width != 0 and min_width > original_width:
        return False
    if min_height != 0 and min_height > original_height:
        return False
    if max_width != 0 and max_width < original_width:
        return False
    if max_height != 0 and max_height < original_height:
        return False

    # color / grayscale tests
    if is_grayscale and not chain["IsGrayscale"]:
        return False
    if not is_grayscale and not chain["IsColor"]:
        return False

    # scale tests
    if chain["MaxScaleFactor"] != 0 and target_scale > chain["MaxScaleFactor"]:
        return False
    if chain["MinScaleFactor"] != 0 and target_scale < chain["MinScaleFactor"]:
        return False

    return True


def ai_upscale_image(
    image: np.ndarray, model_tile_size: TileSize, model: ImageModelDescriptor | Any | None
) -> np.ndarray:
    if model is not None:
        if TensorRTUpscaler is not None and isinstance(model, TensorRTUpscaler):
            if image.dtype != np.uint8:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)

            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)
            
            _, _, c = get_h_w_c(image)
            alpha = None
            if c == 4:
                alpha = image[:, :, 3]  # Extract Alpha
                image = image[:, :, :3] # Keep RGB
            
            result = model.upscale_image(image, overlap=16)

            # Re-attach and upscale alpha if it exists
            if alpha is not None:
                scale = result.shape[0] // image.shape[0] # dynamically calculate scale factor
                result_alpha = upscale_alpha(alpha, scale)
                result = np.dstack([result, result_alpha])

            if result.dtype == np.uint8:
                result = result.astype(np.float32) / 255.0

            _, _, c = get_h_w_c(result)
            if c == 1 and result.ndim == 3:
                result = np.squeeze(result, axis=-1)

            return result

        result = upscale_image_node(
            context,
            image,
            model,
            False,
            0,
            model_tile_size,
            256,
            False,
        )

        _, _, c = get_h_w_c(image)

        if c == 1 and result.ndim == 3:
            result = np.squeeze(result, axis=-1)

        return result

    return image


def postprocess_image(image: np.ndarray) -> np.ndarray:
    # print(f"postprocess_image")
    return to_uint8(image, normalized=True)


def final_target_resize(
    image: np.ndarray,
    target_scale: float,
    target_width: int,
    target_height: int,
    original_width: int,
    original_height: int,
    is_grayscale: bool,
    force_standard_resize: bool = False,
    is_webtoon: bool = False,
) -> np.ndarray:
    
    # --- WEBTOON EXACT DIMENSION OVERRIDE (ASPECT RATIO LOCKED) ---
    if is_webtoon and target_width > 0:
        # We completely ignore target_height (4000) here to prevent squishing.
        # We calculate the mathematically perfect proportional height.
        expected_h = original_height * (target_width / original_width)
        expected_h_int = math.ceil(expected_h)
        
        h, w, _ = get_h_w_c(image)
        if w != target_width or h != expected_h_int:
            return image_resize(image, (target_width, expected_h_int), is_grayscale, force_standard_resize)
        return image
    # ----------------------------------------

    # fit to dimensions
    if target_height != 0 and target_width != 0:
        h, w, _ = get_h_w_c(image)
        # determine whether to fit to height or width
        if target_height / original_height < target_width / original_width:
            target_width = 0
        else:
            target_height = 0

    # resize height, keep proportional width
    if target_height != 0:
        h, w, _ = get_h_w_c(image)
        if h != target_height:
            return image_resize(
                image, (round(w * target_height / h), target_height), is_grayscale, force_standard_resize
            )
    # resize width, keep proportional height
    elif target_width != 0:
        h, w, _ = get_h_w_c(image)
        if w != target_width:
            return image_resize(
                image, (target_width, round(h * target_width / w)), is_grayscale, force_standard_resize
            )
    else:
        h, w, _ = get_h_w_c(image)
        new_target_height = round(original_height * target_scale)
        if h != new_target_height:
            return image_resize(
                image,
                (round(w * new_target_height / h), new_target_height),
                is_grayscale,
                force_standard_resize
            )

    return image


def save_image_zip(
    image: np.ndarray,
    file_name: str,
    output_zip: ZipFile,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    original_width: int,
    original_height: int,
    target_scale: float,
    target_width: int,
    target_height: int,
    is_grayscale: bool,
    force_standard_resize: bool = False,
    is_webtoon: bool = False,
) -> None:
    print(f"save image to zip: {file_name}", flush=True)

    image = to_uint8(image, normalized=True)

    image = final_target_resize(
        image,
        target_scale,
        target_width,
        target_height,
        original_width,
        original_height,
        is_grayscale,
        force_standard_resize,
        is_webtoon,
    )

    # Convert the resized image back to bytes
    args = {"Q": int(lossy_compression_quality)}
    if image_format in {"webp"}:
        args["lossless"] = use_lossless_compression
        # Webtoon strict override: Use high-quality chroma subsampling (-sharp_yuv equivalent)
        if is_webtoon and not use_lossless_compression:
            args["smart_subsample"] = True
            
    buf_img = pyvips.Image.new_from_array(image).write_to_buffer(f".{image_format}", **args)
    output_buffer = io.BytesIO(buf_img)  # type: ignore

    upscaled_image_data = output_buffer.getvalue()

    # Add the resized image to the output zip
    output_zip.writestr(file_name, upscaled_image_data)


def save_image(
    image: np.ndarray,
    output_file_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    original_width: int,
    original_height: int,
    target_scale: float,
    target_width: int,
    target_height: int,
    is_grayscale: bool,
    force_standard_resize: bool = False,
    is_webtoon: bool = False,
) -> None:
    print(f"save image: {output_file_path}", flush=True)

    image = to_uint8(image, normalized=True)

    image = final_target_resize(
        image,
        target_scale,
        target_width,
        target_height,
        original_width,
        original_height,
        is_grayscale,
        force_standard_resize,
        is_webtoon,
    )

    args = {"Q": int(lossy_compression_quality)}
    if image_format in {"webp"}:
        print(f"Saving with lossless={use_lossless_compression}", flush=True)
        args["lossless"] = use_lossless_compression
        # Webtoon strict override: Use high-quality chroma subsampling (-sharp_yuv equivalent)
        if is_webtoon and not use_lossless_compression:
            args["smart_subsample"] = True
            
    pyvips.Image.new_from_array(image).write_to_file(output_file_path, **args)


def preprocess_worker_archive(
    upscale_queue: Queue,
    input_archive_path: str,
    output_archive_path: str,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
    force_standard_resize: bool,
) -> None:
    """
    given a zip or rar path, read images out of the archive, apply auto levels, add the image to upscale queue
    """

    if input_archive_path.endswith(ZIP_EXTENSIONS):
        with ZipFile(input_archive_path, "r") as input_zip:
            preprocess_worker_archive_file(
                upscale_queue,
                input_zip,
                output_archive_path,
                target_scale,
                target_width,
                target_height,
                chains,
                loaded_models,
                grayscale_detection_threshold,
                force_standard_resize,
            )
    elif input_archive_path.endswith(RAR_EXTENSIONS):
        with rarfile.RarFile(input_archive_path, "r") as input_rar:
            preprocess_worker_archive_file(
                upscale_queue,
                input_rar,
                output_archive_path,
                target_scale,
                target_width,
                target_height,
                chains,
                loaded_models,
                grayscale_detection_threshold,
                force_standard_resize,
            )


def preprocess_worker_archive_file(
    upscale_queue: Queue,
    input_archive: RarFile | ZipFile,
    output_archive_path: str,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
    force_standard_resize: bool,
) -> None:
    """
    given an input zip or rar archive, read images out of the archive, apply auto levels, add the image to upscale queue.
    Includes smart pre-upscale Webtoon splicing with optimal math rounding, majority voting, and metadata preservation.
    """
    import math
    import re

    os.makedirs(os.path.dirname(output_archive_path), exist_ok=True)
    namelist = input_archive.namelist()
    print(f"TOTALZIP={len(namelist)}", flush=True)

    def natural_sort_key(s: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

    def process_and_queue_image(image_array: np.ndarray, decoded_filename: str, is_webtoon_chunk: bool = False, force_odd_w_pad: str = None, unpadded_h: int = 0):
        """Helper to process a numpy array through the chain logic and queue it."""
        chain, is_grayscale, original_width, original_height = (
            get_chain_for_image(
                image_array,
                target_scale,
                target_width,
                target_height,
                chains,
                grayscale_detection_threshold,
            )
        )

        if is_grayscale:
            image_array = convert_image_to_grayscale(image_array)

        model = None
        tile_size_str = ""
        model_file_path = None
        
        if chain is not None:
            model_file_path = chain.get("ModelFilePath", "")
            
            # --- SMART PADDING INJECTION ---
            current_target_scale = target_scale
            if current_target_scale is None:
                if target_height != 0:
                    current_target_scale = target_height / original_height
                elif target_width != 0:
                    current_target_scale = target_width / original_width
                else:
                    current_target_scale = 2.0 
            
            resize_factor_before = chain.get("ResizeFactorBeforeUpscale", 100) if chain else 100
            d_pre = max(1, round(100.0 / resize_factor_before)) if resize_factor_before > 0 else 1
            
            # Dynamically detect native model scale based on filename prefix
            model_filename = Path(model_file_path).name if model_file_path else ""
            scale_match = re.match(r'^(\d+)x', model_filename, re.IGNORECASE)
            native_scale = int(scale_match.group(1)) if scale_match else 1
            
            # --- FIX: NO DUMMY HEIGHT REQUIRED (ALGEBRAIC CANCELLATION) ---
            if target_width != 0:
                d_post = max(1, round((native_scale * original_width) / (d_pre * target_width)))
            elif current_target_scale is not None:
                d_post = max(1, round(native_scale / (d_pre * current_target_scale)))
            else:
                d_post = 1
            # ----------------------------------------------------------------------

            # --- FIX: BYPASS POINTLESS PADDING FOR WEBTOON CHUNKS ---
            if not is_webtoon_chunk:
                image_array = add_smart_padding(
                    image_array, 
                    d_pre, 
                    d_post, 
                    native_scale, 
                    force_bottom=False,
                    force_odd_w_pad=force_odd_w_pad
                )
        
            # Update dimensions to native canvas size so script logic flows perfectly
            original_height, original_width, _ = get_h_w_c(image_array)

            crop_bottom_out = 0
            if is_webtoon_chunk and unpadded_h > 0 and target_width > 0:
                padded_out_h = original_height * (target_width / original_width)
                exact_out_h = math.ceil(unpadded_h * (target_width / original_width))
                crop_bottom_out = int(round(padded_out_h - exact_out_h))
                if crop_bottom_out < 0:
                    crop_bottom_out = 0
            # -------------------------------

            resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
            resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
            resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

            # resize width and height, distorting image
            if (resize_height_before_upscale != 0 and resize_width_before_upscale != 0):
                h, w, _ = get_h_w_c(image_array)
                image_array = image_resize(
                    image_array,
                    (resize_width_before_upscale, resize_height_before_upscale),
                    is_grayscale, force_standard_resize
                )
            # resize height, keep proportional width
            elif resize_height_before_upscale != 0:
                h, w, _ = get_h_w_c(image_array)
                image_array = image_resize(
                    image_array,
                    (round(w * resize_height_before_upscale / h), resize_height_before_upscale),
                    is_grayscale, force_standard_resize
                )
            # resize width, keep proportional height
            elif resize_width_before_upscale != 0:
                h, w, _ = get_h_w_c(image_array)
                image_array = image_resize(
                    image_array,
                    (resize_width_before_upscale, round(h * resize_width_before_upscale / w)),
                    is_grayscale, force_standard_resize
                )
            elif resize_factor_before_upscale != 100:
                h, w, _ = get_h_w_c(image_array)
                # --- OVERRIDE FOR PERFECT FRACTIONS ---
                d_pre_clean = max(1, round(100.0 / resize_factor_before_upscale))
                image_array = image_resize(
                    image_array,
                    (w // d_pre_clean, h // d_pre_clean),
                    is_grayscale, force_standard_resize
                )
                # --------------------------------------
                
            # ensure the resized image dimensions are correctly updated    
            original_height, original_width, _ = get_h_w_c(image_array) 

            if is_grayscale and chain["AutoAdjustLevels"]:
                image_array = enhance_contrast(image_array)
            else:
                image_array = normalize(image_array)

            model_abs_path = get_model_abs_path(chain["ModelFilePath"])

            if model_abs_path in loaded_models:
                model = loaded_models[model_abs_path]
            elif os.path.exists(model_abs_path):
                if model_abs_path.lower().endswith(".onnx") and TensorRTUpscaler is not None:
                    print(f"Loading TensorRT model: {model_abs_path}", flush=True)
                    
                    filename_lower = model_abs_path.lower()
                    
                    if "fp16" in filename_lower:
                        use_fp16_val = True
                        use_strong_types_val = True
                        use_bf16_val = False
                    elif "fp32" in filename_lower:
                        use_fp16_val = False
                        use_strong_types_val = False
                        use_bf16_val = True
                    else:
                        use_fp16_val = False
                        use_strong_types_val = False
                        use_bf16_val = True

                    with engine_build_lock:
                        model = TensorRTUpscaler(
                            onnx_path=model_abs_path,
                            batch_size=1,
                            use_fp16=use_fp16_val,
                            use_bf16=use_bf16_val,
                            use_strong_types=use_strong_types_val,
                            device_id=settings_parser.get_int("accelerator_device_index", 0),
                            engine_cache_dir=os.path.join(os.path.dirname(model_abs_path), ".trt_cache"),
                            shape_min=(32, 32),
                            shape_opt=(512, 512),
                            shape_max=(512, 512),
                            tile_align=16,
                            builder_opt_level=3,
                            trt_workspace_gb=24
                        )
                else:
                    model, _, _ = load_model_node(context, Path(model_abs_path))
                loaded_models[model_abs_path] = model
            tile_size_str = chain["ModelTileSize"]
        else:
            image_array = normalize(image_array)
            crop_bottom_out = 0

        # image = np.ascontiguousarray(image)
        upscale_queue.put(
            (
                image_array,
                decoded_filename,
                True,
                is_grayscale,
                original_width,
                original_height,
                get_tile_size(tile_size_str),
                model,
                model_file_path,
                force_standard_resize,
                crop_bottom_out
            )
        )

    # ==========================================
    # CORE ROUTING & WEBTOON LOGIC
    # ==========================================
    archive_stem = Path(input_archive.filename).stem if hasattr(input_archive, "filename") and input_archive.filename else "Archive"
    is_webtoon = bool(re.search(r'(?i)\(webtoon[1-4]?\)', archive_stem))
    
    image_namelist = [f for f in namelist if f.lower().endswith(IMAGE_EXTENSIONS)]
    image_namelist.sort(key=natural_sort_key)
    
    if is_webtoon and len(image_namelist) > 0:
        print(f"\n[Webtoon] Detected Webtoon mode for: {archive_stem}", flush=True)
        first_w = None
        total_input_h = 0
        valid_webtoon = True
        
        # Pass 1: Validate Exact Widths & Calculate Total Height
        for filename in image_namelist:
            try:
                with input_archive.open(filename) as f:
                    vips_img = pyvips.Image.new_from_buffer(f.read(), "", access="sequential")
                    if first_w is None:
                        first_w = vips_img.width
                    elif vips_img.width != first_w:
                        print(f"[Webtoon] REJECTED: Width mismatch in '{filename}'. Expected {first_w}px, got {vips_img.width}px.", flush=True)
                        valid_webtoon = False
                        break
                    total_input_h += vips_img.height
            except Exception as e:
                print(f"[Webtoon] REJECTED: Could not read '{filename}' for dimensions: {e}", flush=True)
                valid_webtoon = False
                break
                
        if valid_webtoon and first_w is not None and total_input_h > 0:
            try:
                with input_archive.open(image_namelist[0]) as f:
                    test_img = _read_image(f.read(), image_namelist[0])
                chain, _, _, _ = get_chain_for_image(
                    test_img, target_scale, target_width, target_height, chains, grayscale_detection_threshold
                )
            except Exception as e:
                print(f"[Webtoon] Could not peek at first image for chain logic: {e}. Falling back to standard processing.", flush=True)
                valid_webtoon = False
                chain = None

            if valid_webtoon and chain is not None:
                current_target_scale = target_scale
                if current_target_scale is None:
                    if target_height != 0:
                        current_target_scale = target_height / test_img.shape[0]
                    elif target_width != 0:
                        current_target_scale = target_width / test_img.shape[1]
                    else:
                        current_target_scale = 2.0 

                resize_factor_before = chain.get("ResizeFactorBeforeUpscale", 100) if chain else 100
                d_pre = max(1, round(100.0 / resize_factor_before)) if resize_factor_before > 0 else 1
                
                model_file_path = chain.get("ModelFilePath", "")
                model_filename = Path(model_file_path).name if model_file_path else ""
                scale_match = re.match(r'^(\d+)x', model_filename, re.IGNORECASE)
                native_scale = int(scale_match.group(1)) if scale_match else 1
                
                # --- FIX: NO DUMMY HEIGHT REQUIRED ---
                if target_width != 0:
                    d_post = max(1, round((native_scale * first_w) / (d_pre * target_width)))
                elif current_target_scale is not None:
                    d_post = max(1, round(native_scale / (d_pre * current_target_scale)))
                else:
                    d_post = 1
                # ----------------------------------------------------------------------

                # --- MAJORITY VOTE FOR WIDTH PADDING ---
                pad_w = 0
                max_w_search_space = max(50, (d_pre * d_post) + 1)
                for pad in range(max_w_search_space):
                    new_w = first_w + pad
                    if new_w % d_pre == 0:
                        up_w = (new_w // d_pre) * native_scale
                        if up_w % d_post == 0:
                            pad_w = pad
                            break

                majority_pad_side = "right" # Default
                if pad_w > 0 and pad_w % 2 != 0:
                    print(f"\n[Webtoon] Odd width padding required (+{pad_w}px). Calculating majority alignment vote...", flush=True)
                    left_votes = 0
                    right_votes = 0
                    
                    def get_dominant_freq(edge_array):
                        if edge_array.ndim == 2:
                            pixels = [tuple(p) for p in edge_array]
                        else:
                            pixels = edge_array.tolist()
                        counts = Counter(pixels)
                        return counts.most_common(1)[0][1] if counts else 0
                        
                    for filename in image_namelist:
                        try:
                            with input_archive.open(filename) as f:
                                vote_img = _read_image(f.read(), filename)
                                freq_l = get_dominant_freq(vote_img[:, 0])
                                freq_r = get_dominant_freq(vote_img[:, -1])
                                if freq_l > freq_r:
                                    left_votes += 1
                                else:
                                    right_votes += 1
                        except Exception:
                            pass
                            
                    if left_votes > right_votes:
                        majority_pad_side = "left"
                        
                    print(f"[Webtoon] Vote complete -> Left: {left_votes} | Right: {right_votes}. Forcing padding to the {majority_pad_side}.", flush=True)


                # --- CALCULATE GRID STEP (REQUIRED FOR ALL MODES) ---
                from fractions import Fraction
                
                # Calculate EXACT Grid Step using fractions to guarantee zero sub-pixel seam bleeding
                if target_width != 0:
                    exact_fraction = Fraction(target_width, first_w).limit_denominator()
                    grid_step = exact_fraction.denominator
                elif target_scale is not None:
                    exact_fraction = Fraction(current_target_scale).limit_denominator(100)
                    grid_step = exact_fraction.denominator
                else:
                    grid_step = 1

                # Make sure it also satisfies the AI model's native d_pre/d_post padding rules
                while (grid_step * native_scale) % d_post != 0 or grid_step % d_pre != 0:
                    grid_step += exact_fraction.denominator if 'exact_fraction' in locals() else 1

                # --- OPTIMAL SPLICING MATHEMATICS ---
                is_exact_dimension_mode = (target_width > 0 and target_height > 0)

                if is_exact_dimension_mode:
                    # Logic for Exact Dimension Mode with Strict Forward Rolling Buffer
                    raw_ideal_h = (target_height * first_w) / target_width
                    ideal_chunk_h = math.ceil(raw_ideal_h / grid_step) * grid_step
                    if ideal_chunk_h <= 0: ideal_chunk_h = grid_step

                    num_splits = math.ceil(total_input_h / ideal_chunk_h)
                    if num_splits == 0: num_splits = 1

                    chunk_heights = []
                    for i in range(num_splits - 1):
                        chunk_heights.append(ideal_chunk_h)
                    
                    remainder_h = total_input_h - sum(chunk_heights)
                    if remainder_h > 0:
                        snapped_remainder = math.ceil(remainder_h / grid_step) * grid_step
                        if snapped_remainder == 0: snapped_remainder = grid_step
                        chunk_heights.append(snapped_remainder)
                    
                    print(f"\n[Webtoon] Exact Dimension Mode Active (Forward Rolling Buffer):", flush=True)
                    print(f"  -> Target Output: {target_width}x{target_height} (Grid Step: {grid_step}px)", flush=True)
                    print(f"  -> Calculated Ceiling Chunk Height: {ideal_chunk_h}px", flush=True)
                    if len(chunk_heights) > 1:
                        print(f"  -> Splits: {num_splits - 1} full chunks, 1 final chunk of {chunk_heights[-1]}px (padded by {chunk_heights[-1] - remainder_h}px)", flush=True)
                    else:
                        print(f"  -> Splits: 1 chunk of {chunk_heights[-1]}px (padded by {chunk_heights[-1] - remainder_h}px)", flush=True)

                else:
                    # TARGET_ASPECT_RATIO fallback
                    TARGET_ASPECT_RATIO_W = 36
                    TARGET_ASPECT_RATIO_H = 125
                    ideal_chunk_h = round((first_w * TARGET_ASPECT_RATIO_H) / TARGET_ASPECT_RATIO_W)
                    
                    num_splits = math.ceil(total_input_h / ideal_chunk_h)
                    if num_splits == 0: num_splits = 1

                    base_h = total_input_h // num_splits
                    snapped_h = (base_h // grid_step) * grid_step
                    if snapped_h == 0: snapped_h = grid_step

                    remainder = total_input_h - (num_splits * snapped_h)

                    step_additions = remainder // grid_step
                    loss_px = remainder % grid_step 

                    global_step_boost = step_additions // num_splits
                    staggered_step_boost = step_additions % num_splits

                    chunk_heights = []
                    for i in range(num_splits):
                        h = snapped_h + (global_step_boost * grid_step)
                        
                        if loss_px > 0:
                            if i == num_splits - 1:
                                h += loss_px
                            elif i >= (num_splits - 1 - staggered_step_boost):
                                h += grid_step
                        else:
                            if i >= (num_splits - staggered_step_boost):
                                h += grid_step
                                
                        chunk_heights.append(h)

                    print(f"\n[Webtoon] Optimal Math Splicing Active (Grid Step: {grid_step}px):", flush=True)
                    print(f"  -> Snapped Base: {snapped_h}px | Splits: {num_splits}", flush=True)
                    print(f"  -> Distributed {staggered_step_boost} chunks with +{grid_step}px padding-safe height.", flush=True)
                    if loss_px > 0:
                        print(f"  -> Final chunk took +{loss_px}px height. (Smart padding will equalize it by adding +{grid_step - loss_px}px to the bottom).", flush=True)

                buffer_img = None
                output_index = 0
                
                def force_c(im: np.ndarray, target_c: int) -> np.ndarray:
                    curr_c = im.shape[2] if im.ndim == 3 else 1
                    if curr_c == target_c: return im
                    if im.ndim == 2: im = np.expand_dims(im, axis=2)
                    if curr_c == 1 and target_c == 3: return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                    if curr_c == 1 and target_c == 4: return cv2.cvtColor(im, cv2.COLOR_GRAY2RGBA)
                    if curr_c == 3 and target_c == 4: return cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
                    if curr_c == 4 and target_c == 3: return cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
                    return im

                first_image_stem = Path(image_namelist[0]).stem

                def get_new_filename(base_name: str, index: int) -> str:
                    parts = re.split(r'(-| )', base_name)
                    replaced = False
                    index_str = str(index)
                    
                    for i in range(len(parts) - 1, -1, -1):
                        part = parts[i]
                        if not part or part in ('-', ' '):
                            continue
                        
                        match = re.match(r'^(\D*)(\d+)$', part)
                        if match:
                            prefix = match.group(1)
                            number_str = match.group(2)
                            pad_length = max(len(number_str), 3)
                            parts[i] = f"{prefix}{index_str.zfill(pad_length)}"
                            replaced = True
                            break
                            
                    if replaced:
                        return "".join(parts)
                        
                    fallback_base = re.sub(r'\d+$', '', base_name)
                    return f"{fallback_base}{index_str.zfill(3)}"

                # Pass 2: The Rolling Memory Buffer
                for filename in image_namelist:
                    try:
                        with input_archive.open(filename) as f:
                            # image_bytes = io.BytesIO(f.read())
                            img = _read_image(f.read(), filename)
                            
                            if buffer_img is None:
                                buffer_img = img
                            else:
                                c1 = buffer_img.shape[2] if buffer_img.ndim == 3 else 1
                                c2 = img.shape[2] if img.ndim == 3 else 1
                                max_c = max(c1, c2)
                                buffer_img = np.vstack((force_c(buffer_img, max_c), force_c(img, max_c)))
                                
                            while output_index < len(chunk_heights):
                                if buffer_img is None:
                                    break
                                    
                                target_h = chunk_heights[output_index]
                                if buffer_img.shape[0] < target_h:
                                    break
                                    
                                chunk = buffer_img[:target_h, :, :]
                                if buffer_img.shape[0] == target_h:
                                    buffer_img = None
                                else:
                                    buffer_img = buffer_img[target_h:, :, :]
                                    
                                new_stem = get_new_filename(first_image_stem, output_index + 1)
                                chunk_filename = f"{new_stem}.png"
                                
                                print(f"[Webtoon] Spliced Chunk {output_index + 1}/{num_splits} -> {chunk_filename}", flush=True)
                                
                                process_and_queue_image(
                                    chunk, 
                                    chunk_filename, 
                                    is_webtoon_chunk=True, 
                                    force_odd_w_pad=majority_pad_side
                                )
                                output_index += 1
                                
                    except Exception as e:
                        print(f"[Webtoon] ERROR processing '{filename}': {e}", flush=True)

                # NEW: Handle the final padded chunk for Exact Dimension Mode (Minimal Padding)
                if is_exact_dimension_mode and output_index < len(chunk_heights) and buffer_img is not None and buffer_img.shape[0] > 0:
                    actual_h = buffer_img.shape[0]
                    remainder = actual_h % grid_step
                    
                    target_h = actual_h + (grid_step - remainder) if remainder != 0 else actual_h
                    missing_h = target_h - actual_h
                    
                    if missing_h > 0:
                        buffer_img = cv2.copyMakeBorder(buffer_img, 0, missing_h, 0, 0, cv2.BORDER_REPLICATE)
                        
                    chunk = buffer_img
                    new_stem = get_new_filename(first_image_stem, output_index + 1)
                    chunk_filename = f"{new_stem}.png"
                    
                    process_and_queue_image(
                        chunk, 
                        chunk_filename, 
                        is_webtoon_chunk=True, 
                        force_odd_w_pad=majority_pad_side,
                        unpadded_h=actual_h
                    )
                    output_index += 1
                        
                # Ensure non-image assets (txt, etc) from the Webtoon still make it into the final output ZIP
                non_images = [f for f in namelist if not f.lower().endswith(IMAGE_EXTENSIONS)]
                for filename in non_images:
                    try:
                        decoded_filename = filename.encode("cp437").decode(f"cp{system_codepage}")
                    except:
                        decoded_filename = filename
                    try:
                        with input_archive.open(filename) as f:
                            upscale_queue.put((f.read(), decoded_filename, False, False, None, None, None, None, None, force_standard_resize, 0))
                    except Exception:
                        pass
                        
                upscale_queue.put(UPSCALE_SENTINEL)
                # print("preprocess_worker_archive exiting")
                return # Webtoon processing successfully completed
            else:
                print("[Webtoon] Calculation/Validation failed. Falling back to standard CBZ processing.", flush=True)
            
    # ==========================================
    # STANDARD FALLBACK / NORMAL PROCESSING
    # ==========================================
    for filename in namelist:
        decoded_filename = filename
        image_data = None
        try:
            decoded_filename = decoded_filename.encode("cp437").decode(f"cp{system_codepage}")
        except:  # noqa: E722
            pass

        try:
            with input_archive.open(filename) as file_in_archive:
                image_data = file_in_archive.read()
                
                # image_bytes = io.BytesIO(image_data)
                image = _read_image(image_data, filename)
                print("read image", filename, flush=True)
                
                process_and_queue_image(image, decoded_filename, is_webtoon_chunk=False)
                
        except Exception as e:
            # Matches original logic perfectly: catches non-images and copies them over cleanly.
            print(f"could not read as image, copying file to zip instead of upscaling: {decoded_filename}, {e}", flush=True)
            upscale_queue.put((image_data, decoded_filename, False, False, None, None, None, None, None, force_standard_resize, 0))
        #     pass
            
    upscale_queue.put(UPSCALE_SENTINEL)
    # print("preprocess_worker_archive exiting")


def preprocess_worker_folder(
    upscale_queue: Queue,
    input_folder_path: str,
    output_folder_path: str,
    output_filename: str,
    upscale_images: bool,
    upscale_archives: bool,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given a folder path, recursively iterate the folder
    """
    print(
        f"preprocess_worker_folder entering {input_folder_path} {output_folder_path} {output_filename}",
        flush=True,
    )
    for root, _dirs, files in os.walk(input_folder_path):
        for filename in files:
            # for output file, create dirs if necessary, or skip if file exists and overwrite not enabled
            input_file_base = Path(filename).stem
            filename_rel = os.path.relpath(
                os.path.join(root, filename), input_folder_path
            )
            output_filename_rel = os.path.join(
                os.path.dirname(filename_rel),
                output_filename.replace("%filename%", input_file_base),
            )
            output_file_path = Path(
                os.path.join(output_folder_path, output_filename_rel)
            )

            if filename.lower().endswith(IMAGE_EXTENSIONS):  # TODO if image
                force_standard_resize = get_force_standard_resize(input_file_base)

                if upscale_images:
                    output_file_path = str(
                        Path(f"{output_file_path}.{image_format}")
                    ).replace("%filename%", input_file_base)

                    if not overwrite_existing_files and os.path.isfile(
                        output_file_path
                    ):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue

                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    image = _read_image_from_path(os.path.join(root, filename))

                    chain, is_grayscale, original_width, original_height = (
                        get_chain_for_image(
                            image,
                            target_scale,
                            target_width,
                            target_height,
                            chains,
                            grayscale_detection_threshold,
                        )
                    )

                    if is_grayscale:
                        image = convert_image_to_grayscale(image)

                    model = None
                    tile_size_str = ""
                    model_file_path = None
                    if chain is not None:
                        model_file_path = chain.get("ModelFilePath", "")
                        
                        # --- SMART PADDING INJECTION ---
                        current_target_scale = target_scale
                        if current_target_scale is None:
                            if target_height != 0:
                                current_target_scale = target_height / original_height
                            elif target_width != 0:
                                current_target_scale = target_width / original_width
                            else:
                                current_target_scale = 2.0 
                        
                        resize_factor_before = chain.get("ResizeFactorBeforeUpscale", 100) if chain else 100
                        d_pre = max(1, round(100.0 / resize_factor_before)) if resize_factor_before > 0 else 1
                        
                        # Dynamically detect native model scale based on filename prefix
                        model_filename = Path(model_file_path).name if model_file_path else ""
                        scale_match = re.match(r'^(\d+)x', model_filename, re.IGNORECASE)
                        if scale_match:
                            native_model_scale = int(scale_match.group(1))
                            print(f"[Smart Padding] Detected native model scale {native_model_scale}x from filename: '{model_filename}'", flush=True)
                        else:
                            native_model_scale = 1
                            print(f"[Smart Padding] Could not detect native scale from filename '{model_filename}'. Defaulting to {native_model_scale}x.", flush=True)
                            
                        # --- FIX: NO DUMMY HEIGHT REQUIRED (ALGEBRAIC CANCELLATION) ---
                        if target_width != 0:
                            d_post = max(1, round((native_model_scale * original_width) / (d_pre * target_width)))
                        elif current_target_scale is not None:
                            d_post = max(1, round(native_model_scale / (d_pre * current_target_scale)))
                        else:
                            d_post = 1
                        # ----------------------------------------------------------------------

                        is_webtoon = bool(re.search(r'(?i)\(webtoon[1-4]?\)', input_file_base))
                        # --- FIX: BYPASS POINTLESS PADDING FOR WEBTOON CHUNKS ---
                        if not is_webtoon:
                            image = add_smart_padding(image, d_pre, d_post, native_model_scale)

                        original_height, original_width, _ = get_h_w_c(image)
                        # -------------------------------

                        resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
                        resize_height_before_upscale = chain[
                            "ResizeHeightBeforeUpscale"
                        ]
                        resize_factor_before_upscale = chain[
                            "ResizeFactorBeforeUpscale"
                        ]

                        # resize width and height, distorting image
                        if (
                            resize_height_before_upscale != 0
                            and resize_width_before_upscale != 0
                        ):
                            h, w, _ = get_h_w_c(image)
                            image = image_resize(
                                image,
                                (
                                    resize_width_before_upscale,
                                    resize_height_before_upscale,
                                ),
                                is_grayscale, force_standard_resize
                            )
                        # resize height, keep proportional width
                        elif resize_height_before_upscale != 0:
                            h, w, _ = get_h_w_c(image)
                            image = image_resize(
                                image,
                                (
                                    round(w * resize_height_before_upscale / h),
                                    resize_height_before_upscale,
                                ),
                                is_grayscale, force_standard_resize
                            )
                        # resize width, keep proportional height
                        elif resize_width_before_upscale != 0:
                            h, w, _ = get_h_w_c(image)
                            image = image_resize(
                                image,
                                (
                                    resize_width_before_upscale,
                                    round(h * resize_width_before_upscale / w),
                                ),
                                is_grayscale, force_standard_resize
                            )
                        elif resize_factor_before_upscale != 100:
                            h, w, _ = get_h_w_c(image)
                            # --- OVERRIDE FOR PERFECT FRACTIONS ---
                            d_pre_clean = max(1, round(100.0 / resize_factor_before_upscale))
                            image = image_resize(
                                image,
                                (
                                    w // d_pre_clean,
                                    h // d_pre_clean,
                                ),
                                is_grayscale, force_standard_resize
                            )
                            # --------------------------------------
                            
                        # ensure the resized image dimensions are correctly updated    
                        original_height, original_width, _ = get_h_w_c(image) 

                        if is_grayscale and chain["AutoAdjustLevels"]:
                            image = enhance_contrast(image)
                        else:
                            image = normalize(image)

                        model_abs_path = get_model_abs_path(chain["ModelFilePath"])

                        if model_abs_path in loaded_models:
                            model = loaded_models[model_abs_path]

                        elif os.path.exists(model_abs_path):
                            if model_abs_path.lower().endswith(".onnx") and TensorRTUpscaler is not None:
                                print(f"Loading TensorRT model: {model_abs_path}", flush=True)
                                
                                filename_lower = model_abs_path.lower()
                                
                                if "fp16" in filename_lower:
                                    use_fp16_val = True
                                    use_strong_types_val = True
                                    use_bf16_val = False
                                elif "fp32" in filename_lower:
                                    use_fp16_val = False
                                    use_strong_types_val = False
                                    use_bf16_val = True
                                else:
                                    use_fp16_val = False
                                    use_strong_types_val = False
                                    use_bf16_val = True

                                with engine_build_lock:
                                    model = TensorRTUpscaler(
                                        onnx_path=model_abs_path,
                                        batch_size=1,
                                        use_fp16=use_fp16_val,
                                        use_bf16=use_bf16_val,
                                        use_strong_types=use_strong_types_val,
                                        device_id=settings_parser.get_int("accelerator_device_index", 0),
                                        engine_cache_dir=os.path.join(os.path.dirname(model_abs_path), ".trt_cache"),
                                        shape_min=(32, 32),
                                        shape_opt=(512, 512),
                                        shape_max=(512, 512),
                                        tile_align=16,
                                        builder_opt_level=3,
                                        trt_workspace_gb=24
                                    )
                            else:
                                model, _, _ = load_model_node(context, Path(model_abs_path))
                            loaded_models[model_abs_path] = model
                        tile_size_str = chain["ModelTileSize"]
                    else:
                        image = normalize(image)
                   
                    # image = np.ascontiguousarray(image)

                    upscale_queue.put(
                        (
                            image,
                            output_filename_rel,
                            True,
                            is_grayscale,
                            original_width,
                            original_height,
                            get_tile_size(tile_size_str),
                            model,
                            model_file_path,
                            force_standard_resize,
                            0
                        )
                    )
            elif filename.lower().endswith(ARCHIVE_EXTENSIONS):
                if upscale_archives:
                    output_file_path = f"{output_file_path}.cbz"
                    if not overwrite_existing_files and os.path.isfile(
                        output_file_path
                    ):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                    upscale_archive_file(
                        os.path.join(root, filename),
                        output_file_path,
                        image_format,
                        lossy_compression_quality,
                        use_lossless_compression,
                        target_scale,
                        target_width,
                        target_height,
                        chains,
                        loaded_models,
                        grayscale_detection_threshold,
                    )  # TODO custom output extension
    upscale_queue.put(UPSCALE_SENTINEL)
    # print("preprocess_worker_folder exiting")


def preprocess_worker_image(
    upscale_queue: Queue,
    input_image_path: str,
    output_image_path: str,
    overwrite_existing_files: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given an image path, apply auto levels and add to upscale queue
    """
    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        if not overwrite_existing_files and os.path.isfile(output_image_path):
            print(f"file exists, skip: {output_image_path}", flush=True)
            return
            
        force_standard_resize = get_force_standard_resize(Path(input_image_path).stem)

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # with Image.open(input_image_path) as img:
        image = _read_image_from_path(input_image_path)

        chain, is_grayscale, original_width, original_height = get_chain_for_image(
            image,
            target_scale,
            target_width,
            target_height,
            chains,
            grayscale_detection_threshold,
        )

        if is_grayscale:
            image = convert_image_to_grayscale(image)

        model = None
        tile_size_str = ""
        model_file_path = None
        if chain is not None:
            model_file_path = chain.get("ModelFilePath", "")
            
            # --- SMART PADDING INJECTION ---
            current_target_scale = target_scale
            if current_target_scale is None:
                if target_height != 0:
                    current_target_scale = target_height / original_height
                elif target_width != 0:
                    current_target_scale = target_width / original_width
                else:
                    current_target_scale = 2.0 
            
            resize_factor_before = chain.get("ResizeFactorBeforeUpscale", 100) if chain else 100
            d_pre = max(1, round(100.0 / resize_factor_before)) if resize_factor_before > 0 else 1
            
            # Dynamically detect native model scale based on filename prefix
            model_filename = Path(model_file_path).name if model_file_path else ""
            scale_match = re.match(r'^(\d+)x', model_filename, re.IGNORECASE)
            if scale_match:
                native_model_scale = int(scale_match.group(1))
                print(f"[Smart Padding] Detected native model scale {native_model_scale}x from filename: '{model_filename}'", flush=True)
            else:
                native_model_scale = 1
                print(f"[Smart Padding] Could not detect native scale from filename '{model_filename}'. Defaulting to {native_model_scale}x.", flush=True)
                
            # --- FIX: NO DUMMY HEIGHT REQUIRED (ALGEBRAIC CANCELLATION) ---
            if target_width != 0:
                d_post = max(1, round((native_model_scale * original_width) / (d_pre * target_width)))
            elif current_target_scale is not None:
                d_post = max(1, round(native_model_scale / (d_pre * current_target_scale)))
            else:
                d_post = 1
            # ----------------------------------------------------------------------

            is_webtoon = bool(re.search(r'(?i)\(webtoon[1-4]?\)', Path(input_image_path).stem))
            # --- FIX: BYPASS POINTLESS PADDING FOR WEBTOON CHUNKS ---
            if not is_webtoon:
                image = add_smart_padding(
                    image, 
                    d_pre, 
                    d_post, 
                    native_model_scale, 
                    force_bottom=False,
                    force_odd_w_pad=None
                )
                
            # Update dimensions to native canvas size so script logic flows perfectly
            original_height, original_width, _ = get_h_w_c(image)
            # -------------------------------

            resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
            resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
            resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

            # resize width and height, distorting image
            if resize_height_before_upscale != 0 and resize_width_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = image_resize(
                    image, (resize_width_before_upscale, resize_height_before_upscale),
                    is_grayscale, force_standard_resize
                )
            # resize height, keep proportional width
            elif resize_height_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = image_resize(
                    image,
                    (
                        round(w * resize_height_before_upscale / h),
                        resize_height_before_upscale,
                    ),
                    is_grayscale, force_standard_resize
                )
            # resize width, keep proportional height
            elif resize_width_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = image_resize(
                    image,
                    (
                        resize_width_before_upscale,
                        round(h * resize_width_before_upscale / w),
                    ),
                    is_grayscale, force_standard_resize
                )
            elif resize_factor_before_upscale != 100:
                h, w, _ = get_h_w_c(image)
                # --- OVERRIDE FOR PERFECT FRACTIONS ---
                d_pre_clean = max(1, round(100.0 / resize_factor_before_upscale))
                image = image_resize(
                    image,
                    (
                        w // d_pre_clean,
                        h // d_pre_clean,
                    ),
                    is_grayscale, force_standard_resize
                )
                # --------------------------------------
                
            # ensure the resized image dimensions are correctly updated    
            original_height, original_width, _ = get_h_w_c(image) 

            if is_grayscale and chain["AutoAdjustLevels"]:
                image = enhance_contrast(image)
            else:
                image = normalize(image)

            if chain["ModelFilePath"] == "No Model":
                pass
            else:
                model_abs_path = get_model_abs_path(chain["ModelFilePath"])

                if not os.path.exists(model_abs_path):
                    raise FileNotFoundError(model_abs_path)

                if model_abs_path in loaded_models:
                    model = loaded_models[model_abs_path]

                elif os.path.exists(model_abs_path):
                    if model_abs_path.lower().endswith(".onnx") and TensorRTUpscaler is not None:
                        print(f"Loading TensorRT model: {model_abs_path}", flush=True)
                        
                        filename_lower = model_abs_path.lower()
                        
                        if "fp16" in filename_lower:
                            use_fp16_val = True
                            use_strong_types_val = True
                            use_bf16_val = False
                        elif "fp32" in filename_lower:
                            use_fp16_val = False
                            use_strong_types_val = False
                            use_bf16_val = True
                        else:
                            use_fp16_val = False
                            use_strong_types_val = False
                            use_bf16_val = True

                        with engine_build_lock:
                            model = TensorRTUpscaler(
                                onnx_path=model_abs_path,
                                batch_size=1,
                                use_fp16=use_fp16_val,
                                use_bf16=use_bf16_val,
                                use_strong_types=use_strong_types_val,
                                device_id=settings_parser.get_int("accelerator_device_index", 0),
                                engine_cache_dir=os.path.join(os.path.dirname(model_abs_path), ".trt_cache"),
                                shape_min=(32, 32),
                                shape_opt=(512, 512),
                                shape_max=(512, 512),
                                tile_align=16,
                                builder_opt_level=3,
                                trt_workspace_gb=24
                            )
                    else:
                        model, _, _ = load_model_node(context, Path(model_abs_path))
                    loaded_models[model_abs_path] = model
                tile_size_str = chain["ModelTileSize"]
        else:
            print("No chain!!!!!!!")
            image = normalize(image)

        # image = np.ascontiguousarray(image)
        
        upscale_queue.put(
            (
                image,
                None,
                True,
                is_grayscale,
                original_width,
                original_height,
                get_tile_size(tile_size_str),
                model,
                model_file_path,
                force_standard_resize,
                0
            )
        )
    upscale_queue.put(UPSCALE_SENTINEL)


def upscale_worker(upscale_queue: Queue, postprocess_queue: Queue) -> None:
    """
    wait for upscale queue, for each queue entry, upscale image and add result to postprocess queue
    """
    # print("upscale_worker entering")
    while True:
        (
            image,
            file_name,
            is_image,
            is_grayscale,
            original_width,
            original_height,
            model_tile_size,
            model,
            model_file_path,
            force_standard_resize,
            crop_bottom_out
        ) = upscale_queue.get()
        if image is None:
            break

        if is_image:
            # Save original image (pre-upscale) to be used as color reference
            original_for_fix = image

            image = ai_upscale_image(image, model_tile_size, model)

            # Apply Wavelet Color Fix if image is NOT grayscale
            if not is_grayscale:
                if model_file_path and "scunet_color_real_psnr.pth" in model_file_path:
                    print(f"Wavelet Color Fix disabled for model: {model_file_path}", flush=True)
                else:
                    try:
                        # You can adjust levels (e.g., 5) as needed
                        image = apply_wavelet_color_fix(image, original_for_fix, levels=5)
                        print("Applied Wavelet Color Fix", flush=True)
                    except Exception as e:
                        error_msg = f"Failed to apply Wavelet Color Fix: {e}"
                        print(error_msg, flush=True)
                        # HARD LOGGING: Write to error.log on the device so you don't miss it
                        try:
                            import time
                            with open("/workspace/error.log", "a", encoding="utf-8") as err_file:
                                err_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
                        except Exception as log_e:
                            print(f"Could not write to error.log: {log_e}", flush=True)

            # convert back to grayscale
            if is_grayscale:
                image = convert_image_to_grayscale(image)

        postprocess_queue.put(
            (image, file_name, is_image, is_grayscale, original_width, original_height, force_standard_resize, crop_bottom_out)
        )
    postprocess_queue.put(POSTPROCESS_SENTINEL)
    # print("upscale_worker exiting")


def postprocess_worker_zip(
    postprocess_queue: Queue,
    output_zip_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    """
    wait for postprocess queue, for each queue entry, save the image to the zip file
    """
    is_webtoon = bool(re.search(r'(?i)\(webtoon[1-4]?\)', Path(output_zip_path).stem))

    def force_c(im: np.ndarray, target_c: int) -> np.ndarray:
        curr_c = im.shape[2] if im.ndim == 3 else 1
        if curr_c == target_c: return im
        if im.ndim == 2: im = np.expand_dims(im, axis=2)
        if curr_c == 1 and target_c == 3: return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        if curr_c == 1 and target_c == 4: return cv2.cvtColor(im, cv2.COLOR_GRAY2RGBA)
        if curr_c == 3 and target_c == 4: return cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
        if curr_c == 4 and target_c == 3: return cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        return im

    def get_new_filename(base_name: str, index: int) -> str:
        parts = re.split(r'(-| )', base_name)
        replaced = False
        index_str = str(index)
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            if not part or part in ('-', ' '): continue
            match = re.match(r'^(\D*)(\d+)$', part)
            if match:
                prefix = match.group(1)
                number_str = match.group(2)
                pad_length = max(len(number_str), 3)
                parts[i] = f"{prefix}{index_str.zfill(pad_length)}"
                replaced = True
                break
        if replaced: return "".join(parts)
        return f"{re.sub(r'\d+$', '', base_name)}{index_str.zfill(3)}"

    with ZipFile(output_zip_path, "w", ZIP_DEFLATED) as output_zip:
        rolling_buffer = None
        output_index = 1
        base_stem = None
        
        while True:
            data = postprocess_queue.get()
            
            # FIX: Check if the first element (image) is None to avoid NumPy collision
            if data[0] is None:
                break
                
            (image, file_name, is_image, is_grayscale, original_width, original_height, force_standard_resize, crop_bottom_out) = data
            
            if not is_image:
                output_zip.writestr(file_name, image)
                continue

            if base_stem is None:
                base_stem = Path(file_name).stem

            # 1. Resize to Perfect Aspect Ratio (e.g., 1440x4005)
            # Webtoon logic in final_target_resize guarantees proportional height scaling
            image = final_target_resize(
                image, target_scale, target_width, target_height,
                original_width, original_height, is_grayscale, force_standard_resize, is_webtoon
            )
            
            # Cleanly slice off ONLY the padding added to fix the upscale fraction
            if is_webtoon and crop_bottom_out > 0:
                image = image[:-crop_bottom_out, :, :]

            # 2. Attach any leftover pixels from the previous chunk
            if is_webtoon and rolling_buffer is not None:
                max_c = max(rolling_buffer.shape[2] if rolling_buffer.ndim == 3 else 1, image.shape[2] if image.ndim == 3 else 1)
                rolling_buffer = force_c(rolling_buffer, max_c)
                image = force_c(image, max_c)
                image = np.vstack((rolling_buffer, image))

            # 3. Slice perfect 4000px chunks
            if is_webtoon:
                slice_h = target_height if target_height > 0 else 4000
                while image.shape[0] >= slice_h:
                    chunk = image[:slice_h, :, :]
                    image = image[slice_h:, :, :]

                    new_stem = get_new_filename(base_stem, output_index)
                    new_file_name = str(Path(file_name).with_stem(new_stem).with_suffix(f".{image_format}"))
                    
                    actual_h, actual_w, _ = get_h_w_c(chunk)
                    save_image_zip(
                        chunk, new_file_name, output_zip, image_format, 
                        lossy_compression_quality, use_lossless_compression, 
                        actual_w, actual_h, target_scale, actual_w, actual_h, 
                        is_grayscale, force_standard_resize, False
                    )
                    output_index += 1
                
                rolling_buffer = image if image.shape[0] > 0 else None
            else:
                # Standard save for non-webtoons
                save_image_zip(
                    image, str(Path(file_name).with_suffix(f".{image_format}")), 
                    output_zip, image_format, lossy_compression_quality, 
                    use_lossless_compression, original_width, original_height, 
                    target_scale, target_width, target_height, is_grayscale, 
                    force_standard_resize, False
                )
            print("PROGRESS=postprocess_worker_zip_image", flush=True)

        # 4. Save any remaining pixels as the final page
        if is_webtoon and rolling_buffer is not None:
            new_stem = get_new_filename(base_stem, output_index)
            new_file_name = str(Path(base_stem).with_stem(new_stem).with_suffix(f".{image_format}"))
            
            actual_h, actual_w, _ = get_h_w_c(rolling_buffer)
            save_image_zip(
                rolling_buffer, new_file_name, output_zip, image_format, 
                lossy_compression_quality, use_lossless_compression, 
                actual_w, actual_h, target_scale, actual_w, actual_h, 
                is_grayscale, force_standard_resize, False
            )

        print("PROGRESS=postprocess_worker_zip_archive", flush=True)


def postprocess_worker_folder(
    postprocess_queue: Queue,
    output_folder_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    """
    wait for postprocess queue, for each queue entry, save the image to the output folder
    """
    # Detect if the destination folder files should be webtoon
    # We will rely on the incoming file names to toggle rolling buffer per file sequence.
    # However, since folder mode can process multiple disparate files, we maintain
    # the rolling buffer only if the current file's stem indicates it's a webtoon.

    def force_c(im: np.ndarray, target_c: int) -> np.ndarray:
        curr_c = im.shape[2] if im.ndim == 3 else 1
        if curr_c == target_c: return im
        if im.ndim == 2: im = np.expand_dims(im, axis=2)
        if curr_c == 1 and target_c == 3: return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        if curr_c == 1 and target_c == 4: return cv2.cvtColor(im, cv2.COLOR_GRAY2RGBA)
        if curr_c == 3 and target_c == 4: return cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
        if curr_c == 4 and target_c == 3: return cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        return im

    def get_new_filename(base_name: str, index: int) -> str:
        parts = re.split(r'(-| )', base_name)
        replaced = False
        index_str = str(index)
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            if not part or part in ('-', ' '): continue
            match = re.match(r'^(\D*)(\d+)$', part)
            if match:
                prefix = match.group(1)
                number_str = match.group(2)
                pad_length = max(len(number_str), 3)
                parts[i] = f"{prefix}{index_str.zfill(pad_length)}"
                replaced = True
                break
        if replaced: return "".join(parts)
        return f"{re.sub(r'\d+$', '', base_name)}{index_str.zfill(3)}"

    rolling_buffer = None
    output_index = 1
    base_stem = None
    last_file_name_rel = None

    while True:
        data = postprocess_queue.get()
        
        # FIX: Check if the first element (image) is None
        if data[0] is None:
            break
            
        (image, file_name_rel, is_image, is_grayscale, original_width, original_height, force_standard_resize, crop_bottom_out) = data
        
        if not is_image:
            continue

        is_webtoon = bool(re.search(r'(?i)\(webtoon[1-4]?\)', str(file_name_rel)))

        # 1. Resize to Perfect Aspect Ratio
        image = final_target_resize(
            image, target_scale, target_width, target_height,
            original_width, original_height, is_grayscale, force_standard_resize, is_webtoon
        )
        
        # Cleanly slice off ONLY the padding added to fix the upscale fraction
        if is_webtoon and crop_bottom_out > 0:
            image = image[:-crop_bottom_out, :, :]

        if is_webtoon:
            if base_stem is None:
                base_stem = Path(file_name_rel).stem

            # 2. Attach any leftover pixels
            if rolling_buffer is not None:
                max_c = max(rolling_buffer.shape[2] if rolling_buffer.ndim == 3 else 1, image.shape[2] if image.ndim == 3 else 1)
                rolling_buffer = force_c(rolling_buffer, max_c)
                image = force_c(image, max_c)
                image = np.vstack((rolling_buffer, image))

            # 3. Slice perfect chunks
            slice_h = target_height if target_height > 0 else 4000
            while image.shape[0] >= slice_h:
                chunk = image[:slice_h, :, :]
                image = image[slice_h:, :, :]

                new_stem = get_new_filename(base_stem, output_index)
                new_file_name = str(Path(file_name_rel).with_stem(new_stem).with_suffix(f".{image_format}"))
                output_path = os.path.join(output_folder_path, new_file_name)
                
                actual_h, actual_w, _ = get_h_w_c(chunk)
                save_image(
                    chunk, output_path, image_format, lossy_compression_quality, 
                    use_lossless_compression, actual_w, actual_h, target_scale, 
                    actual_w, actual_h, is_grayscale, force_standard_resize, False
                )
                output_index += 1
            
            rolling_buffer = image if image.shape[0] > 0 else None
            last_file_name_rel = file_name_rel
        else:
            output_path = os.path.join(output_folder_path, str(Path(f"{file_name_rel}.{image_format}")))
            save_image(
                image, output_path, image_format, lossy_compression_quality, 
                use_lossless_compression, original_width, original_height, 
                target_scale, target_width, target_height, is_grayscale, 
                force_standard_resize, False
            )

        print("PROGRESS=postprocess_worker_folder", flush=True)

    # 4. Save any remaining pixels as the final page
    if rolling_buffer is not None and last_file_name_rel is not None:
        new_stem = get_new_filename(base_stem, output_index)
        new_file_name = str(Path(last_file_name_rel).with_stem(new_stem).with_suffix(f".{image_format}"))
        output_path = os.path.join(output_folder_path, new_file_name)
        
        actual_h, actual_w, _ = get_h_w_c(rolling_buffer)
        save_image(
            rolling_buffer, output_path, image_format, lossy_compression_quality, 
            use_lossless_compression, actual_w, actual_h, target_scale, 
            actual_w, actual_h, is_grayscale, force_standard_resize, False
        )


def postprocess_worker_image(
    postprocess_queue: Queue,
    output_file_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    """
    wait for postprocess queue, for each queue entry, save the image to the output file path
    """
    # Detect if the explicit image output path is a webtoon
    is_webtoon = bool(re.search(r'(?i)\(webtoon[1-4]?\)', Path(output_file_path).stem))
    
    while True:
        data = postprocess_queue.get()
        
        # FIX: Check if the first element (image) is None
        if data[0] is None:
            break
            
        (
            image, 
            _, 
            _, 
            is_grayscale, 
            original_width, 
            original_height, 
            force_standard_resize,
            crop_bottom_out
        ) = data
        
        # For a single image pass, we can still crop if needed
        if is_webtoon and crop_bottom_out > 0:
            image = image[:-crop_bottom_out, :, :]

        save_image(
            image,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            original_width,
            original_height,
            target_scale,
            target_width,
            target_height,
            is_grayscale,
            force_standard_resize,
            is_webtoon=is_webtoon,
        )
        print("PROGRESS=postprocess_worker_image", flush=True)


def upscale_archive_file(
    input_zip_path: str,
    output_zip_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    # TODO accept multiple paths to reuse simple queues?
    
    input_name = Path(input_zip_path).stem.lower()
    force_standard_resize = get_force_standard_resize(input_name)

    upscale_queue = Queue(maxsize=1)
    postprocess_queue = MPQueue(maxsize=1)

    # start preprocess zip process
    preprocess_process = Thread(
        target=preprocess_worker_archive,
        args=(
            upscale_queue,
            input_zip_path,
            output_zip_path,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
            force_standard_resize,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess zip process
    postprocess_process = Process(
        target=postprocess_worker_zip,
        args=(
            postprocess_queue,
            output_zip_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_image_file(
    input_image_path: str,
    output_image_path: str,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = MPQueue(maxsize=1)

    # start preprocess image process
    preprocess_process = Thread(
        target=preprocess_worker_image,
        args=(
            upscale_queue,
            input_image_path,
            output_image_path,
            overwrite_existing_files,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess image process
    postprocess_process = Process(
        target=postprocess_worker_image,
        args=(
            postprocess_queue,
            output_image_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_file(
    input_file_path: str,
    output_folder_path: str,
    output_filename: str,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    input_file_base = Path(input_file_path).stem

    if input_file_path.lower().endswith(ARCHIVE_EXTENSIONS):
        output_file_path = str(
            Path(
                f"{os.path.join(output_folder_path,output_filename.replace('%filename%', input_file_base))}.cbz"
            )
        )
        print("output_file_path", output_file_path, flush=True)
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_archive_file(
            input_file_path,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        )

    elif input_file_path.lower().endswith(IMAGE_EXTENSIONS):
        output_file_path = str(
            Path(
                f"{os.path.join(output_folder_path,output_filename.replace('%filename%', input_file_base))}.{image_format}"
            )
        )
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_image_file(
            input_file_path,
            output_file_path,
            overwrite_existing_files,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        )


def upscale_folder(
    input_folder_path: str,
    output_folder_path: str,
    output_filename: str,
    upscale_images: bool,
    upscale_archives: bool,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    # print("upscale_folder: entering")

    # preprocess_queue = Queue(maxsize=1)
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = MPQueue(maxsize=1)

    # start preprocess folder process
    preprocess_process = Thread(
        target=preprocess_worker_folder,
        args=(
            upscale_queue,
            input_folder_path,
            output_folder_path,
            output_filename,
            upscale_images,
            upscale_archives,
            overwrite_existing_files,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess folder process
    postprocess_process = Process(
        target=postprocess_worker_folder,
        args=(
            postprocess_queue,
            output_folder_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


current_file_directory = os.path.dirname(os.path.abspath(__file__))


def get_model_abs_path(chain_model_file_path: str) -> str:
    return os.path.abspath(os.path.join(models_directory, chain_model_file_path))


def get_gamma_icc_profile() -> ImageCmsProfile:
    profile_path = os.path.join(
        current_file_directory, "../ImageMagick/Custom Gray Gamma 1.0.icc"
    )
    return ImageCms.getOpenProfile(profile_path)


def get_dot20_icc_profile() -> ImageCmsProfile:
    profile_path = os.path.join(
        current_file_directory, "../ImageMagick/Dot Gain 20%.icc"
    )
    return ImageCms.getOpenProfile(profile_path)


def parse_settings_from_cli():
    parser = argparse.ArgumentParser(prog="python run_upscale.py",
                                     description="By default, used by MangaJaNaiConverterGui as an internal tool. "
                                                 "Alternative options made available to make it easier to skip the GUI "
                                                 "and run upscaling jobs directly from CLI.")

    execution_type_group = parser.add_mutually_exclusive_group(required=True)
    execution_type_group.add_argument("--settings",
                                      help="Default behaviour, based on provided appstate configuration. "
                                           "For advanced usage.")
    execution_type_group.add_argument("-f", "--file-path",
                                      help="Upscale single file")
    execution_type_group.add_argument("-d", "--folder-path",
                                      help="Upscale whole directory")

    parser.add_argument("-o", "--output-folder-path",
                        default=os.path.join(".", "out"),
                        help="Output directory for upscaled files. Default: ./out")
    parser.add_argument("-m", "--models-directory-path",
                        default=os.path.join("..", "models"),
                        help="Directory with models used for upscaling. "
                             "Supports only models bundled with MangaJaNaiConvertedGui. "
                             "Default: MangaJaNaiConverterGui/chaiNNer/models/")
    parser.add_argument("-u", "--upscale-factor",
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=2,
                        help="Used for calculating which model will be used. Default: 2")
    parser.add_argument("--device-index",
                        type=int,
                        default=0,
                        help="Device used to run upscaling jobs in case more than one is available. Default: 0")

    args = parser.parse_args()

    return parse_auto_settings(args) if args.settings else parse_manual_settings(args)


def parse_auto_settings(args):
    with open(args.settings, encoding="utf-8") as f:
        json_settings = json.load(f)

    return json_settings


def parse_manual_settings(args):
    default_file_path = os.path.join("..", "resources", "default_cli_configuration.json")
    with open(default_file_path, "r") as default_file:
        default_json = json.load(default_file)

    default_json["SelectedDeviceIndex"] = int(args.device_index)
    default_json["ModelsDirectory"] = args.models_directory_path

    default_json["Workflows"]["$values"][0]["OutputFolderPath"] = args.output_folder_path
    default_json["Workflows"]["$values"][0]["SelectedDeviceIndex"] = args.device_index
    default_json["Workflows"]["$values"][0]["UpscaleScaleFactor"] = args.upscale_factor
    if args.file_path:
        default_json["Workflows"]["$values"][0]["SelectedTabIndex"] = 0
        default_json["Workflows"]["$values"][0]["InputFilePath"] = args.file_path
    elif args.folder_path:
        default_json["Workflows"]["$values"][0]["SelectedTabIndex"] = 1
        default_json["Workflows"]["$values"][0]["InputFolderPath"] = args.folder_path

    return default_json


is_windows = platform.system() == "win32"
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

settings = parse_settings_from_cli()

workflow = settings["Workflows"]["$values"][settings["SelectedWorkflowIndex"]]
models_directory = settings["ModelsDirectory"]

UPSCALE_SENTINEL = (None,) * 11
POSTPROCESS_SENTINEL = (None,) * 8
CV2_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_EXTENSIONS = (*CV2_IMAGE_EXTENSIONS, ".avif")
ZIP_EXTENSIONS = (".zip", ".cbz")
RAR_EXTENSIONS = (".rar", ".cbr")
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + RAR_EXTENSIONS
loaded_models = {}
system_codepage = get_system_codepage()

settings_parser = SettingsParser(
    {
        "use_cpu": False,
        "use_fp16": settings["UseFp16"],
        "accelerator_device_index": settings["SelectedDeviceIndex"] if settings['SelectedDeviceIndex'] is not None else 0,
        "budget_limit": 0,
    }
)

print("settings", settings_parser.get_int("accelerator_device_index", 0), flush=True)

context = _ExecutorNodeContext(ProgressController(), settings_parser, Path())

gamma1icc = get_gamma_icc_profile()
dotgain20icc = get_dot20_icc_profile()

dotgain20togamma1transform = ImageCms.buildTransformFromOpenProfiles(
    dotgain20icc, gamma1icc, "L", "L"
)
gamma1todotgain20transform = ImageCms.buildTransformFromOpenProfiles(
    gamma1icc, dotgain20icc, "L", "L"
)

if __name__ == "__main__":
    spandrel_custom.install()
    # gc.disable() #TODO!!!!!!!!!!!!
    # Record the start time
    start_time = time.time()

    vram_monitor = None
    if MultiGPUVRAMMonitor is not None:
        device_idx = settings_parser.get_int("accelerator_device_index", 0)
        vram_monitor = MultiGPUVRAMMonitor(device_ids=[device_idx])
        vram_monitor.start()

    image_format = None
    if workflow["WebpSelected"]:
        image_format = "webp"
    elif workflow["PngSelected"]:
        image_format = "png"
    elif workflow["AvifSelected"]:
        image_format = "avif"
    else:
        image_format = "jpeg"

    target_scale: float | None = None
    target_width = 0
    target_height = 0

    grayscale_detection_threshold = workflow["GrayscaleDetectionThreshold"]

    if workflow["ModeScaleSelected"]:
        target_scale = workflow["UpscaleScaleFactor"]
    elif workflow["ModeWidthSelected"]:
        target_width = workflow.get("ResizeWidthAfterUpscale", 0)
        target_height = workflow.get("ResizeHeightAfterUpscale", 0)
    elif workflow["ModeHeightSelected"]:
        target_height = workflow.get("ResizeHeightAfterUpscale", 0)
        target_width = workflow.get("ResizeWidthAfterUpscale", 0)
    else:
        target_width = workflow["DisplayDeviceWidth"]
        target_height = workflow["DisplayDeviceHeight"]

    if workflow["SelectedTabIndex"] == 1:
        upscale_folder(
            workflow["InputFolderPath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["UpscaleImages"],
            workflow["UpscaleArchives"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            loaded_models,
            grayscale_detection_threshold,
        )
    elif workflow["SelectedTabIndex"] == 0:
        upscale_file(
            workflow["InputFilePath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            loaded_models,
            grayscale_detection_threshold,
        )

    if vram_monitor is not None:
        vram_stats = vram_monitor.stop()
        print("VRAM Stats:", vram_stats, flush=True)

    # # Record the end time
    end_time = time.time()

    # # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")