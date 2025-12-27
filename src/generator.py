import os
import random
from tqdm import tqdm
from faker import Faker
from src.utils import initialize_font_samplers
import numpy as np
from src.bg_manager import get_random_non_overlapping_roi
from PIL import Image
from src.text_render import generate_text_image


def get_random_rgb_alpha():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)


def get_natural_color(is_white=True):
    """Returns a color that looks white or black but isn't pure #000 or #FFF."""
    if is_white:
        # Off-white / Cream / Light Gray
        val = random.randint(238, 255)
        return (val, val, random.randint(230, 255), 255)
    else:
        # Charcoal / Deep Gray / Off-black
        val = random.randint(10, 45)
        return (val, val, random.randint(10, 50), 255)


def get_avg_brightness(roi_np):
    """Calculates luminance: $L = 0.299R + 0.587G + 0.114B$"""
    return np.mean(
        0.299 * roi_np[:, :, 0] + 0.587 * roi_np[:, :, 1] + 0.114 * roi_np[:, :, 2]
    )


def generate_composited_sample(
    dataset_root: str, text: str, font_path: str, FIXED_WIDTH=320
):
    # --- 1. DETERMINE MODE (50% Contrast vs 50% Random) ---
    is_contrast_mode = random.random() < 0.5

    if is_contrast_mode:
        want_white_text = random.choice([True, False])
        text_color = get_natural_color(is_white=want_white_text)
        stroke_color = get_natural_color(is_white=not want_white_text)
    else:
        text_color = get_random_rgb_alpha()
        stroke_color = get_random_rgb_alpha()

    # --- 2. GENERATE INITIAL TEXT IMAGE ---
    text_img = generate_text_image(
        text=text,
        font_path=font_path,
        font_size=random.randint(35, 60),
        shape_ratio=random.uniform(0.5, 2.0),
        outline=random.choice([True, False]),
        text_color=text_color,
        stroke_color=stroke_color,
        stroke_width=random.randint(1, 2),
        alignment=random.choice(["left", "center", "right"]),
        line_height_bias=random.uniform(0, 0.5),
    )

    tw, th = text_img.size
    text_ratio = tw / th
    target_roi_w = FIXED_WIDTH
    target_roi_h = int(FIXED_WIDTH / text_ratio)
    target_roi_h = min(target_roi_h, FIXED_WIDTH * 2)

    # --- 3. BACKGROUND SELECTION (NEW 25% LOGIC) ---
    if random.random() < 0.25:
        bg_is_white = random.choice([True, False])
        bg_color = get_natural_color(is_white=bg_is_white)
        roi_background = np.full(
            (target_roi_h, target_roi_w, 3), bg_color[:3], dtype=np.uint8
        )
    else:
        roi_background, _ = get_random_non_overlapping_roi(
            dataset_root, roi_size=(target_roi_w, target_roi_h)
        )

    if roi_background is None:
        return None

    # --- 4. CONTRAST CORRECTION ---
    if is_contrast_mode:
        avg_brightness = get_avg_brightness(roi_background)
        is_bg_light = avg_brightness > 127.5

        if (want_white_text and is_bg_light) or (
            not want_white_text and not is_bg_light
        ):
            want_white_text = not want_white_text
            text_color = get_natural_color(is_white=want_white_text)
            stroke_color = get_natural_color(is_white=not want_white_text)

            text_img = generate_text_image(
                text=text,
                font_path=font_path,
                font_size=random.randint(35, 60),
                shape_ratio=text_ratio,
                outline=random.choice([True, False]),
                text_color=text_color,
                stroke_color=stroke_color,
                stroke_width=random.randint(1, 2),
                alignment=random.choice(["left", "center", "right"]),
            )

    # --- 5. FINAL COMPOSITE ---
    final_bg = Image.fromarray(roi_background)
    bg_w, bg_h = final_bg.size
    text_img.thumbnail((bg_w, bg_h), Image.Resampling.LANCZOS)
    curr_tw, curr_th = text_img.size
    offset = ((bg_w - curr_tw) // 2, (bg_h - curr_th) // 2)

    mask = Image.new("L", (bg_w, bg_h), 0)
    text_alpha = text_img.split()[3]
    mask.paste(text_alpha, offset)
    mask = mask.point(lambda p: 255 if p > 10 else 0)

    final_bg.paste(text_img, offset, text_img)

    # --- 6. RESIZE TO 128x128 (NEW STEP) ---
    # We do this at the very end to preserve the original composition logic
    final_bg = final_bg.resize((128, 128), Image.Resampling.LANCZOS)
    mask = mask.resize(
        (128, 128), Image.Resampling.NEAREST
    )  # Nearest used for mask to keep it binary

    return final_bg, mask


def get_random_word_count(mu, sigma):
    count = np.random.normal(mu, sigma)
    return max(1, int(round(count)))


class DatasetGenerator:
    def __init__(self, dataset_root, output_dir):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.fake = Faker()
        self.train_fonts, self.test_fonts = initialize_font_samplers(
            "fonts", split_ratio=0.8
        )

    def _prepare_dirs(self):
        for split in ["train", "test"]:
            for sub in ["images", "masks"]:
                path = os.path.join(self.output_dir, split, sub)
                os.makedirs(path, exist_ok=True)

    def generate(self, n_samples=100, train_ratio=0.8):
        self._prepare_dirs()
        n_train = int(n_samples * train_ratio)

        print(f"Generating {n_samples} samples...")

        for i in tqdm(range(n_samples)):
            if i < n_train:
                split = "train"
                font_sampler = self.train_fonts
            else:
                split = "test"
                font_sampler = self.test_fonts

            word_count = get_random_word_count(mu=6.83, sigma=6.47)
            text = " ".join(self.fake.words(nb=word_count))
            font_path = font_sampler.get_random_font()
            if not font_path:
                continue

            # FIXED_WIDTH here still influences the "origin logic" aspect ratio,
            # but the output is guaranteed 128x128 by the function above.
            result = generate_composited_sample(
                dataset_root=self.dataset_root,
                text=text,
                font_path=font_path,
                FIXED_WIDTH=random.choice([128, 192]),
            )

            if result is None:
                continue

            img, mask = result
            file_id = f"sample_{i:06d}"
            img.save(
                os.path.join(self.output_dir, split, "images", f"{file_id}.jpg"),
                quality=95,
            )
            mask.save(os.path.join(self.output_dir, split, "masks", f"{file_id}.png"))


if __name__ == "__main__":
    SOURCE_DATA = "444-2/train"
    EXPORT_DEST = "synthetic_dataset"
    TOTAL_SAMPLES = 30000

    generator = DatasetGenerator(SOURCE_DATA, EXPORT_DEST)
    generator.generate(n_samples=TOTAL_SAMPLES)
