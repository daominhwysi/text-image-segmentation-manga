import os
import random
from tqdm import tqdm
from faker import Faker
from src.data.v1.utils import initialize_font_samplers, check_font_chars_support
import numpy as np
from src.data.v1.bg_manager import get_random_non_overlapping_roi
from PIL import Image, ImageDraw
from src.data.v1.text_render import generate_text_image
from src.data.v1.augment_text import augment_output_image
from src.data.openocr.det_infer import OpenOCRDetector


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


def add_random_noise_to_bg(
    bg_img: Image.Image, is_contrast_mode: bool, avg_brightness: float
):
    """Draws random lines and Bezier curves on the background image."""
    draw = ImageDraw.Draw(bg_img)
    w, h = bg_img.size

    def get_noise_color():
        if is_contrast_mode:
            # Usually try to contrast with background
            is_bg_light = avg_brightness > 127.5
            if random.random() < 0.8:
                return get_natural_color(is_white=not is_bg_light)
            else:
                return get_natural_color(is_white=is_bg_light)
        else:
            return get_random_rgb_alpha()

    # Random straight lines
    num_lines = random.randint(0, 10)
    for _ in range(num_lines):
        p1 = (random.randint(0, w), random.randint(0, h))
        p2 = (random.randint(0, w), random.randint(0, h))
        color = get_noise_color()
        width = random.randint(1, 4)
        draw.line([p1, p2], fill=color, width=width)

    # Random Bezier curves
    num_curves = random.randint(0, 10)
    for _ in range(num_curves):
        p0 = (random.randint(0, w), random.randint(0, h))
        p1 = (random.randint(0, w), random.randint(0, h))
        p2 = (random.randint(0, w), random.randint(0, h))
        p3 = (random.randint(0, w), random.randint(0, h))

        # Approximate cubic Bezier
        steps = 20
        points = []
        for t_val in np.linspace(0, 1, steps):
            x = (
                (1 - t_val) ** 3 * p0[0]
                + 3 * (1 - t_val) ** 2 * t_val * p1[0]
                + 3 * (1 - t_val) * t_val**2 * p2[0]
                + t_val**3 * p3[0]
            )
            y = (
                (1 - t_val) ** 3 * p0[1]
                + 3 * (1 - t_val) ** 2 * t_val * p1[1]
                + 3 * (1 - t_val) * t_val**2 * p2[1]
                + t_val**3 * p3[1]
            )
            points.append((x, y))

        color = get_noise_color()
        width = random.randint(1, 4)
        draw.line(points, fill=color, width=width)

    return bg_img


def generate_composited_sample(
    dataset_root: str, text: str, font_path: str, FIXED_WIDTH=320, detector=None
):
    # --- 1. DETERMINE MODE (90% Contrast vs 10% Random) ---
    is_contrast_mode = random.random() < 0.9

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
        stroke_width=random.randint(1, 5),
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
            dataset_root, roi_size=(target_roi_w, target_roi_h), detector=detector
        )

    if roi_background is None:
        return None

    # --- 4. CONTRAST CORRECTION ---
    avg_brightness = get_avg_brightness(roi_background)
    if is_contrast_mode:
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
                stroke_width=random.randint(1, 5),
                alignment=random.choice(["left", "center", "right"]),
            )

    # --- 5. FINAL COMPOSITE ---
    final_bg = Image.fromarray(roi_background)

    # Add extra random noise to background before pasting text
    if random.random() < 0.7:  # 70% chance to add noise
        final_bg = add_random_noise_to_bg(final_bg, is_contrast_mode, avg_brightness)

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
    final_bg = final_bg.resize((256, 256), Image.Resampling.LANCZOS)

    mask = mask.resize(
        (256, 256), Image.Resampling.NEAREST
    )  # Nearest used for mask to keep it binary
    final_bg = augment_output_image(final_bg)
    return final_bg, mask


def get_random_word_count(mu, sigma):
    count = np.random.normal(mu, sigma)
    return max(1, int(round(count)))


class DatasetGenerator:
    def __init__(
        self,
        dataset_root,
        font_dir,
        output_dir,
        ocr_det_model=None,
    ):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.fake = Faker()
        self.train_fonts, self.test_fonts = initialize_font_samplers(
            font_dir, split_ratio=0.8
        )
        if ocr_det_model:
            print("Initializing Background Cleaning Detector (OpenOCR)...")
            self.detector = OpenOCRDetector(ocr_det_model)
        else:
            self.detector = None

    def _prepare_dirs(self):
        for split in ["train", "test"]:
            for sub in ["images", "masks"]:
                path = os.path.join(self.output_dir, split, sub)
                os.makedirs(path, exist_ok=True)

    def get_sentence(self):
        word_count = get_random_word_count(mu=6.83, sigma=6.47)
        words = self.fake.words(nb=word_count)
        if not words:
            return ""

        # 1. Randomly wrap words in "Odd" characters (Lower probability)
        for i in range(len(words)):
            prob = random.random()
            if prob < 0.02:  # 2% chance for "The Odd Ones"
                pair = random.choice(
                    [
                        ("(", ")"),
                        ("[", "]"),
                        ("{", "}"),
                        ("<", ">"),
                        ('"', '"'),
                        ("'", "'"),
                    ]
                )
                words[i] = f"{pair[0]}{words[i]}{pair[1]}"
            elif prob < 0.04:  # 2% chance for "Prefix/Suffix Noise"
                char = random.choice(
                    ["#", "@", "~", "-", "+", "=", "%", "^", "&", "*", "|"]
                )
                words[i] = (
                    f"{char}{words[i]}"
                    if random.getrandbits(1)
                    else f"{words[i]}{char}"
                )

        text = " ".join(words)

        # 2. Weighted Terminators
        # We use a list of weights to ensure standard terminators remain dominant
        # while "Odd" ones show up just enough for the AI to learn them.
        terminators = [
            ".",
            "!",
            "?",
            "-",
            "...",  # Standard (High weight)
            ":",
            ";",  # Common (Medium weight)
            " #",
            " @",
            " %",
            " -",  # Odd (Low weight)
            " >",
            " ]",
            " }",
            " |",  # Rare (Very low weight)
        ]
        weights = [
            12,
            12,
            12,
            12,
            12,  # Standard (High)
            8,
            8,  # Common (Medium)
            5,
            5,
            5,
            5,  # Odd (Low)
            1,
            1,
            1,
            1,  # Rare (Very low)
        ]
        # Total Sum: 100

        chosen_terminator = random.choices(terminators, weights=weights)[0]

        # Capitalize for a standard starting look, then add the tail
        return text.capitalize() + chosen_terminator

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

            font_path = font_sampler.get_random_font()
            text = self.get_sentence()

            # Ensure font supports characters in text
            retry_count = 0
            while font_path and not check_font_chars_support(font_path, text) and retry_count < 10:
                font_path = font_sampler.get_random_font()
                text = self.get_sentence()
                retry_count += 1

            if not font_path or not check_font_chars_support(font_path, text):
                continue

            # FIXED_WIDTH here still influences the "origin logic" aspect ratio,
            # but the output is guaranteed 128x128 by the function above.
            result = generate_composited_sample(
                dataset_root=self.dataset_root,
                text=text,
                font_path=font_path,
                FIXED_WIDTH=256,
                detector=self.detector,
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
    SOURCE_DATA = "resource/444-2/train"
    FONT_DIR = "resource/fonts"
    EXPORT_DEST = "synthetic_dataset"
    TOTAL_SAMPLES = 50000

    DET_MODEL = "checkpoints/openocr_det_model.onnx"

    generator = DatasetGenerator(
        dataset_root=SOURCE_DATA,
        font_dir=FONT_DIR,
        output_dir=EXPORT_DEST,
        ocr_det_model=DET_MODEL,
    )
    generator.generate(n_samples=TOTAL_SAMPLES)
