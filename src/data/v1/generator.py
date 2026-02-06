import os
import random
import sys
from tqdm import tqdm
from faker import Faker
from src.data.v1.utils import initialize_font_samplers, check_font_chars_support
import numpy as np
from src.data.v1.bg_manager import get_random_non_overlapping_roi
from PIL import Image, ImageDraw
from src.data.v1.text_render import generate_text_image
from src.data.v1.augment_text import augment_output_image
from src.data.openocr.det_infer import OpenOCRDetector
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools


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
    is_contrast_mode = random.random() < 0.9
    is_roi_bg = random.random() >= 0.25

    # Determine outline usage based on background type
    if is_roi_bg:
        use_outline = random.random() < 0.75
    else:
        use_outline = random.random() < 0.5

    # Deciding effects early to reuse them in contrast correction if needed
    use_shadow = random.random() < 0.2
    use_glow = random.random() < 0.2
    use_morphology = random.choice([None, "dilation", "erosion"]) if random.random() < 0.3 else None
    morph_size = random.randint(1, 3)

    if is_contrast_mode:
        want_white_text = random.choice([True, False])
        text_color = get_natural_color(is_white=want_white_text)
        stroke_color = get_natural_color(is_white=not want_white_text)
        shadow_color = get_natural_color(is_white=not want_white_text)[:3] + (128,)
        glow_color = get_natural_color(is_white=not want_white_text)[:3] + (128,)
    else:
        text_color = get_random_rgb_alpha()
        stroke_color = get_random_rgb_alpha()
        shadow_color = get_random_rgb_alpha()[:3] + (128,)
        glow_color = get_random_rgb_alpha()[:3] + (128,)

    def call_render(t, f, tc, sc, shc, gc):
        return generate_text_image(
            text=t,
            font_path=f,
            font_size=random.randint(35, 60),
            shape_ratio=random.uniform(0.25, 4.0),
            outline=use_outline,
            text_color=tc,
            stroke_color=sc,
            stroke_width=random.randint(1, 5),
            shadow=use_shadow,
            shadow_color=shc,
            shadow_offset=(random.randint(1,4), random.randint(1,4)),
            shadow_blur=random.randint(0, 3),
            glow=use_glow,
            glow_color=gc,
            glow_width=random.randint(1, 6),
            morphology_step=use_morphology,
            morphology_size=morph_size,
            alignment=random.choice(["left", "center", "right"]),
            line_height_bias=random.uniform(0, 0.5),
        )

    # --- 2. GENERATE INITIAL TEXT IMAGE ---
    text_img = call_render(text, font_path, text_color, stroke_color, shadow_color, glow_color)

    tw, th = text_img.size
    text_ratio = tw / th
    target_roi_w = FIXED_WIDTH
    target_roi_h = int(FIXED_WIDTH / text_ratio)
    target_roi_h = min(target_roi_h, FIXED_WIDTH * 2)

    # --- 3. BACKGROUND SELECTION ---
    if not is_roi_bg:
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
            shadow_color = get_natural_color(is_white=not want_white_text)[:3] + (128,)
            glow_color = get_natural_color(is_white=not want_white_text)[:3] + (128,)

            text_img = call_render(text, font_path, text_color, stroke_color, shadow_color, glow_color)

    # --- 5. FINAL COMPOSITE ---
    final_bg = Image.fromarray(roi_background)

    if random.random() < 0.7:
        final_bg = add_random_noise_to_bg(final_bg, is_contrast_mode, avg_brightness)

    bg_w, bg_h = final_bg.size
    text_img.thumbnail((bg_w, bg_h), Image.Resampling.LANCZOS)
    curr_tw, curr_th = text_img.size
    offset = ((bg_w - curr_tw) // 2, (bg_h - curr_th) // 2)

    # SOFT MASK IMPLEMENTATION
    mask = Image.new("L", (bg_w, bg_h), 0)
    text_alpha = text_img.split()[3]
    mask.paste(text_alpha, offset)

    final_bg.paste(text_img, offset, text_img)

    final_bg = final_bg.resize((256, 256), Image.Resampling.LANCZOS)
    mask = mask.resize((256, 256), Image.Resampling.BILINEAR)
    # Keeping soft mask values for better learning (anti-aliasing)

    final_bg = augment_output_image(final_bg)
    return final_bg, mask


_worker_detector = None


def get_detector(model_path):
    global _worker_detector
    if _worker_detector is None and model_path:
        _worker_detector = OpenOCRDetector(model_path)
    return _worker_detector


def generate_single_task(task_info):
    (
        dataset_root,
        text,
        font_path,
        output_dir,
        split,
        i,
        detector_model_path,
    ) = task_info

    detector = get_detector(detector_model_path)

    result = generate_composited_sample(
        dataset_root=dataset_root,
        text=text,
        font_path=font_path,
        FIXED_WIDTH=256,
        detector=detector,
    )

    if result is None:
        return False

    img, mask = result
    file_id = f"sample_{i:06d}"
    img.save(
        os.path.join(output_dir, split, "images", f"{file_id}.jpg"), quality=95
    )
    mask.save(os.path.join(output_dir, split, "masks", f"{file_id}.png"))
    return True


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

        for i in range(len(words)):
            prob = random.random()
            if prob < 0.02:
                pair = random.choice([("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ('"', '"'), ("'", "'")])
                words[i] = f"{pair[0]}{words[i]}{pair[1]}"
            elif prob < 0.04:
                char = random.choice(["#", "@", "~", "-", "+", "=", "%", "^", "&", "*", "|"])
                words[i] = f"{char}{words[i]}" if random.getrandbits(1) else f"{words[i]}{char}"

        text = " ".join(words)
        terminators = [".", "!", "?", "-", "...", ":", ";", " #", " @", " %", " -", " >", " ]", " }", " |"]
        weights = [12, 12, 12, 12, 12, 8, 8, 5, 5, 5, 5, 1, 1, 1, 1]
        chosen_terminator = random.choices(terminators, weights=weights)[0]
        return text.capitalize() + chosen_terminator

    def generate(self, n_samples=100, train_ratio=0.8, num_workers=4):
        self._prepare_dirs()
        n_train = int(n_samples * train_ratio)

        print(f"Generating {n_samples} samples with {num_workers} workers...")

        tasks = []
        for i in range(n_samples):
            if i < n_train:
                split = "train"
                font_sampler = self.train_fonts
            else:
                split = "test"
                font_sampler = self.test_fonts

            font_path = font_sampler.get_random_font()
            text = self.get_sentence()

            retry_count = 0
            while (
                font_path
                and not check_font_chars_support(font_path, text)
                and retry_count < 10
            ):
                font_path = font_sampler.get_random_font()
                text = self.get_sentence()
                retry_count += 1

            if not font_path or not check_font_chars_support(font_path, text):
                continue

            # We pass the path to the detector model instead of the detector instance itself
            # because the ONNX session is not pickleable.
            detector_path = (
                self.detector.model_path if self.detector else None
            )

            tasks.append(
                (
                    self.dataset_root,
                    text,
                    font_path,
                    self.output_dir,
                    split,
                    i,
                    detector_path,
                )
            )

        results_count = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(generate_single_task, task) for task in tasks]
            for _ in tqdm(as_completed(futures), total=len(tasks)):
                if _.result():
                    results_count += 1

        print(f"Finished. Generated {results_count} samples.")


if __name__ == "__main__":
    SOURCE_DATA = "resource/444-2/train"
    FONT_DIR = "resource/fonts"
    EXPORT_DEST = "output/synthetic_dataset"
    TOTAL_SAMPLES = 50000
    # Take args as number of sample
    TOTAL_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else TOTAL_SAMPLES
    DET_MODEL = "checkpoints/openocr_det_model.onnx"
    generator = DatasetGenerator(
        dataset_root=SOURCE_DATA,
        font_dir=FONT_DIR,
        output_dir=EXPORT_DEST,
        ocr_det_model=DET_MODEL,
    )
    generator.generate(n_samples=TOTAL_SAMPLES)
