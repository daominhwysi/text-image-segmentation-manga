import math
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import textwrap
from src.data.v1.augment_text import augment_text_image


def crop_tight_with_contours(pil_img: Image.Image) -> Image.Image:
    open_cv_image = np.array(pil_img)
    if open_cv_image.shape[2] < 4:
        return pil_img

    alpha_channel = open_cv_image[:, :, 3]
    _, thresh = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        all_pts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        return pil_img.crop((x, y, x + w, y + h))

    return pil_img


def apply_morphology(img: Image.Image, kernel_size: int, op_type: str) -> Image.Image:
    """Applies dilation or erosion to the alpha channel of an image."""
    if kernel_size <= 0:
        return img

    img_np = np.array(img)
    alpha = img_np[:, :, 3]

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if op_type == "dilation":
        alpha = cv2.dilate(alpha, kernel, iterations=1)
    elif op_type == "erosion":
        alpha = cv2.erode(alpha, kernel, iterations=1)

    # We only modify the alpha channel to preserve text color
    img_np[:, :, 3] = alpha
    return Image.fromarray(img_np)


def generate_text_image(
    text: str,
    font_path: str,
    font_size: int,
    shape_ratio: float,
    outline: bool = False,
    text_color: tuple = (0, 0, 0, 255),
    stroke_color: tuple = (255, 0, 0, 255),
    stroke_width: int = None,
    shadow: bool = False,
    shadow_color: tuple = (0, 0, 0, 128),
    shadow_offset: tuple = (2, 2),
    shadow_blur: int = 2,
    glow: bool = False,
    glow_color: tuple = (255, 255, 255, 128),
    glow_width: int = 5,
    morphology_step: str = None, # "dilation" or "erosion"
    morphology_size: int = 0,
    alignment: str = "left",
    line_height_bias: float = 0.0,
    min_threshold_height: int = 45,
    max_retries: int = 5,
) -> Image.Image:
    """
    Generates a transparent image containing text with various effects.
    """

    attempt = 0
    last_valid_img = None

    while attempt < max_retries:
        attempt += 1

        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()

        # Better wrapping logic
        dummy_bbox = font.getbbox("a")
        avg_char_width = dummy_bbox[2] - dummy_bbox[0]
        base_height = (dummy_bbox[3] - dummy_bbox[1]) + int(font_size * 0.2)
        line_height = int(base_height * (1 + line_height_bias))

        text_area = len(text) * avg_char_width * line_height
        target_width_px = int(math.sqrt(text_area * shape_ratio))

        # Approximate characters per line for textwrap
        avg_char_len = font.getlength(text) / len(text) if len(text) > 0 else avg_char_width
        chars_per_line = max(1, int(target_width_px / avg_char_len))

        lines = textwrap.wrap(text, width=chars_per_line, break_long_words=False, replace_whitespace=False)
        if not lines:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        padding = int(font_size * 2) # Extra padding for effects
        final_text_width = max([font.getlength(line) for line in lines])
        final_text_height = len(lines) * line_height

        canvas_width = int(final_text_width + (padding * 2))
        canvas_height = int(final_text_height + (padding * 2))

        # We'll use multiple layers for effects
        # Base layer for main text
        base_text_layer = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        draw_base = ImageDraw.Draw(base_text_layer)

        # Drawing text helper to avoid code duplication
        def draw_lines(draw_obj, current_lines, fill_color, s_width=0, s_fill=None):
            curr_y = padding
            for line in current_lines:
                lw = font.getlength(line)
                if alignment == "center":
                    lx = padding + (final_text_width - lw) / 2
                elif alignment == "right":
                    lx = padding + (final_text_width - lw)
                else:
                    lx = padding

                draw_obj.text((lx, curr_y), line, font=font, fill=fill_color,
                             stroke_width=s_width, stroke_fill=s_fill)
                curr_y += line_height

        # 1. Draw Base Text
        draw_lines(draw_base, lines, text_color)

        # 2. Apply Morphology (Dilation/Erosion) to base text layer
        if morphology_step in ["dilation", "erosion"] and morphology_size > 0:
            base_text_layer = apply_morphology(base_text_layer, morphology_size, morphology_step)

        # 3. Create composite
        final_img = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

        # A. Shadow Layer
        if shadow:
            shadow_layer = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
            draw_sh = ImageDraw.Draw(shadow_layer)
            # Shifted draw
            curr_y = padding + shadow_offset[1]
            for line in lines:
                lw = font.getlength(line)
                lx = padding + shadow_offset[0]
                if alignment == "center": lx += (final_text_width - lw) / 2
                elif alignment == "right": lx += (final_text_width - lw)
                draw_sh.text((lx, curr_y), line, font=font, fill=shadow_color)
                curr_y += line_height

            if shadow_blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
            final_img.alpha_composite(shadow_layer)

        # B. Glow Layer
        if glow:
            glow_layer = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
            draw_gl = ImageDraw.Draw(glow_layer)
            draw_lines(draw_gl, lines, glow_color)
            if glow_width > 0:
                glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(glow_width))
            final_img.alpha_composite(glow_layer)

        # C. Outline (Stroke) - rendered behind text but can be part of text
        if outline:
            stroke_layer = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
            draw_st = ImageDraw.Draw(stroke_layer)
            s_width = stroke_width if stroke_width is not None else max(1, int(font_size / 15))
            draw_lines(draw_st, lines, text_color, s_width=s_width, s_fill=stroke_color)
            final_img.alpha_composite(stroke_layer)
        else:
            final_img.alpha_composite(base_text_layer)

        # Final composite assembly (ensure text color is on top)
        # Note: If outline is on, we already have text_color inside it.
        # If not, we composite base text on top of shadow/glow.
        if not outline:
            final_img.alpha_composite(base_text_layer)

        # Augmentation and Cropping
        augmented = augment_text_image(final_img)
        cropped = crop_tight_with_contours(augmented)

        cw, ch = cropped.size
        if ch >= min_threshold_height and cw > 5:
            return cropped

        last_valid_img = cropped

    return last_valid_img
