import math
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.data.v1.augment_text import augment_text_image


def crop_tight_with_contours(pil_img: Image.Image) -> Image.Image:
    open_cv_image = np.array(pil_img)
    if open_cv_image.shape[2] < 4:  # Phòng trường hợp không có kênh Alpha
        return pil_img

    alpha_channel = open_cv_image[:, :, 3]
    _, thresh = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        all_pts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        return pil_img.crop((x, y, x + w, y + h))

    return pil_img


def generate_text_image(
    text: str,
    font_path: str,
    font_size: int,
    shape_ratio: float,
    outline: bool = False,
    text_color: tuple = (0, 0, 0, 255),
    stroke_color: tuple = (255, 0, 0, 255),
    stroke_width: int = None,
    alignment: str = "left",
    line_height_bias: float = 0.0,
    min_threshold_height: int = 45,  # Ngưỡng chiều cao tối thiểu
    max_retries: int = 5,  # Số lần thử lại tối đa
) -> Image.Image:
    """
    Generates a transparent image containing text.
    If the resulting image is too small (e.g. due to augmentation), it retries.
    """

    attempt = 0
    last_valid_img = None

    while attempt < max_retries:
        attempt += 1

        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()

        dummy_bbox = font.getbbox("a")
        avg_char_width = dummy_bbox[2] - dummy_bbox[0]
        base_height = (dummy_bbox[3] - dummy_bbox[1]) + int(font_size * 0.2)
        line_height = int(base_height * (1 + line_height_bias))

        text_area = len(text) * avg_char_width * line_height
        target_width = int(math.sqrt(text_area * shape_ratio))

        words = text.split()
        if not words:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        longest_word_len = max([font.getlength(w) for w in words])
        target_width = max(target_width, int(longest_word_len))

        # Line wrapping
        lines = []
        current_line = []
        current_line_width = 0
        space_width = font.getlength(" ")

        for word in words:
            word_width = font.getlength(word)
            if current_line_width + word_width <= target_width:
                current_line.append(word)
                current_line_width += word_width + space_width
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_line_width = word_width + space_width
        if current_line:
            lines.append(" ".join(current_line))

        padding = int(font_size * 1.5)
        final_text_width = max([font.getlength(line) for line in lines])
        final_text_height = len(lines) * line_height

        canvas_width = int(final_text_width + (padding * 2))
        canvas_height = int(final_text_height + (padding * 2))

        img = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        actual_stroke_width = 0
        if outline:
            actual_stroke_width = (
                stroke_width
                if stroke_width is not None
                else max(1, int(font_size / 15))
            )

        y_text = padding
        for line in lines:
            line_w = font.getlength(line)
            if alignment == "center":
                x_text = padding + (final_text_width - line_w) / 2
            elif alignment == "right":
                x_text = padding + (final_text_width - line_w)
            else:
                x_text = padding

            draw.text(
                (x_text, y_text),
                line,
                font=font,
                fill=text_color,
                stroke_width=actual_stroke_width,
                stroke_fill=stroke_color if outline else None,
            )
            y_text += line_height

        # Áp dụng Augmentation (Bước này có thể làm ảnh nhỏ đi do xoay/phóng đại)
        augmented = augment_text_image(img)
        # Crop sát
        cropped = crop_tight_with_contours(augmented)

        # KIỂM TRA ĐIỀU KIỆN KÍCH THƯỚC
        cw, ch = cropped.size
        if ch >= min_threshold_height and cw > 5:
            return cropped

        # Nếu không đạt, lưu lại ảnh "tệ nhất" phòng trường hợp fail hết cả 5 lần
        last_valid_img = cropped

    # Nếu sau max_retries vẫn nhỏ, trả về cái cuối cùng (hoặc xử lý tùy ý)
    return last_valid_img
