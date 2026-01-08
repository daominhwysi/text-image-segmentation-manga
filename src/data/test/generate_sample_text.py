from src.data.text_render import generate_text_image, crop_tight_with_contours
from src.data.generator import DatasetGenerator
from src.data.utils import initialize_font_samplers
from src.data.augment_text import augment_text_image

FONT_DIR = "resource/fonts"
DATASET_ROOT = "resource/444-2"
dataset_gen = DatasetGenerator(
    dataset_root=DATASET_ROOT, font_dir=FONT_DIR, output_dir=""
)

train_gen, test_gen = initialize_font_samplers(FONT_DIR, split_ratio=0.8)

font_path = train_gen.get_random_font()

print(font_path)
img = generate_text_image(
    text=dataset_gen.get_sentence(),
    font_path=font_path,
    font_size=45,
    shape_ratio=1,
    text_color=(255, 255, 255, 255),
)
augmented = augment_text_image(img)

cropped = crop_tight_with_contours(augmented)

cropped = cropped.convert("RGB")
augmented = augmented.convert("RGB")
cropped.save("quynh mup.jpg")
augmented.save("quynh mups.jpg")
