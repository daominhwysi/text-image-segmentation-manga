import os
import cv2
import numpy as np
import time
from src.data.openocr.det_infer import OpenOCRDetector
from src.data.openocr.rec_infer import StandaloneRecognizer

def get_rotate_crop_image(img, points):
    """
    Crop the image based on the 4 points of the bounding box and rectify it.
    Points should be in order: TL, TR, BR, BL
    """
    assert len(points) == 4, "shape of points must be 4*2"
    points = np.array(points, dtype=np.float32)

    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])
        )
    )

    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height]
    ])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC
    )

    dst_img_height, dst_img_width = dst_img.shape[0:2]
    # If the aspect ratio is too high (vertical text line), rotate it
    if dst_img_height * 1.0 / dst_img_width >= 2.0:
        dst_img = np.rot90(dst_img, k=-1) # Rotate 90 degrees clockwise

    return dst_img

class OpenOCRPipeline:
    def __init__(self, det_model_path, rec_model_path, dict_path):
        self.detector = OpenOCRDetector(det_model_path)
        self.recognizer = StandaloneRecognizer(rec_model_path, dict_path)

    def __call__(self, input_data):
        # 1. Read/Prepare image
        if isinstance(input_data, str):
            img = cv2.imread(input_data)
        else:
            img = input_data

        if img is None:
            print(f"Error: Could not read image or image is None")
            return None

        # 2. Detection
        det_start = time.time()
        det_res = self.detector(img)
        det_elapse = time.time() - det_start

        boxes = det_res['boxes']
        if not boxes:
            return {
                "boxes": [],
                "texts": [],
                "scores": [],
                "det_elapse": det_elapse,
                "rec_elapse": 0
            }

        # 3. Crop and rectify each detected text region
        img_list = []
        for box in boxes:
            img_crop = get_rotate_crop_image(img, box)
            img_list.append(img_crop)

        # 4. Recognition
        # StandaloneRecognizer.run handles batching
        rec_start = time.time()
        rec_res = self.recognizer.run(img_list)
        rec_elapse = time.time() - rec_start

        # 5. Compile results
        texts = [res['text'] for res in rec_res]
        scores = [res['score'] for res in rec_res]

        return {
            "boxes": boxes,
            "texts": texts,
            "scores": scores,
            "det_elapse": det_elapse,
            "rec_elapse": rec_elapse
        }

if __name__ == "__main__":
    # Default paths (adjust if needed)
    DET_MODEL = "checkpoints/openocr_det_model.onnx"
    REC_MODEL = "checkpoints/openocr_rec_model.onnx"
    DICT_FILE = "ppocr_keys_v1.txt"

    # Test image from det.py or rec.py examples
    IMAGE_PATH = "resource/444-2/train/images/image_0021_idx20_webp.rf.33d1233667b30eb93bc3c39ce9f95944.jpg"

    # Ensure paths exist
    for p in [DET_MODEL, REC_MODEL, DICT_FILE]:
        if not os.path.exists(p):
            print(f"Warning: Path {p} does not exist. Please check your model locations.")

    # Initialize and run pipeline
    print("Initializing OCR Pipeline...")
    pipeline = OpenOCRPipeline(DET_MODEL, REC_MODEL, DICT_FILE)

    print(f"Processing image: {IMAGE_PATH}")
    result = pipeline(IMAGE_PATH)

    if result:
        print(f"\nDetection took: {result['det_elapse']:.4f}s")
        print(f"Recognition took: {result['rec_elapse']:.4f}s")
        print(f"Total Text Blocks Found: {len(result['boxes'])}\n")

        # Visualization
        img = cv2.imread(IMAGE_PATH)
        for i, (box, text, score) in enumerate(zip(result['boxes'], result['texts'], result['scores'])):
            print(f"Block {i+1}: '{text}' (score: {score:.4f})")

            # Draw box
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

            # Label index (optional, since Chinese/Japanese text doesn't render well in cv2.putText)
            cv2.putText(img, str(i+1), (int(box[0][0]), int(box[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        output_path = "pipeline_result.jpg"
        cv2.imwrite(output_path, img)
        print(f"\nResult visualization saved to: {output_path}")
