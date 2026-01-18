import cv2
import numpy as np
import time
from typing import List, Dict, Any, Union
from src.data.v2.text_block_detection import TextBlockDetector
from src.data.v2.openocr.e2e_infer import OpenOCRPipeline

class MangaOCRPipeline:
    """
    A two-stage pipeline for Manga OCR:
    1. Detect text blocks (e.g., speech bubbles, captions) using YOLO.
    2. Perform OCR on each detected block using OpenOCR (Detection + Recognition).
    """
    def __init__(
        self,
        text_block_model_path: str,
        ocr_det_model_path: str,
        ocr_rec_model_path: str,
        ocr_dict_path: str,
        device: str = 'cpu'
    ):
        print(f"Initializing MangaOCRPipeline on {device}...")
        self.text_block_detector = TextBlockDetector(text_block_model_path, device=device)
        self.ocr_pipeline = OpenOCRPipeline(ocr_det_model_path, ocr_rec_model_path, ocr_dict_path)

    def __call__(self, input_data: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process an image through the full pipeline.

        Args:
            input_data (str or np.ndarray): Path to the image or a numpy array (BGR).

        Returns:
            List[Dict]: A list of results for each detected text block.
        """
        # 1. Read/Prepare image
        if isinstance(input_data, str):
            img = cv2.imread(input_data)
        else:
            img = input_data

        if img is None:
            print("Error: Could not read image or image is None")
            return []

        # 2. Step 1: Detect text blocks
        det_start = time.time()
        blocks = self.text_block_detector(img)
        det_elapse = time.time() - det_start
        print(f"Detected {len(blocks)} text blocks in {det_elapse:.4f}s")

        results = []
        for i, block in enumerate(blocks):
            bbox = block['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)

            # Ensure bbox is within image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop ROI
            roi_img = img[y1:y2, x1:x2]
            if roi_img.size == 0:
                continue

            # Step 2: Perform OCR on ROI (includes internal line detection and recognition)
            ocr_res = self.ocr_pipeline(roi_img)

            if ocr_res:
                # Adjust line bboxes in ocr_res to be relative to the original image
                adjusted_boxes = []
                for box in ocr_res['boxes']:
                    # box is usually 4 points: [[x, y], [x, y], [x, y], [x, y]]
                    adjusted_box = []
                    for pt in box:
                        adjusted_box.append([pt[0] + x1, pt[1] + y1])
                    adjusted_boxes.append(adjusted_box)

                ocr_res['boxes'] = adjusted_boxes

            results.append({
                'block_index': i,
                'block_bbox': [x1, y1, x2, y2],
                'block_conf': block['conf'],
                'block_label': block['label'],
                'ocr_results': ocr_res
            })

        return results

if __name__ == "__main__":
    # Default paths (adjust as needed)
    TEXT_BLOCK_MODEL = "checkpoints/text_det_yolo.onnx"
    OCR_DET_MODEL = "checkpoints/openocr_det_model.onnx"
    OCR_REC_MODEL = "checkpoints/openocr_rec_model.onnx"
    OCR_DICT = "ppocr_keys_v1.txt"
    IMAGE_PATH = "sample/test_manga_page.jpg"

    # Initialize pipeline
    # Note: Ensure you have the models at the specified paths
    # pipeline = MangaOCRPipeline(TEXT_BLOCK_MODEL, OCR_DET_MODEL, OCR_REC_MODEL, OCR_DICT)
    # results = pipeline(IMAGE_PATH)
    # print(f"Results: {results}")
    print("MangaOCRPipeline script loaded. Ready to process images.")
