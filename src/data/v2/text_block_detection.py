from typing import List, Union
import numpy as np
from ultralytics import YOLO
from PIL import Image

class TextBlockDetector:
    """
    A class to run YOLO detection models using Ultralytics for text block detection.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the YOLO model.

        Args:
            model_path (str): Path to the pretrained YOLO model (.pt or .onnx).
            device (str): Device to run the model on (e.g., 'cpu', 'cuda', '0').
        """
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, image: Union[str, np.ndarray, Image.Image], conf: float = 0.25, iou: float = 0.45) -> List[dict]:
        """
        Perform detection on the provided image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
                Can be a file path, a numpy array (BGR), or a PIL Image.
            conf (float): Confidence threshold for filtering detections.
            iou (float): Intersection over Union (IoU) threshold for NMS.

        Returns:
            List[dict]: A list of dictionaries containing detection results.
                Format: [
                    {
                        'bbox': [x1, y1, x2, y2],
                        'conf': confidence_score,
                        'class_id': class_integer,
                        'label': class_name
                    },
                    ...
                ]
        """
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )

        detections = []
        # Ultralytics predict returns a list of Results objects (one per image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Convert tensor values to standard Python types
                coords = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                label = self.model.names[class_id]

                detections.append({
                    'bbox': coords,
                    'conf': confidence,
                    'class_id': class_id,
                    'label': label
                })

        return detections

    def __call__(self, image: Union[str, np.ndarray, Image.Image], **kwargs) -> List[dict]:
        """
        Convenience method to call detect().
        """
        return self.detect(image, **kwargs)

if __name__ == "__main__":
    detector = TextBlockDetector("checkpoints/text_det_yolo.onnx")
    results = detector("sample/image_0008_idx7_webp.rf.8e1d94bd2ef32f3feebd0a9e44a86d12.jpg")
    print(results)
