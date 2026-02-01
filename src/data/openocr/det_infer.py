import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon
import pyclipper

# ==========================================
# 1. YOUR POST-PROCESS LOGIC (Integrated)
# ==========================================
class DBPostProcess(object):
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000,
                 unclip_ratio=2.0, use_dilation=False, score_mode='fast',
                 box_type='quad', **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        idx1, idx2, idx3, idx4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]: idx1, idx4 = 0, 1
        else: idx1, idx4 = 1, 0
        if points[3][1] > points[2][1]: idx2, idx3 = 2, 3
        else: idx2, idx3 = 3, 2
        box = [points[idx1], points[idx2], points[idx3], points[idx4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype('int32'), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype('int32'), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype('int32'), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype('int32'), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype('int32'), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[0] if len(outs) == 2 else outs[1]
        num_contours = min(len(contours), self.max_candidates)
        boxes, scores = [], []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size: continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score: continue
            box = self.unclip(points, self.unclip_ratio)
            if len(box) != 1: continue
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2: continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype('int32'))
            scores.append(score)
        return boxes, scores

    def __call__(self, preds_map, shape_list, **kwargs):
        # shape_list: [src_h, src_w, ratio_h, ratio_w]
        pred = preds_map[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        for i in range(pred.shape[0]):
            src_h, src_w, _, _ = shape_list[i]
            mask = segmentation[i]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(mask.astype(np.uint8), self.dilation_kernel)
            boxes, _ = self.boxes_from_bitmap(pred[i], mask, src_w, src_h)
            boxes_batch.append({'points': boxes})
        return boxes_batch

# ==========================================
# 2. STANDALONE ONNX DETECTOR
# ==========================================
class OpenOCRDetector:
    def __init__(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if \
                    'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Initialize your post-process class
        self.post_process = DBPostProcess(thresh=0.3, box_thresh=0.6, unclip_ratio=1.5)

    def preprocess(self, img, target_size=960):
        h, w, _ = img.shape
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h, new_w = (new_h // 32) * 32, (new_w // 32) * 32

        resized_img = cv2.resize(img, (new_w, new_h))
        img_data = resized_img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img_data = (img_data - mean) / std
        img_data = img_data.transpose((2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        # Meta info for post-process: [src_h, src_w, ratio_h, ratio_w]
        shape_list = np.array([[h, w, new_h/h, new_w/w]], dtype=np.float32)
        return img_data, shape_list

    def __call__(self, input_data):
        if isinstance(input_data, str):
            img = cv2.imread(input_data)
        else:
            img = input_data

        if img is None: return None

        blob, shape_list = self.preprocess(img)

        # ONNX Inference
        start = time.time()
        preds_det = self.session.run(None, {self.input_name: blob})[0]
        elapse = time.time() - start

        # Use your DBPostProcess class
        # Note: we pass torch_tensor=False because we are using numpy
        results = self.post_process(preds_det, shape_list)

        return {"boxes": results[0]['points'], "elapse": elapse}
if __name__ == "__main__":
    # Path to your downloaded .onnx file
    #https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_model.onnx
    MODEL_PATH = "checkpoints/openocr_det_model.onnx"
    IMAGE_PATH = "resource/444-2/train/images/image_0021_idx20_webp.rf.33d1233667b30eb93bc3c39ce9f95944.jpg" # Change to your image

    detector = OpenOCRDetector(MODEL_PATH)
    result = detector(IMAGE_PATH)

    if result:
        img = cv2.imread(IMAGE_PATH)
        for box in result['boxes']:
            cv2.polylines(img, [np.array(box).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite("det_result.jpg", img)
        print("Result saved to det_result.jpg")
