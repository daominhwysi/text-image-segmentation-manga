import os
import cv2
import numpy as np
import onnxruntime as ort
import time

# ==========================================
# 1. POST-PROCESSING (from ctc_postprocess.py)
# ==========================================
class CTCLabelDecode:
    def __init__(self, character_dict_path=None, use_space_char=False):
        self.character_str = []
        if character_dict_path and os.path.exists(character_dict_path):
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(' ')
        else:
            # Default fallback dictionary
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")

        # CTC Blank is always at index 0 in this repo
        self.character = ['blank'] + self.character_str

    def __call__(self, preds):
        """
        preds: numpy array [batch, steps, classes]
        """
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        result_list = []
        for b in range(len(preds_idx)):
            selection = np.ones(len(preds_idx[b]), dtype=bool)
            # CTC Duplicate Removal
            selection[1:] = preds_idx[b][1:] != preds_idx[b][:-1]
            # Blank Token Removal (index 0)
            selection &= (preds_idx[b] != 0)

            char_list = [self.character[i] for i in preds_idx[b][selection]]
            conf_list = preds_prob[b][selection]

            text = ''.join(char_list)
            score = np.mean(conf_list) if len(conf_list) > 0 else 0.0
            result_list.append((text, score))
        return result_list

# ==========================================
# 2. PRE-PROCESSING (from infer_rec.py - RatioRecTVReisze)
# ==========================================
class RecPreProcess:
    def __init__(self):
        self.max_ratio = 12
        self.base_h = 48
        self.base_shape = [[96, 48], [144, 48], [192, 48], [240, 48]]

    def resize_norm_img(self, img):
        # img is BGR from cv2
        h, w = img.shape[:2]
        gen_ratio = max(1, round(float(w) / float(h)))
        ratio_resize = min(gen_ratio, self.max_ratio)

        if ratio_resize <= 4:
            imgW, imgH = self.base_shape[ratio_resize - 1]
        else:
            imgW, imgH = self.base_h * ratio_resize, self.base_h

        # Resize using Cubic interpolation (matches BICUBIC in torchvision)
        img = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)

        # Normalize (matches T.Normalize(0.5, 0.5))
        # Logic: (x / 255.0 - 0.5) / 0.5  => x / 127.5 - 1.0
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0

        # HWC to CHW
        img = img.transpose((2, 0, 1))
        return img

# ==========================================
# 3. MAIN INFERENCE ENGINE
# ==========================================
class StandaloneRecognizer:
    def __init__(self, model_path, dict_path, use_space_char=True):
        # Setup ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if \
                    'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Setup classes
        self.preprocess = RecPreProcess()
        self.decoder = CTCLabelDecode(dict_path, use_space_char=use_space_char)

    def run(self, img_list):
        if not isinstance(img_list, list):
            img_list = [img_list]

        processed_imgs = [self.preprocess.resize_norm_img(img) for img in img_list]

        # Batching with padding (as per tools/infer_rec.py)
        max_w = max([img.shape[2] for img in processed_imgs])
        max_h = max([img.shape[1] for img in processed_imgs])

        batch_data = np.zeros((len(processed_imgs), 3, max_h, max_w), dtype=np.float32)
        for i, img in enumerate(processed_imgs):
            _, h, w = img.shape
            batch_data[i, :, :h, :w] = img

        # Inference
        start_time = time.time()
        # preds[0] shape is [Batch, TimeSteps, NumClasses]
        preds = self.session.run(None, {self.input_name: batch_data})[0]
        elapse = time.time() - start_time

        # Post-process
        results = self.decoder(preds)

        return [{"text": res[0], "score": float(res[1]), "elapse": elapse} for res in results]

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Settings
    ONNX_MODEL = "checkpoints/openocr_rec_model.onnx"
    DICT_FILE = "ppocr_keys_v1.txt"
    IMAGE_PATH = "sample/image_00000.jpg" # This should be a cropped image of a text line

    if not os.path.exists(ONNX_MODEL):
        print(f"Error: Model {ONNX_MODEL} not found.")
    elif not os.path.exists(DICT_FILE):
        print(f"Error: Dictionary {DICT_FILE} not found.")
    else:
        recognizer = StandaloneRecognizer(ONNX_MODEL, DICT_FILE)

        img = cv2.imread(IMAGE_PATH)
        if img is not None:
            results = recognizer.run(img)
            for res in results:
                print(f"Recognized: '{res['text']}' | Score: {res['score']:.4f} | Time: {res['elapse']:.4f}s")
        else:
            print(f"Error: Could not read image {IMAGE_PATH}")
