from openocr import OpenOCR
onnx_engine = OpenOCR(backend='onnx', device='cpu')
img_path = 'resource/444-2/train/images/image_0005_idx4_webp.rf.6b9ec665d5d9d7901bf35235b181c1d0.jpg'
result, elapse = onnx_engine(img_path)
print(result)
print(elapse)
