# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from src.models import Unet_B0
# def predict_image(model_path, image_path, device):
#     # 1. Cấu hình các bước tiền xử lý (phải giống hệt lúc train)
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 2. Khởi tạo model và tải trọng số đã lưu
#     # Đảm bảo lớp UNet đã được định nghĩa trước đó trong script
#     num_classes = 2
#     model = Unet_B0(num_classes=2, in_channels=3)

#     # Tải trọng số
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()

#     # 3. Đọc và xử lý ảnh đầu vào
#     original_img = Image.open(image_path).convert("RGB")
#     input_tensor = transform(original_img).unsqueeze(0).to(device) # Thêm dimension Batch [1, 3, 128, 128]

#     # 4. Dự đoán
#     with torch.no_grad():
#         output = model(input_tensor)
#         # Lấy lớp có xác suất cao nhất (argmax)
#         # Output từ model là [1, 2, 128, 128] -> pred là [128, 128]
#         pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

#     # 5. Hiển thị kết quả
#     plt.figure(figsize=(15, 5))

#     # Ảnh gốc (đã resize để dễ so sánh)
#     plt.subplot(1, 3, 1)
#     plt.title("Original Image")
#     plt.imshow(original_img.resize((128, 128)))
#     plt.axis("off")

#     # Mask dự đoán
#     plt.subplot(1, 3, 2)
#     plt.title("Predicted Mask")
#     plt.imshow(pred, cmap='gray')
#     plt.axis("off")

#     # Ảnh đè Mask (Overlay)
#     plt.subplot(1, 3, 3)
#     plt.title("Overlay")
#     # Chuyển ảnh gốc sang numpy để vẽ đè
#     img_np = np.array(original_img.resize((256, 256)))
#     mask_colored = np.zeros_like(img_np)
#     mask_colored[pred == 1] = [255, 0, 0] # Tô màu đỏ cho vật thể (class 1)

#     # Trộn ảnh gốc với màu đỏ của mask (độ trong suốt 0.5)
#     overlay = cv2.addWeighted(img_np, 0.7, mask_colored, 0.3, 0) if 'cv2' in globals() else img_np
#     # Nếu không có cv2, vẽ đơn giản bằng cách dùng mask làm alpha:
#     plt.imshow(img_np)
#     plt.imshow(pred, cmap='jet', alpha=0.4) # Jet tạo màu rực rỡ cho vùng object
#     plt.axis("off")

#     plt.show()

# # --- CHẠY INFERENCE ---
# # Thiết lập thiết bị
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Đường dẫn file
# MODEL_PATH = "best_unet_128.pth"
# IMAGE_PATH = "2.png"

# # Gọi hàm dự đoán
# predict_image(MODEL_PATH, IMAGE_PATH, device)