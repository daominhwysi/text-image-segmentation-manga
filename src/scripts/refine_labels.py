import os
import argparse
from tqdm import tqdm
from ultralytics import YOLO

def refine_labels(model_path, data_dir, limit=None):
    """
    Refines YOLO labels using a pre-trained model.
    Maps all detected classes to class 0 (text_bubble) for the Speech-bubble-6 dataset.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Path relative to project root or absolute if provided
    img_dir = os.path.join(data_dir, 'train/images')
    label_dir = os.path.join(data_dir, 'train/labels')

    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found at {img_dir}")
        return

    os.makedirs(label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if limit:
        image_files = image_files[:limit]

    print(f"Found {len(image_files)} images in {img_dir}. Starting refinement...")

    refined_count = 0
    for img_name in tqdm(image_files):
        img_path = os.path.join(img_dir, img_name)

        try:
            results = model(img_path, verbose=False)

            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)

            with open(label_path, 'w') as f:
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        # Map all detected classes to 0 ('text_bubble')
                        # The dataset we are refining has only one class.
                        target_cls = 0

                        # box.xywhn returns [x_center, y_center, width, height] normalized
                        if box.xywhn is not None and len(box.xywhn) > 0:
                            xywhn = box.xywhn[0].tolist()
                            line = f"{target_cls} {' '.join([f'{x:.6f}' for x in xywhn])}\n"
                            f.write(line)

            refined_count += 1
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print(f"\nRefinement complete! Processed {refined_count} images.")
    print(f"Labels saved to {label_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine YOLO labels for a dataset.")
    parser.add_argument('--model', type=str, default='checkpoints/yolo_alimi_sfx_text-2.pt', help='Path to YOLO model')
    parser.add_argument('--data', type=str, default='resource/Speech-bubble-6', help='Path to dataset root')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of images to process (for testing)')
    args = parser.parse_args()

    # Ensure paths are absolute or correctly relative to the project root
    project_root = os.getcwd()
    model_path = os.path.join(project_root, args.model) if not os.path.isabs(args.model) else args.model
    data_dir = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data

    refine_labels(model_path, data_dir, args.limit)
