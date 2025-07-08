import os
import json
import shutil
from pathlib import Path

def create_yolo_dataset():
    """
    Converts COCO format annotations to YOLO format and creates
    the proper directory structure for YOLOv8 training.
    """
    project_root = Path(__file__).parent
    archive_dir = project_root / 'archive'
    new_dataset_dir = project_root / 'dataset'

    print("--- Converting COCO to YOLO Format ---")

    # 1. Create new directory structure
    print(f"Creating dataset directory at: {new_dataset_dir}")
    if new_dataset_dir.exists():
        shutil.rmtree(new_dataset_dir)
    
    train_images_dir = new_dataset_dir / 'images' / 'train'
    train_labels_dir = new_dataset_dir / 'labels' / 'train'
    val_images_dir = new_dataset_dir / 'images' / 'val'
    val_labels_dir = new_dataset_dir / 'labels' / 'val'
    
    train_images_dir.mkdir(parents=True)
    train_labels_dir.mkdir(parents=True)
    val_images_dir.mkdir(parents=True)
    val_labels_dir.mkdir(parents=True)

    # YOLO class mapping (0-indexed)
    class_mapping = {
        'damage': 0,
        'headlamp': 1,
        'front_bumper': 2,
        'hood': 3,
        'door': 4,
        'rear_bumper': 5
    }

    def convert_bbox_to_yolo(bbox, img_width, img_height):
        """Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized)"""
        x, y, w, h = bbox
        
        # Calculate center coordinates
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Normalize by image dimensions
        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height
        
        return [x_center, y_center, w, h]

    def process_split(split_name):
        """Process train or val split"""
        print(f"\nProcessing {split_name} split...")
        
        # Load COCO annotations
        damage_json = archive_dir / split_name / f'COCO_{split_name}_annos.json'
        parts_json = archive_dir / split_name / f'COCO_mul_{split_name}_annos.json'
        
        damage_data = json.load(open(damage_json))
        parts_data = json.load(open(parts_json))
        
        # Create image ID to details mapping
        damage_images = {img['id']: img for img in damage_data['images']}
        parts_images = {img['id']: img for img in parts_data['images']}
        
        # Combine all images
        all_images = {**damage_images, **parts_images}
        
        # Process each image
        for img_id, img_details in all_images.items():
            filename = img_details['file_name']
            img_width = img_details['width']
            img_height = img_details['height']
            
            # Create YOLO label file
            label_filename = filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_path = train_labels_dir / label_filename if split_name == 'train' else val_labels_dir / label_filename
            
            yolo_annotations = []
            
            # Process damage annotations
            for ann in damage_data['annotations']:
                if ann['image_id'] == img_id:
                    bbox = convert_bbox_to_yolo(ann['bbox'], img_width, img_height)
                    yolo_annotations.append(f"0 {' '.join(map(str, bbox))}")
            
            # Process parts annotations
            for ann in parts_data['annotations']:
                if ann['image_id'] == img_id:
                    # Map original category ID to our class mapping
                    original_cat_id = ann['category_id']
                    # Assuming original categories are 1-5 for parts
                    if original_cat_id in [1, 2, 3, 4, 5]:
                        yolo_class_id = original_cat_id  # 1-5 map to our 1-5
                        bbox = convert_bbox_to_yolo(ann['bbox'], img_width, img_height)
                        yolo_annotations.append(f"{yolo_class_id} {' '.join(map(str, bbox))}")
            
            # Write YOLO label file
            if yolo_annotations:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            
            # Copy image file
            img_source = archive_dir / 'img' / filename
            img_dest = train_images_dir / filename if split_name == 'train' else val_images_dir / filename
            shutil.copy(img_source, img_dest)
        
        return len(all_images)

    # Process both splits
    train_count = process_split('train')
    val_count = process_split('val')
    
    print(f"\n--- Dataset Conversion Complete! ---")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Dataset ready at: {new_dataset_dir}")
    print("YOLO format labels created in labels/ directory")

if __name__ == '__main__':
    create_yolo_dataset() 