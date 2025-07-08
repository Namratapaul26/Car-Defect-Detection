from ultralytics import YOLO

def train_model():
    """
    Trains a YOLOv8 model on the custom car defect dataset with
    aggressive data augmentation to improve generalization.
    """
    print("--- Initializing Model Training with Augmentation ---")

    # Load a pre-trained YOLOv8 model. 'yolov8n.pt' is small and fast.
    # The model will be downloaded automatically if not present.
    model = YOLO('yolov8n.pt')

    # Train the model with the dataset and extensive augmentation.
    # We increase epochs to 100 to give the model time to learn from augmentations.
    print("Starting training... This will take longer due to more epochs and augmentation.")
    results = model.train(
        data='car_defects.yaml', 
        epochs=100, 
        imgsz=640,
        # --- Augmentation Parameters ---
        degrees=10,      # random rotation (-10, +10 degrees)
        translate=0.1,   # random translation (-10%, +10%)
        scale=0.2,       # random scale (-20%, +20%)
        fliplr=0.5,      # 50% chance of horizontal flip
        mosaic=1.0,      # 100% chance of applying mosaic augmentation
        mixup=0.1,       # 10% chance of applying mixup augmentation
        # --- Key change for auto-splitting ---
        split='val',     # Explicitly use the 'val' split for validation metrics
        val=True         # Ensure validation is run
    )

    print("--- Model Training Complete! ---")
    print("A new, more robust model has been trained.")
    print("Results are saved in the 'runs' directory (likely in a new 'train' folder).")
    print("Find the best model at 'runs/detect/train*/weights/best.pt'")

if __name__ == '__main__':
    train_model() 