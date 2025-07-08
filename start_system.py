#!/usr/bin/env python3
"""
Startup Script for Car Defect Detection System
Easy launcher for different components of the system.
"""

import os
import sys
import subprocess
import argparse
import time

def print_banner():
    """Print system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚗 Car Defect Detection System 🚗         ║
    ║                                                              ║
    ║  A comprehensive real-time car defect detection system      ║
    ║  using YOLOv8 and Streamlit.                                ║
    ║                                                              ║
    ║  Features: Real-time webcam detection with start/stop       ║
    ║  controls, image upload analysis, and model training.       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('ultralytics', 'ultralytics'), 
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('Pillow', 'PIL'),
        ('pycocotools', 'pycocotools')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_model():
    """Check if trained model exists"""
    model_paths = [
        'runs/detect/train/weights/best.pt',
        'runs/detect/train2/weights/best.pt',
        'runs/detect/train3/weights/best.pt',
        'runs/detect/train4/weights/best.pt',
        'runs/detect/train5/weights/best.pt',
        'runs/detect/train6/weights/best.pt'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found trained model: {path}")
            return path
    
    print("⚠️  No trained model found. Will use pre-trained model.")
    return None

def start_basic_app():
    """Start the basic Streamlit app"""
    print("🚀 Starting Streamlit app...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

def run_dataset_preparation():
    """Run dataset preparation"""
    print("📊 Preparing dataset...")
    try:
        subprocess.run([sys.executable, "prepare_dataset.py"], check=True)
        print("✅ Dataset preparation completed!")
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")

def run_model_training():
    """Run model training"""
    print("🏋️  Starting model training...")
    try:
        subprocess.run([sys.executable, "train.py"], check=True)
        print("✅ Model training completed!")
    except Exception as e:
        print(f"❌ Error training model: {e}")

def run_model_testing():
    """Run model testing"""
    model_path = check_model()
    if not model_path:
        print("❌ No trained model found. Please train a model first.")
        return
    
    print("🧪 Starting model testing...")
    try:
        # Test on a sample image if available
        test_image = "archive/img/1.jpg"  # Adjust path as needed
        if os.path.exists(test_image):
            subprocess.run([
                sys.executable, "test_model.py", 
                "--model", model_path, 
                "--image", test_image,
                "--report"
            ], check=True)
        else:
            print("⚠️  No test image found. Use --image or --batch options manually.")
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def show_system_status():
    """Show system status"""
    print("\n📊 System Status")
    print("=" * 50)
    
    # Check model
    model_path = check_model()
    
    # Check dataset
    dataset_exists = os.path.exists("dataset")
    print(f"Dataset: {'✅' if dataset_exists else '❌'}")
    
    # Check requirements
    req_exists = os.path.exists("requirements.txt")
    print(f"Requirements: {'✅' if req_exists else '❌'}")
    
    # Check apps
    app_exists = os.path.exists("app.py")
    print(f"Main App: {'✅' if app_exists else '❌'}")
    
    # Check test script
    test_exists = os.path.exists("test_model.py")
    print(f"Test Script: {'✅' if test_exists else '❌'}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Car Defect Detection System Launcher - Real-time webcam detection with start/stop controls"
    )
    parser.add_argument("--mode", choices=["basic", "prepare", "train", "test", "status"], 
                       default="basic", help="Mode to run (basic: webcam with controls)")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        sys.exit(1)
    
    # Run selected mode
    if args.mode == "basic":
        start_basic_app()
    elif args.mode == "prepare":
        run_dataset_preparation()
    elif args.mode == "train":
        run_model_training()
    elif args.mode == "test":
        run_model_testing()
    elif args.mode == "status":
        show_system_status()

if __name__ == "__main__":
    main() 