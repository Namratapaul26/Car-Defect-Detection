"""
Model Testing Script for Car Defect Detection
Test your trained model on real car images and evaluate performance.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json
from PIL import Image
import argparse

class ModelTester:
    """Comprehensive model testing class"""
    
    def __init__(self, model_path=None):
        """Initialize the model tester"""
        self.model_path = model_path
        self.model = None
        self.results = []
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Loaded custom model: {self.model_path}")
            else:
                self.model = YOLO('yolov8n.pt')
                print("⚠️ Using pre-trained model (not custom)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True
    
    def test_single_image(self, image_path, save_result=True, output_dir="test_results"):
        """Test model on a single image"""
        if not self.model:
            print("❌ Model not loaded")
            return None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            # Run inference
            start_time = time.time()
            results = self.model(image)
            inference_time = time.time() - start_time
            
            result = results[0]
            
            # Extract detections
            detections = []
            confidence_scores = []
            bounding_boxes = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    class_name = result.names[class_id]
                    
                    detections.append(class_name)
                    confidence_scores.append(confidence)
                    bounding_boxes.append(bbox)
            
            # Create result dictionary
            result_data = {
                'image_path': image_path,
                'detections': detections,
                'confidence_scores': confidence_scores,
                'bounding_boxes': bounding_boxes,
                'inference_time': inference_time,
                'image_size': image.shape[:2]
            }
            
            # Save annotated image if requested
            if save_result:
                os.makedirs(output_dir, exist_ok=True)
                
                # Get annotated image
                annotated_img = result.plot()
                
                # Save image
                filename = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{filename}_result.jpg")
                cv2.imwrite(output_path, annotated_img)
                
                result_data['output_path'] = output_path
                print(f"✅ Result saved: {output_path}")
            
            self.results.append(result_data)
            return result_data
            
        except Exception as e:
            print(f"❌ Error testing image {image_path}: {e}")
            return None
    
    def test_batch(self, image_dir, save_results=True, output_dir="test_results"):
        """Test model on a batch of images"""
        if not os.path.exists(image_dir):
            print(f"❌ Image directory not found: {image_dir}")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return
        
        print(f"🔍 Found {len(image_files)} images to test")
        
        # Test each image
        for i, image_path in enumerate(image_files):
            print(f"Testing {i+1}/{len(image_files)}: {image_path.name}")
            self.test_single_image(str(image_path), save_results, output_dir)
    
    def analyze_results(self):
        """Analyze test results and generate statistics"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print("\n📊 Test Results Analysis")
        print("=" * 50)
        
        # Basic statistics
        total_images = len(self.results)
        total_detections = sum(len(r['detections']) for r in self.results)
        avg_inference_time = np.mean([r['inference_time'] for r in self.results])
        
        print(f"Total images tested: {total_images}")
        print(f"Total detections: {total_detections}")
        print(f"Average inference time: {avg_inference_time:.3f}s")
        print(f"Average FPS: {1/avg_inference_time:.1f}")
        
        # Detection statistics
        all_detections = []
        all_confidences = []
        
        for result in self.results:
            all_detections.extend(result['detections'])
            all_confidences.extend(result['confidence_scores'])
        
        if all_detections:
            # Count detections by class
            from collections import Counter
            detection_counts = Counter(all_detections)
            
            print(f"\nDetection breakdown:")
            for class_name, count in detection_counts.most_common():
                print(f"  {class_name}: {count}")
            
            print(f"\nConfidence statistics:")
            print(f"  Average confidence: {np.mean(all_confidences):.3f}")
            print(f"  Min confidence: {np.min(all_confidences):.3f}")
            print(f"  Max confidence: {np.max(all_confidences):.3f}")
        
        # Save detailed results
        results_file = "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed results saved to: {results_file}")
    
    def create_performance_report(self, output_file="performance_report.html"):
        """Create an HTML performance report"""
        if not self.results:
            print("❌ No results to report")
            return
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Car Defect Detection - Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .stats { display: flex; justify-content: space-around; margin: 20px 0; }
                .stat-box { background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }
                .results { margin: 20px 0; }
                .result-item { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .detection { background-color: #d4edda; padding: 5px; margin: 5px; border-radius: 3px; display: inline-block; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚗 Car Defect Detection - Performance Report</h1>
                <p>Model: """ + (self.model_path or "Pre-trained YOLOv8") + """</p>
            </div>
        """
        
        # Add statistics
        total_images = len(self.results)
        total_detections = sum(len(r['detections']) for r in self.results)
        avg_inference_time = np.mean([r['inference_time'] for r in self.results])
        
        html_content += f"""
            <div class="stats">
                <div class="stat-box">
                    <h3>Images Tested</h3>
                    <h2>{total_images}</h2>
                </div>
                <div class="stat-box">
                    <h3>Total Detections</h3>
                    <h2>{total_detections}</h2>
                </div>
                <div class="stat-box">
                    <h3>Avg Inference Time</h3>
                    <h2>{avg_inference_time:.3f}s</h2>
                </div>
                <div class="stat-box">
                    <h3>Avg FPS</h3>
                    <h2>{1/avg_inference_time:.1f}</h2>
                </div>
            </div>
        """
        
        # Add individual results
        html_content += "<div class='results'><h2>Individual Results</h2>"
        
        for result in self.results:
            image_name = Path(result['image_path']).name
            detections_html = ""
            
            for detection, confidence in zip(result['detections'], result['confidence_scores']):
                detections_html += f'<span class="detection">{detection} ({confidence:.2f})</span>'
            
            html_content += f"""
                <div class="result-item">
                    <h3>{image_name}</h3>
                    <p><strong>Inference Time:</strong> {result['inference_time']:.3f}s</p>
                    <p><strong>Detections:</strong> {detections_html}</p>
                    <p><strong>Image Size:</strong> {result['image_size'][1]}x{result['image_size'][0]}</p>
                </div>
            """
        
        html_content += "</div></body></html>"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Performance report saved to: {output_file}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Test Car Defect Detection Model")
    parser.add_argument("--model", type=str, help="Path to trained model (.pt file)")
    parser.add_argument("--image", type=str, help="Path to single image to test")
    parser.add_argument("--batch", type=str, help="Path to directory of images to test")
    parser.add_argument("--output", type=str, default="test_results", help="Output directory for results")
    parser.add_argument("--no-save", action="store_true", help="Don't save annotated images")
    parser.add_argument("--report", action="store_true", help="Generate HTML performance report")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester(args.model)
    
    if args.image:
        print(f"🔍 Testing single image: {args.image}")
        result = tester.test_single_image(args.image, not args.no_save, args.output)
        if result:
            print(f"✅ Detections: {result['detections']}")
    
    elif args.batch:
        print(f"🔍 Testing batch of images: {args.batch}")
        tester.test_batch(args.batch, not args.no_save, args.output)
    
    else:
        print("❌ Please specify --image or --batch")
        return
    
    # Analyze results
    tester.analyze_results()
    
    # Generate report if requested
    if args.report:
        tester.create_performance_report()

if __name__ == "__main__":
    main() 