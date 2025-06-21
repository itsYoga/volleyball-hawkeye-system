#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volleyball Hawkeye System - Main Runner
Uses YOLOv11 exclusively
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np
import cv2

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from volleyball_hawkeye import VolleyballHawkeye
from calibration_tools import VolleyballCalibrationTools

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_calibration_mode(args):
    """Run calibration mode"""
    logging.info("Starting calibration mode...")
    
    # Initialize calibration tool
    calib_tool = VolleyballCalibrationTools()
    
    # Create calibration image template
    calib_image_path = Path(args.output).parent / "calibration_template.jpg"
    calib_tool.create_calibration_image(str(calib_image_path))
    
    logging.info(f"Calibration template created: {calib_image_path}")
    logging.info("Please use the calibration template to calibrate your cameras")
    
    return True

def run_analysis_mode(args):
    """Run analysis mode"""
    logging.info("Starting analysis mode...")
    
    # Initialize Hawkeye system
    try:
        hawkeye = VolleyballHawkeye(
            yolo_model_path=args.yolo_model
        )
        logging.info(f"Initialized Hawkeye system with YOLOv11")
    except Exception as e:
        logging.error(f"Failed to initialize Hawkeye system: {str(e)}")
        return False
    
    # Load camera matrices if provided
    if args.camera_matrices:
        success = hawkeye.load_camera_matrices(args.camera_matrices)
        if not success:
            logging.warning("Failed to load camera matrices, using dummy 3D reconstruction")
    
    # Process video or images
    if args.input:
        if args.input.endswith(('.mp4', '.avi', '.mov')):
            success = process_video(hawkeye, args.input, args.output)
        else:
            success = process_images(hawkeye, args.input, args.output)
    else:
        # Generate demo data
        success = generate_demo_analysis(hawkeye, args.output)
    
    return success

def run_realtime_mode(args):
    """Run real-time mode"""
    logging.info("Starting real-time mode...")
    
    # Initialize Hawkeye system
    try:
        hawkeye = VolleyballHawkeye(
            yolo_model_path=args.yolo_model
        )
        logging.info(f"Initialized Hawkeye system with YOLOv11")
    except Exception as e:
        logging.error(f"Failed to initialize Hawkeye system: {str(e)}")
        return False
    
    # Load camera matrices if provided
    if args.camera_matrices:
        hawkeye.load_camera_matrices(args.camera_matrices)
    
    # Start real-time processing
    return run_realtime_processing(hawkeye, args.camera_id)

def process_video(hawkeye, video_path: str, output_path: str) -> bool:
    """Process video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return False
        
        frame_count = 0
        ball_tracks = {"camera_1": []}
        frame_timestamps = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect ball
            detections = hawkeye.detect_ball(frame, "camera_1")
            ball_tracks["camera_1"].append(detections)
            frame_timestamps.append(frame_count / 30.0)  # Assuming 30 fps
            
            frame_count += 1
            
            # Process every 30 frames (1 second at 30 fps)
            if frame_count % 30 == 0:
                logging.info(f"Processed {frame_count} frames")
        
        cap.release()
        
        # Reconstruct 3D trajectory
        trajectory_3d = hawkeye.reconstruct_3d_trajectory(ball_tracks, frame_timestamps)
        
        # Analyze trajectory
        analysis = hawkeye.analyze_trajectory(trajectory_3d)
        
        # Save results
        hawkeye.save_results(output_path)
        
        # Visualize trajectory
        hawkeye.visualize_trajectory(trajectory_3d, analysis)
        
        logging.info(f"Video processing completed. Results saved to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Video processing failed: {str(e)}")
        return False

def process_images(hawkeye, input_path: str, output_path: str) -> bool:
    """Process image directory"""
    try:
        input_dir = Path(input_path)
        if not input_dir.exists():
            logging.error(f"Input directory does not exist: {input_path}")
            return False
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
        
        if not image_files:
            logging.error(f"No image files found in {input_path}")
            return False
        
        logging.info(f"Found {len(image_files)} images")
        
        ball_tracks = {"camera_1": []}
        frame_timestamps = []
        
        for i, image_file in enumerate(sorted(image_files)):
            # Load image
            frame = cv2.imread(str(image_file))
            if frame is None:
                continue
            
            # Detect ball
            detections = hawkeye.detect_ball(frame, "camera_1")
            ball_tracks["camera_1"].append(detections)
            frame_timestamps.append(i / 30.0)  # Assuming 30 fps
        
        # Reconstruct 3D trajectory
        trajectory_3d = hawkeye.reconstruct_3d_trajectory(ball_tracks, frame_timestamps)
        
        # Analyze trajectory
        analysis = hawkeye.analyze_trajectory(trajectory_3d)
        
        # Save results
        hawkeye.save_results(output_path)
        
        # Visualize trajectory
        hawkeye.visualize_trajectory(trajectory_3d, analysis)
        
        logging.info(f"Image processing completed. Results saved to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Image processing failed: {str(e)}")
        return False

def generate_demo_analysis(hawkeye, output_path: str) -> bool:
    """Generate demo analysis with simulated data"""
    try:
        logging.info("Generating demo analysis...")
        
        # Create simulated ball tracks
        ball_tracks = {"camera_1": []}
        frame_timestamps = []
        
        # Simulate a volleyball serve trajectory
        for i in range(60):  # 2 seconds at 30 fps
            # Simulate ball movement
            t = i / 30.0
            x = 2.0 * np.sin(t * 2)  # Horizontal movement
            y = 1.5 * np.cos(t * 1.5)  # Vertical movement
            z = 3.0 * np.exp(-t) * np.sin(t * 3)  # Height with bounce
            
            # Add some noise
            x += np.random.normal(0, 0.1)
            y += np.random.normal(0, 0.1)
            z += np.random.normal(0, 0.05)
            
            # Convert to 2D detection (simplified)
            detection = np.array([320 + x * 50, 240 + y * 30, 0.8])  # Center of frame
            ball_tracks["camera_1"].append([detection])
            frame_timestamps.append(t)
        
        # Reconstruct 3D trajectory
        trajectory_3d = hawkeye.reconstruct_3d_trajectory(ball_tracks, frame_timestamps)
        
        # Analyze trajectory
        analysis = hawkeye.analyze_trajectory(trajectory_3d)
        
        # Save results
        hawkeye.save_results(output_path)
        
        # Visualize trajectory
        hawkeye.visualize_trajectory(trajectory_3d, analysis)
        
        logging.info(f"Demo analysis completed. Results saved to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Demo analysis failed: {str(e)}")
        return False

def run_realtime_processing(hawkeye, camera_id: int = 0) -> bool:
    """Run real-time processing"""
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Failed to open camera {camera_id}")
            return False
        
        logging.info("Starting real-time processing. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect ball
            detections = hawkeye.detect_ball(frame, f"camera_{camera_id}")
            
            # Draw detections
            for detection in detections:
                x, y, confidence = detection
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)
                cv2.putText(frame, f"Ball: {confidence:.2f}", 
                           (int(x) + 15, int(y) - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show frame
            cv2.imshow("Volleyball Hawkeye - Real-time", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        logging.error(f"Real-time processing failed: {str(e)}")
        return False

def upgrade_to_yolov11():
    """Upgrade system to YOLOv11"""
    logging.info("Checking YOLOv11 availability...")
    
    try:
        # Try to upgrade
        hawkeye = VolleyballHawkeye(yolo_model_path="yolov11n.pt")
        success = True
        
        if success:
            logging.info("✅ Successfully upgraded to YOLOv11!")
            return True
        else:
            logging.warning("❌ YOLOv11 not yet available, continuing with YOLOv11")
            return False
            
    except Exception as e:
        logging.error(f"Upgrade failed: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Volleyball Hawkeye System - Multi-camera 3D ball tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibration mode
  python run_hawkeye.py --mode calibration --output calibration_data.pkl
  
  # Analysis mode with YOLOv11
  python run_hawkeye.py --mode analyze --input video.mp4 --output results.json
  
  # Real-time mode
  python run_hawkeye.py --mode realtime --camera-id 0
  
  # Upgrade to YOLOv11
  python run_hawkeye.py --upgrade-yolov11
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", choices=["calibration", "analyze", "realtime"], 
                       default="analyze", help="Operation mode")
    
    # YOLO configuration
    parser.add_argument("--yolo-model", default="yolo11m.pt",
                       help="YOLOv11 model path")
    
    # Input/Output
    parser.add_argument("--input", help="Input video or image directory")
    parser.add_argument("--output", default="hawkeye_results.json",
                       help="Output file path")
    
    # Calibration options
    parser.add_argument("--num-cameras", type=int, default=2,
                       help="Number of cameras for calibration")
    parser.add_argument("--camera-matrices", help="Path to camera matrices file")
    
    # Real-time options
    parser.add_argument("--camera-id", type=int, default=0,
                       help="Camera ID for real-time mode")
    
    # Upgrade options
    parser.add_argument("--upgrade-yolov11", action="store_true",
                       help="Upgrade system to YOLOv11")
    
    # Other options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle upgrade
    if args.upgrade_yolov11:
        success = upgrade_to_yolov11()
        return 0 if success else 1
    
    # Run selected mode
    if args.mode == "calibration":
        success = run_calibration_mode(args)
    elif args.mode == "analyze":
        success = run_analysis_mode(args)
    elif args.mode == "realtime":
        success = run_realtime_mode(args)
    else:
        logging.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 