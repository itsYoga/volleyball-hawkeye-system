#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Volleyball Hawkeye System
This script tests the main components of the system
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from volleyball_hawkeye import VolleyballHawkeye
        print("✓ VolleyballHawkeye imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VolleyballHawkeye: {e}")
        return False
    
    try:
        from calibration_tools import VolleyballCalibrationTools
        print("✓ VolleyballCalibrationTools imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VolleyballCalibrationTools: {e}")
        return False
    
    return True

def test_hawkeye_initialization():
    """Test Hawkeye system initialization"""
    print("\nTesting Hawkeye system initialization...")
    
    try:
        from volleyball_hawkeye import VolleyballHawkeye
        hawkeye = VolleyballHawkeye()
        print("✓ Hawkeye system initialized successfully")
        
        # Test court reference points
        if hasattr(hawkeye, 'court_reference_points'):
            print(f"✓ Court reference points: {len(hawkeye.court_reference_points)} points")
        else:
            print("✗ Court reference points not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize Hawkeye system: {e}")
        return False

def test_calibration_tools():
    """Test calibration tools"""
    print("\nTesting calibration tools...")
    
    try:
        from calibration_tools import VolleyballCalibrationTools
        calibration_tools = VolleyballCalibrationTools()
        print("✓ Calibration tools initialized successfully")
        
        # Test reference points
        if hasattr(calibration_tools, 'court_reference_points'):
            print(f"✓ Calibration reference points: {len(calibration_tools.court_reference_points)} points")
        else:
            print("✗ Calibration reference points not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize calibration tools: {e}")
        return False

def test_ball_detection():
    """Test ball detection with a dummy image"""
    print("\nTesting ball detection...")
    
    try:
        from volleyball_hawkeye import VolleyballHawkeye
        hawkeye = VolleyballHawkeye()
        
        # Create a dummy image
        dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Test ball detection
        detections = hawkeye.detect_ball(dummy_image, "test_camera")
        print(f"✓ Ball detection completed: {len(detections)} detections")
        
        return True
        
    except Exception as e:
        print(f"✗ Ball detection failed: {e}")
        return False

def test_trajectory_analysis():
    """Test trajectory analysis with dummy data"""
    print("\nTesting trajectory analysis...")
    
    try:
        from volleyball_hawkeye import VolleyballHawkeye
        hawkeye = VolleyballHawkeye()
        
        # Create dummy 3D trajectory
        t = np.linspace(0, 2, 60)
        x = 5 * np.sin(t)
        y = 2 * np.cos(t)
        z = 0.1 + 3 * np.exp(-t) * np.sin(3 * t) # Minimum height 0.1m
        
        trajectory_3d = [np.array([x[i], y[i], z[i]]) for i in range(len(t))]
        
        # Test trajectory analysis
        analysis = hawkeye.analyze_trajectory(trajectory_3d)
        
        if analysis:
            print("✓ Trajectory analysis completed successfully")
            print(f"  - Max height: {analysis.get('max_height', 0):.2f}m")
            print(f"  - Trajectory type: {analysis.get('trajectory_type', 'unknown')}")
            print(f"  - In bounds: {analysis.get('is_in_bounds', True)}")
        else:
            print("✗ Trajectory analysis returned empty results")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Trajectory analysis failed: {e}")
        return False

def test_3d_reconstruction():
    """Test 3D reconstruction with dummy data"""
    print("\nTesting 3D reconstruction...")
    
    try:
        from volleyball_hawkeye import VolleyballHawkeye
        hawkeye = VolleyballHawkeye()
        
        # Create dummy ball tracks from multiple cameras
        ball_tracks = {
            'camera_0': [[np.array([300, 200, 0.8])] for _ in range(30)],
            'camera_1': [[np.array([400, 250, 0.9])] for _ in range(30)]
        }
        
        # Create dummy timestamps
        timestamps = [i / 30.0 for i in range(30)]
        
        # Test 3D reconstruction
        trajectory_3d = hawkeye.reconstruct_3d_trajectory(ball_tracks, timestamps)
        
        if trajectory_3d:
            print(f"✓ 3D reconstruction completed: {len(trajectory_3d)} points")
        else:
            print("✗ 3D reconstruction returned empty results")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 3D reconstruction failed: {e}")
        return False

def test_calibration_image_creation():
    """Test calibration image creation"""
    print("\nTesting calibration image creation...")
    
    try:
        from calibration_tools import VolleyballCalibrationTools
        calibration_tools = VolleyballCalibrationTools()
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Create calibration image
        calibration_image_path = output_dir / "test_calibration.jpg"
        calibration_tools.create_calibration_image(str(calibration_image_path))
        
        if calibration_image_path.exists():
            print(f"✓ Calibration image created: {calibration_image_path}")
        else:
            print("✗ Calibration image creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Calibration image creation failed: {e}")
        return False

def test_data_saving():
    """Test data saving functionality"""
    print("\nTesting data saving...")
    
    try:
        from volleyball_hawkeye import VolleyballHawkeye
        hawkeye = VolleyballHawkeye()
        
        # Create dummy trajectory
        trajectory_3d = [np.array([1.0, 2.0, 3.0]) for _ in range(10)]
        hawkeye.ball_3d_trajectory = trajectory_3d
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Test saving results
        results_path = output_dir / "test_results.json"
        hawkeye.save_results(str(results_path))
        
        if results_path.exists():
            print(f"✓ Results saved: {results_path}")
        else:
            print("✗ Results saving failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data saving failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Volleyball Hawkeye System - Component Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_hawkeye_initialization,
        test_calibration_tools,
        test_ball_detection,
        test_trajectory_analysis,
        test_3d_reconstruction,
        test_calibration_image_creation,
        test_data_saving
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    # Clean up test files
    try:
        import shutil
        if Path("test_output").exists():
            shutil.rmtree("test_output")
            print("✓ Test files cleaned up")
    except Exception as e:
        print(f"⚠️  Failed to clean up test files: {e}")

if __name__ == "__main__":
    main() 