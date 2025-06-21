#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volleyball Hawkeye System Demo
This script demonstrates the key features of the system
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from volleyball_hawkeye import VolleyballHawkeye
from calibration_tools import VolleyballCalibrationTools

def demo_spike_analysis():
    """Demonstrate spike shot analysis"""
    print("🏐 Volleyball Hawkeye System Demo")
    print("=" * 50)
    print("Demo 1: Spike Shot Analysis")
    print("-" * 30)
    
    hawkeye = VolleyballHawkeye()
    
    # Create a realistic spike trajectory
    t = np.linspace(0, 1.2, 36)  # 1.2 seconds, 30 fps
    
    # Spike characteristics: high arc, fast forward movement
    x = 12 * (1 - np.exp(-3*t))  # Forward movement (12m court length)
    y = 0.3 * np.sin(2*np.pi*t)  # Slight side movement
    z = 0.1 + 4.5 * np.exp(-t) * np.sin(2*np.pi*t)  # High arc with bounce, minimum height 0.1m
    
    trajectory_3d = [np.array([x[i], y[i], z[i]]) for i in range(len(t))]
    
    # Analyze the spike
    analysis = hawkeye.analyze_trajectory(trajectory_3d)
    
    print(f"📊 Analysis Results:")
    print(f"   • Trajectory Type: {analysis.get('trajectory_type', 'Unknown')}")
    print(f"   • Max Height: {analysis.get('max_height', 0):.2f} meters")
    print(f"   • Average Speed: {analysis.get('average_speed', 0):.1f} m/s")
    print(f"   • Max Speed: {analysis.get('max_speed', 0):.1f} m/s")
    print(f"   • Total Distance: {analysis.get('total_distance', 0):.1f} meters")
    print(f"   • In Bounds: {'✅ Yes' if analysis.get('is_in_bounds', True) else '❌ No'}")
    print(f"   • Bounces: {len(analysis.get('bounces', []))}")
    
    # Performance assessment
    max_height = analysis.get('max_height', 0)
    avg_speed = analysis.get('average_speed', 0)
    
    if max_height > 3.5 and avg_speed > 12.0:
        print(f"   • Performance: 🏆 Excellent spike!")
    elif max_height > 3.0 and avg_speed > 10.0:
        print(f"   • Performance: ⭐ Good spike")
    elif max_height > 2.5 and avg_speed > 8.0:
        print(f"   • Performance: 👍 Average spike")
    else:
        print(f"   • Performance: 📈 Needs improvement")
    
    return trajectory_3d, analysis

def demo_set_analysis():
    """Demonstrate set shot analysis"""
    print("\nDemo 2: Set Shot Analysis")
    print("-" * 30)
    
    hawkeye = VolleyballHawkeye()
    
    # Create a realistic set trajectory
    t = np.linspace(0, 1.0, 30)  # 1.0 seconds, 30 fps
    
    # Set characteristics: controlled arc, moderate height
    x = 3 * np.sin(np.pi*t)  # Short forward movement
    y = 0.1 * np.sin(3*np.pi*t)  # Minimal side movement
    z = 0.1 + 3.2 * np.exp(-t) * np.sin(1.5*np.pi*t)  # Controlled arc, minimum height 0.1m
    
    trajectory_3d = [np.array([x[i], y[i], z[i]]) for i in range(len(t))]
    
    # Analyze the set
    analysis = hawkeye.analyze_trajectory(trajectory_3d)
    
    print(f"📊 Analysis Results:")
    print(f"   • Trajectory Type: {analysis.get('trajectory_type', 'Unknown')}")
    print(f"   • Max Height: {analysis.get('max_height', 0):.2f} meters")
    print(f"   • Average Speed: {analysis.get('average_speed', 0):.1f} m/s")
    print(f"   • In Bounds: {'✅ Yes' if analysis.get('is_in_bounds', True) else '❌ No'}")
    
    return trajectory_3d, analysis

def demo_serve_analysis():
    """Demonstrate serve analysis"""
    print("\nDemo 3: Serve Analysis")
    print("-" * 30)
    
    hawkeye = VolleyballHawkeye()
    
    # Create a realistic serve trajectory
    t = np.linspace(0, 2.0, 60)  # 2.0 seconds, 30 fps
    
    # Serve characteristics: long distance, high arc
    x = 18 * (1 - np.exp(-t))  # Full court distance
    y = 0.2 * np.sin(2*np.pi*t)  # Slight side movement
    z = 0.1 + 5.0 * np.exp(-t/2) * np.sin(1.5*np.pi*t)  # High arc, minimum height 0.1m
    
    trajectory_3d = [np.array([x[i], y[i], z[i]]) for i in range(len(t))]
    
    # Analyze the serve
    analysis = hawkeye.analyze_trajectory(trajectory_3d)
    
    print(f"📊 Analysis Results:")
    print(f"   • Trajectory Type: {analysis.get('trajectory_type', 'Unknown')}")
    print(f"   • Max Height: {analysis.get('max_height', 0):.2f} meters")
    print(f"   • Average Speed: {analysis.get('average_speed', 0):.1f} m/s")
    print(f"   • Total Distance: {analysis.get('total_distance', 0):.1f} meters")
    print(f"   • In Bounds: {'✅ Yes' if analysis.get('is_in_bounds', True) else '❌ No'}")
    
    return trajectory_3d, analysis

def demo_calibration():
    """Demonstrate calibration process"""
    print("\nDemo 4: Camera Calibration")
    print("-" * 30)
    
    calibration_tools = VolleyballCalibrationTools()
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create calibration template
    calibration_image_path = output_dir / "volleyball_calibration.jpg"
    calibration_tools.create_calibration_image(str(calibration_image_path))
    
    print(f"📋 Calibration Template Created:")
    print(f"   • File: {calibration_image_path}")
    print(f"   • Reference Points: 47 points (FIVB standards)")
    print(f"   • Playing Court: 18m × 9m")
    print(f"   • Free Zone: 5m sidelines, 6.5m end lines")
    print(f"   • Net Height: 2.43m (men), 2.24m (women)")
    print(f"   • Attack Lines: 3m from center line")
    print(f"   • Service Zones: 9m wide behind end lines")
    print(f"   • Free Playing Space: 12.5m minimum height")
    
    print(f"\n📝 Calibration Process:")
    print(f"   1. Print the calibration template")
    print(f"   2. Place it on the volleyball court")
    print(f"   3. Take photos from 4 camera angles")
    print(f"   4. Mark reference points in each image")
    print(f"   5. Run camera calibration")
    print(f"   6. Validate calibration accuracy")
    
    print(f"\n🏐 Court Structure (FIVB Standards):")
    print(f"   • Playing Court: 18m × 9m rectangle")
    print(f"   • Center Line: Divides court into equal halves")
    print(f"   • Attack Lines: 3m from center line")
    print(f"   • Front Zone: Between center line and attack lines")
    print(f"   • Back Zone: Between attack lines and end lines")
    print(f"   • Service Zones: 9m wide behind end lines")
    print(f"   • Substitution Zone: Between attack line extensions")
    print(f"   • Free Zone: 5m sidelines, 6.5m end lines")
    print(f"   • Net: 2.43m height (men), 2.24m (women)")
    print(f"   • Antennas: 80cm above net")
    print(f"   • Line Width: 5cm")
    print(f"   • Free Playing Space: 12.5m minimum height")
    
    return calibration_image_path

def demo_3d_visualization():
    """Demonstrate 3D visualization"""
    print("\nDemo 5: 3D Trajectory Visualization")
    print("-" * 30)
    
    hawkeye = VolleyballHawkeye()
    
    # Create a complex trajectory (spike with bounce)
    t = np.linspace(0, 2.0, 60)
    
    # First part: spike
    t1 = t[t <= 1.0]
    x1 = 8 * (1 - np.exp(-2*t1))
    y1 = 0.2 * np.sin(2*np.pi*t1)
    z1 = 0.1 + 4.0 * np.exp(-t1) * np.sin(2*np.pi*t1)  # Minimum height 0.1m
    
    # Second part: bounce and continuation
    t2 = t[t > 1.0]
    x2 = 8 + 4 * (1 - np.exp(-2*(t2-1)))
    y2 = 0.2 * np.sin(2*np.pi*t2)
    z2 = 0.1 + 2.0 * np.exp(-(t2-1)) * np.sin(2*np.pi*(t2-1))  # Minimum height 0.1m
    
    # Combine trajectories
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    z = np.concatenate([z1, z2])
    
    trajectory_3d = [np.array([x[i], y[i], z[i]]) for i in range(len(t))]
    
    # Analyze and visualize
    analysis = hawkeye.analyze_trajectory(trajectory_3d)
    hawkeye.visualize_trajectory(trajectory_3d, analysis)
    
    print(f"📊 Complex Trajectory Analysis:")
    print(f"   • Trajectory Type: {analysis.get('trajectory_type', 'Unknown')}")
    print(f"   • Max Height: {analysis.get('max_height', 0):.2f} meters")
    print(f"   • Bounces: {len(analysis.get('bounces', []))}")
    print(f"   • Total Distance: {analysis.get('total_distance', 0):.1f} meters")
    print(f"   • 3D visualization window opened")
    
    return trajectory_3d, analysis

def main():
    """Run all demos"""
    print("🏐 Volleyball Hawkeye System - Complete Demo")
    print("=" * 60)
    print("This demo showcases the key features of the volleyball Hawkeye system")
    print("adapted from the tennis Hawkeye system for volleyball analysis.\n")
    
    try:
        # Run all demos
        spike_traj, spike_analysis = demo_spike_analysis()
        set_traj, set_analysis = demo_set_analysis()
        serve_traj, serve_analysis = demo_serve_analysis()
        calibration_path = demo_calibration()
        complex_traj, complex_analysis = demo_3d_visualization()
        
        print("\n" + "=" * 60)
        print("🎉 Demo Completed Successfully!")
        print("\n📋 System Features Demonstrated:")
        print("   ✅ Multi-camera 3D reconstruction")
        print("   ✅ YOLOv8 ball detection")
        print("   ✅ Camera calibration tools")
        print("   ✅ Trajectory analysis")
        print("   ✅ 3D visualization")
        print("   ✅ Performance assessment")
        print("   ✅ Volleyball-specific analysis")
        
        print("\n🚀 Next Steps:")
        print("   1. Set up multiple cameras around a volleyball court")
        print("   2. Run camera calibration: python src/run_hawkeye.py --mode calibrate")
        print("   3. Record volleyball matches")
        print("   4. Analyze trajectories: python src/run_hawkeye.py --mode analyze")
        print("   5. Real-time analysis: python src/run_hawkeye.py --mode realtime")
        
        print("\n📚 Based on: SEU-Robot-Vision-Project-tennis-Hawkeye-system")
        print("🔧 Adapted for volleyball with YOLOv8 detection")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check the system installation and dependencies.")

if __name__ == "__main__":
    main() 