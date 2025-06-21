#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage of the Volleyball Hawkeye System
This script demonstrates how to use the system for different scenarios
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from volleyball_hawkeye import VolleyballHawkeye
from calibration_tools import VolleyballCalibrationTools

def example_calibration():
    """Example of camera calibration process"""
    print("=== Camera Calibration Example ===")
    
    # Initialize calibration tools
    calibration_tools = VolleyballCalibrationTools()
    
    # Create calibration image template
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    calibration_image_path = output_dir / "calibration_template.jpg"
    calibration_tools.create_calibration_image(str(calibration_image_path))
    print(f"Calibration template created: {calibration_image_path}")
    
    # Note: In a real scenario, you would:
    # 1. Print the calibration template
    # 2. Place it on the volleyball court
    # 3. Take photos from different camera angles
    # 4. Use select_reference_points() to mark the points
    print("Next steps:")
    print("1. Print the calibration template")
    print("2. Place it on the volleyball court")
    print("3. Take photos from different camera angles")
    print("4. Run the calibration process")
    print("\nCourt Structure (FIVB Standards):")
    print("• Playing Court: 18m × 9m")
    print("• Free Zone: 5m sidelines, 6.5m end lines")
    print("• Net Height: 2.43m (men), 2.24m (women)")
    print("• Attack Lines: 3m from center line")
    print("• Service Zones: 9m wide behind end lines")
    print("• Free Playing Space: 12.5m minimum height")

def example_analysis():
    """Example of trajectory analysis"""
    print("\n=== Trajectory Analysis Example ===")
    
    # Initialize Hawkeye system
    hawkeye = VolleyballHawkeye()
    
    # Create sample 3D trajectory (simulated data)
    import numpy as np
    
    # Simulate a volleyball trajectory
    t = np.linspace(0, 2, 60)  # 2 seconds, 30 fps
    x = 5 * np.sin(t)  # Horizontal movement
    y = 2 * np.cos(t)  # Side movement
    z = 3 * np.exp(-t) * np.sin(3 * t)  # Vertical movement with bounces
    
    trajectory_3d = [np.array([x[i], y[i], z[i]]) for i in range(len(t))]
    
    # Analyze trajectory
    analysis = hawkeye.analyze_trajectory(trajectory_3d)
    
    print("Trajectory Analysis Results:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Visualize trajectory
    hawkeye.visualize_trajectory(trajectory_3d, analysis)
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    hawkeye.ball_3d_trajectory = trajectory_3d
    hawkeye.save_results(str(output_dir / "example_analysis.json"))
    print(f"Results saved to {output_dir / 'example_analysis.json'}")

def example_realtime_simulation():
    """Example of real-time analysis simulation"""
    print("\n=== Real-time Analysis Simulation ===")
    
    # Initialize Hawkeye system
    hawkeye = VolleyballHawkeye()
    
    # Simulate ball detections from multiple cameras
    import numpy as np
    import time
    
    print("Simulating real-time ball detection...")
    print("Press Ctrl+C to stop")
    print("\nCourt Boundaries (FIVB Standards):")
    print(f"• Playing Court: ±{hawkeye.court_length/2:.1f}m × ±{hawkeye.court_width/2:.1f}m")
    print(f"• Free Zone: ±{hawkeye.court_length/2 + hawkeye.free_zone_endline:.1f}m × ±{hawkeye.court_width/2 + hawkeye.free_zone_sideline:.1f}m")
    print(f"• Height Limit: {hawkeye.free_playing_space:.1f}m")
    
    try:
        for frame in range(100):  # Simulate 100 frames
            # Simulate ball detections from 2 cameras
            camera_1_detection = [np.array([300 + frame*2, 200 + np.sin(frame/10)*50, 0.8])]
            camera_2_detection = [np.array([400 + frame*2, 250 + np.sin(frame/10)*50, 0.9])]
            
            hawkeye.ball_tracks = {
                'camera_0': [camera_1_detection],
                'camera_1': [camera_2_detection]
            }
            
            # Simulate processing time
            time.sleep(0.033)  # ~30 fps
            
            if frame % 30 == 0:
                print(f"Processed {frame} frames")
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    print("Real-time simulation completed")

def example_custom_analysis():
    """Example of custom trajectory analysis"""
    print("\n=== Custom Analysis Example ===")
    
    # Initialize Hawkeye system
    hawkeye = VolleyballHawkeye()
    
    # Create a custom trajectory (spike shot)
    import numpy as np
    
    # Simulate a spike trajectory
    t = np.linspace(0, 1.5, 45)  # 1.5 seconds, 30 fps
    
    # Spike characteristics: high arc, fast descent
    x = 8 * (1 - np.exp(-2*t))  # Forward movement
    y = 0.5 * np.sin(2*np.pi*t)  # Slight side movement
    z = 4 * np.exp(-t) * np.sin(2*np.pi*t)  # High arc with bounce
    
    trajectory_3d = [np.array([x[i], y[i], z[i]]) for i in range(len(t))]
    
    # Analyze trajectory
    analysis = hawkeye.analyze_trajectory(trajectory_3d)
    
    print("Spike Shot Analysis:")
    print(f"  Trajectory Type: {analysis.get('trajectory_type', 'Unknown')}")
    print(f"  Max Height: {analysis.get('max_height', 0):.2f} meters")
    print(f"  Average Speed: {analysis.get('average_speed', 0):.2f} m/s")
    print(f"  In Bounds: {analysis.get('is_in_bounds', True)}")
    print(f"  Bounces: {len(analysis.get('bounces', []))}")
    
    # Check if it's a good spike
    if analysis.get('max_height', 0) > 3.0 and analysis.get('average_speed', 0) > 10.0:
        print("  Assessment: Excellent spike!")
    elif analysis.get('max_height', 0) > 2.5 and analysis.get('average_speed', 0) > 8.0:
        print("  Assessment: Good spike")
    else:
        print("  Assessment: Needs improvement")

def example_court_zones():
    """Example of court zone analysis"""
    print("\n=== Court Zone Analysis Example ===")
    
    # Initialize Hawkeye system
    hawkeye = VolleyballHawkeye()
    
    print("Volleyball Court Zones (FIVB Standards):")
    print(f"• Playing Court: {hawkeye.court_length}m × {hawkeye.court_width}m")
    print(f"• Front Zone: Between center line and attack lines (±{hawkeye.attack_line_distance}m)")
    print(f"• Back Zone: Between attack lines and end lines")
    print(f"• Service Zones: {hawkeye.service_zone_width}m wide behind end lines")
    print(f"• Substitution Zone: {hawkeye.substitution_zone_width}m between attack line extensions")
    print(f"• Free Zone: {hawkeye.free_zone_sideline}m sidelines, {hawkeye.free_zone_endline}m end lines")
    print(f"• Free Playing Space: {hawkeye.free_playing_space}m minimum height")
    print(f"• Net Height: {hawkeye.net_height}m (men's height)")
    print(f"• Antenna Height: {hawkeye.net_height + hawkeye.antenna_height}m")
    print(f"• Line Width: {hawkeye.line_width}m (5cm)")
    
    # Create sample trajectories in different zones
    import numpy as np
    
    # Front zone trajectory (spike)
    t = np.linspace(0, 1.0, 30)
    front_zone_traj = [np.array([-1.0 + 2*t[i], 0, 3*np.exp(-t[i])]) for i in range(len(t))]
    
    # Back zone trajectory (serve)
    back_zone_traj = [np.array([-6.0 + 8*t[i], 0, 2*np.exp(-t[i])]) for i in range(len(t))]
    
    # Service zone trajectory
    service_zone_traj = [np.array([-9.0 + 10*t[i], 0, 1.5*np.exp(-t[i])]) for i in range(len(t))]
    
    print("\nZone-specific trajectory analysis:")
    print("Front Zone (Spike):")
    front_analysis = hawkeye.analyze_trajectory(front_zone_traj)
    print(f"  Max Height: {front_analysis.get('max_height', 0):.2f}m")
    print(f"  Type: {front_analysis.get('trajectory_type', 'Unknown')}")
    
    print("Back Zone (Serve):")
    back_analysis = hawkeye.analyze_trajectory(back_zone_traj)
    print(f"  Max Height: {back_analysis.get('max_height', 0):.2f}m")
    print(f"  Type: {back_analysis.get('trajectory_type', 'Unknown')}")
    
    print("Service Zone:")
    service_analysis = hawkeye.analyze_trajectory(service_zone_traj)
    print(f"  Max Height: {service_analysis.get('max_height', 0):.2f}m")
    print(f"  Type: {service_analysis.get('trajectory_type', 'Unknown')}")

def main():
    """Run all examples"""
    print("Volleyball Hawkeye System - Example Usage")
    print("Updated to FIVB Standards")
    print("=" * 50)
    
    # Run examples
    example_calibration()
    example_analysis()
    example_realtime_simulation()
    example_custom_analysis()
    example_court_zones()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo run the full system:")
    print("1. python src/run_hawkeye.py --mode calibrate --create_calibration_image")
    print("2. python src/run_hawkeye.py --mode analyze --cameras video1.mp4 video2.mp4")
    print("3. python src/run_hawkeye.py --mode realtime --cameras 0 1")
    print("\nCourt Structure Reference:")
    print("• Playing Court: 18m × 9m rectangle")
    print("• Free Zone: 5m sidelines, 6.5m end lines")
    print("• Net: 2.43m height (men), 2.24m (women)")
    print("• Attack Lines: 3m from center line")
    print("• Service Zones: 9m wide behind end lines")
    print("• Free Playing Space: 12.5m minimum height")

if __name__ == "__main__":
    main() 