#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolleyballCalibrationTools:
    """
    Tools for calibrating volleyball Hawkeye system cameras
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reference_points_2d = {}
        self.selected_points = []
        self.current_image = None
        self.current_camera_id = None
        
        # Volleyball court reference points (same as in main system)
        self.court_reference_points = {
            # Playing court corners (18m x 9m)
            'corner_1': np.array([-9.0, -4.5, 0]),      # Back left
            'corner_2': np.array([9.0, -4.5, 0]),       # Back right
            'corner_3': np.array([9.0, 4.5, 0]),        # Front right
            'corner_4': np.array([-9.0, 4.5, 0]),       # Front left
            
            # Net posts (at center line)
            'net_post_1': np.array([0, -4.5, 2.43]),    # Left post
            'net_post_2': np.array([0, 4.5, 2.43]),     # Right post
            
            # Net antennas (80cm above net)
            'antenna_1': np.array([0, -4.5, 3.23]),     # Left antenna
            'antenna_2': np.array([0, 4.5, 3.23]),      # Right antenna
            
            # Attack lines (3m from center line)
            'attack_line_1': np.array([-3.0, -4.5, 0]), # Left attack line
            'attack_line_2': np.array([-3.0, 4.5, 0]),  # Right attack line
            'attack_line_3': np.array([3.0, -4.5, 0]),  # Left attack line
            'attack_line_4': np.array([3.0, 4.5, 0]),   # Right attack line
            
            # Center line points
            'center_line_1': np.array([0, -4.5, 0]),     # Left center line
            'center_line_2': np.array([0, 4.5, 0]),      # Right center line
            
            # Service zones (9m wide behind end lines)
            'service_zone_1': np.array([-9.0, -4.5, 0]), # Left service zone
            'service_zone_2': np.array([-9.0, 4.5, 0]),  # Right service zone
            'service_zone_3': np.array([9.0, -4.5, 0]),  # Left service zone
            'service_zone_4': np.array([9.0, 4.5, 0]),   # Right service zone
            
            # Front zone boundaries (between center line and attack lines)
            'front_zone_1': np.array([-1.5, -4.5, 0]),  # Left front zone
            'front_zone_2': np.array([-1.5, 4.5, 0]),   # Right front zone
            'front_zone_3': np.array([1.5, -4.5, 0]),   # Left front zone
            'front_zone_4': np.array([1.5, 4.5, 0]),    # Right front zone
            
            # Back zone boundaries (between attack lines and end lines)
            'back_zone_1': np.array([-7.5, -4.5, 0]),   # Left back zone
            'back_zone_2': np.array([-7.5, 4.5, 0]),    # Right back zone
            'back_zone_3': np.array([7.5, -4.5, 0]),    # Left back zone
            'back_zone_4': np.array([7.5, 4.5, 0]),     # Right back zone
            
            # Substitution zone (between attack line extensions)
            'substitution_zone_1': np.array([-3.0, -1.5, 0]),  # Left substitution zone
            'substitution_zone_2': np.array([-3.0, 1.5, 0]),   # Right substitution zone
            'substitution_zone_3': np.array([3.0, -1.5, 0]),   # Left substitution zone
            'substitution_zone_4': np.array([3.0, 1.5, 0]),    # Right substitution zone
            
            # Free zone boundaries (FIVB standards)
            'free_zone_sideline_1': np.array([-14.0, -4.5, 0]),  # Left sideline free zone
            'free_zone_sideline_2': np.array([14.0, -4.5, 0]),   # Right sideline free zone
            'free_zone_sideline_3': np.array([-14.0, 4.5, 0]),   # Left sideline free zone
            'free_zone_sideline_4': np.array([14.0, 4.5, 0]),    # Right sideline free zone
            'free_zone_endline_1': np.array([-9.0, -11.0, 0]),   # Back endline free zone
            'free_zone_endline_2': np.array([9.0, -11.0, 0]),    # Back endline free zone
            'free_zone_endline_3': np.array([-9.0, 11.0, 0]),    # Front endline free zone
            'free_zone_endline_4': np.array([9.0, 11.0, 0]),     # Front endline free zone
            
            # Additional reference points for better calibration
            'mid_court_1': np.array([-4.5, 0, 0]),       # Mid court left
            'mid_court_2': np.array([4.5, 0, 0]),        # Mid court right
            'mid_width_1': np.array([0, -2.25, 0]),      # Mid width back
            'mid_width_2': np.array([0, 2.25, 0]),       # Mid width front
            
            # Height reference points (for free playing space)
            'height_ref_1': np.array([0, 0, 1.0]),       # 1m height
            'height_ref_2': np.array([0, 0, 2.0]),       # 2m height
            'height_ref_3': np.array([0, 0, 3.0]),       # 3m height
            'height_ref_4': np.array([0, 0, 5.0]),       # 5m height
            'height_ref_5': np.array([0, 0, 7.0]),       # 7m height (minimum free playing space)
            'height_ref_6': np.array([0, 0, 12.5]),      # 12.5m height (FIVB free playing space)
        }
    
    def select_reference_points(self, image_path: str, camera_id: str) -> Dict[str, np.ndarray]:
        """
        Interactive tool to select reference points on calibration image
        
        Args:
            image_path: Path to calibration image
            camera_id: Camera identifier
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of point names and 2D coordinates
        """
        self.logger.info(f"Starting point selection for camera {camera_id}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return {}
        
        self.current_image = image.copy()
        self.current_camera_id = camera_id
        self.selected_points = []
        
        # Create window and set mouse callback
        window_name = f"Camera {camera_id} Calibration - Select Reference Points"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Display instructions
        self._display_instructions(image)
        
        point_names = list(self.court_reference_points.keys())
        current_point_idx = 0
        
        while current_point_idx < len(point_names):
            point_name = point_names[current_point_idx]
            
            # Display current point to select
            display_image = self.current_image.copy()
            cv2.putText(display_image, f"Select: {point_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show already selected points
            for i, (name, point) in enumerate(self.selected_points):
                cv2.circle(display_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                cv2.putText(display_image, name, (int(point[0])+10, int(point[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Check if point was selected
            if len(self.selected_points) > current_point_idx:
                current_point_idx += 1
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset selection
                self.selected_points = []
                current_point_idx = 0
            elif key == ord('u'):
                # Undo last selection
                if self.selected_points:
                    self.selected_points.pop()
                    current_point_idx = max(0, current_point_idx - 1)
        
        cv2.destroyAllWindows()
        
        # Convert to dictionary
        result = {}
        for name, point in self.selected_points:
            result[name] = np.array([point[0], point[1]])
        
        self.logger.info(f"Selected {len(result)} reference points for camera {camera_id}")
        return result
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            point_names = list(self.court_reference_points.keys())
            if len(self.selected_points) < len(point_names):
                point_name = point_names[len(self.selected_points)]
                self.selected_points.append((point_name, (x, y)))
                self.logger.info(f"Selected {point_name} at ({x}, {y})")
    
    def _display_instructions(self, image: np.ndarray) -> None:
        """Display calibration instructions"""
        instructions = [
            "Volleyball Court Calibration",
            "Instructions:",
            "1. Click on each reference point in order",
            "2. Press 'r' to reset selection",
            "3. Press 'u' to undo last selection",
            "4. Press 'q' to quit",
            "",
            "Reference points to select:",
            "- Court corners (4 points)",
            "- Net posts (2 points)",
            "- Attack lines (4 points)",
            "- Service zones (4 points)",
            "- Center lines (2 points)",
            "- Front zones (4 points)",
            "- Back zones (4 points)",
            "- Substitution zones (4 points)",
            "- Free zones (4 points)",
            "- Mid court points (2 points)",
            "- Mid width points (2 points)",
            "- Height reference points (6 points)"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(image, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    def create_calibration_image(self, output_path: str, court_image_path: str = None) -> None:
        """
        Create a calibration image with reference points marked
        
        Args:
            output_path: Path to save calibration image
            court_image_path: Optional path to court background image
        """
        try:
            # Create blank image or load court image
            if court_image_path and Path(court_image_path).exists():
                image = cv2.imread(court_image_path)
            else:
                # Create blank image
                image = np.ones((800, 1200, 3), dtype=np.uint8) * 128
            
            # Draw court outline
            self._draw_court_outline(image)
            
            # Mark reference points
            self._mark_reference_points(image)
            
            # Save image
            cv2.imwrite(output_path, image)
            self.logger.info(f"Calibration image saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create calibration image: {str(e)}")
    
    def _draw_court_outline(self, image: np.ndarray) -> None:
        """Draw volleyball court outline on image"""
        height, width = image.shape[:2]
        
        # Court dimensions in pixels (approximate)
        court_width_px = int(width * 0.8)
        court_height_px = int(height * 0.6)
        
        # Court center
        center_x = width // 2
        center_y = height // 2
        
        # Court corners
        x1 = center_x - court_width_px // 2
        y1 = center_y - court_height_px // 2
        x2 = center_x + court_width_px // 2
        y2 = center_y + court_height_px // 2
        
        # Draw court outline
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Draw center line
        center_x_line = center_x
        cv2.line(image, (center_x_line, y1), (center_x_line, y2), (255, 255, 255), 2)
        
        # Draw attack lines
        attack_line_x1 = center_x - int(court_width_px * 0.17)  # 3m from center
        attack_line_x2 = center_x + int(court_width_px * 0.17)
        cv2.line(image, (attack_line_x1, y1), (attack_line_x1, y2), (255, 255, 255), 1)
        cv2.line(image, (attack_line_x2, y1), (attack_line_x2, y2), (255, 255, 255), 1)
        
        # Draw service zones
        service_zone_y1 = center_y - int(court_height_px * 0.17)  # 1.5m from center
        service_zone_y2 = center_y + int(court_height_px * 0.17)
        cv2.line(image, (x1, service_zone_y1), (x1, service_zone_y2), (255, 255, 255), 1)
        cv2.line(image, (x2, service_zone_y1), (x2, service_zone_y2), (255, 255, 255), 1)
    
    def _mark_reference_points(self, image: np.ndarray) -> None:
        """Mark reference points on calibration image"""
        height, width = image.shape[:2]
        
        # Court dimensions in pixels
        court_width_px = int(width * 0.8)
        court_height_px = int(height * 0.6)
        center_x = width // 2
        center_y = height // 2
        
        # Define point positions (scaled to image)
        point_positions = {
            'corner_1': (center_x - court_width_px//2, center_y - court_height_px//2),
            'corner_2': (center_x + court_width_px//2, center_y - court_height_px//2),
            'corner_3': (center_x + court_width_px//2, center_y + court_height_px//2),
            'corner_4': (center_x - court_width_px//2, center_y + court_height_px//2),
            'net_post_1': (center_x, center_y - court_height_px//2),
            'net_post_2': (center_x, center_y + court_height_px//2),
            'antenna_1': (center_x, center_y - court_height_px//2 + 80),
            'antenna_2': (center_x, center_y + court_height_px//2 + 80),
            'attack_line_1': (center_x - int(court_width_px * 0.17), center_y - court_height_px//2),
            'attack_line_2': (center_x - int(court_width_px * 0.17), center_y + court_height_px//2),
            'attack_line_3': (center_x + int(court_width_px * 0.17), center_y - court_height_px//2),
            'attack_line_4': (center_x + int(court_width_px * 0.17), center_y + court_height_px//2),
            'center_line_1': (center_x, center_y - court_height_px//2),
            'center_line_2': (center_x, center_y + court_height_px//2),
            'service_zone_1': (center_x - court_width_px//2, center_y - court_height_px//2),
            'service_zone_2': (center_x - court_width_px//2, center_y + court_height_px//2),
            'service_zone_3': (center_x + court_width_px//2, center_y - court_height_px//2),
            'service_zone_4': (center_x + court_width_px//2, center_y + court_height_px//2),
            'front_zone_1': (center_x - court_width_px//4, center_y - court_height_px//2),
            'front_zone_2': (center_x - court_width_px//4, center_y + court_height_px//2),
            'front_zone_3': (center_x + court_width_px//4, center_y - court_height_px//2),
            'front_zone_4': (center_x + court_width_px//4, center_y + court_height_px//2),
            'back_zone_1': (center_x - court_width_px//2, center_y - court_height_px//2),
            'back_zone_2': (center_x - court_width_px//2, center_y + court_height_px//2),
            'back_zone_3': (center_x + court_width_px//2, center_y - court_height_px//2),
            'back_zone_4': (center_x + court_width_px//2, center_y + court_height_px//2),
            'substitution_zone_1': (center_x - court_width_px//4, center_y - court_height_px//4),
            'substitution_zone_2': (center_x - court_width_px//4, center_y + court_height_px//4),
            'substitution_zone_3': (center_x + court_width_px//4, center_y - court_height_px//4),
            'substitution_zone_4': (center_x + court_width_px//4, center_y + court_height_px//4),
            'free_zone_sideline_1': (center_x - court_width_px//2, center_y - court_height_px//2),
            'free_zone_sideline_2': (center_x - court_width_px//2, center_y + court_height_px//2),
            'free_zone_sideline_3': (center_x + court_width_px//2, center_y - court_height_px//2),
            'free_zone_sideline_4': (center_x + court_width_px//2, center_y + court_height_px//2),
            'free_zone_endline_1': (center_x - court_width_px//2, center_y - court_height_px//2),
            'free_zone_endline_2': (center_x - court_width_px//2, center_y + court_height_px//2),
            'free_zone_endline_3': (center_x + court_width_px//2, center_y - court_height_px//2),
            'free_zone_endline_4': (center_x + court_width_px//2, center_y + court_height_px//2),
            'mid_court_1': (center_x - court_width_px//4, center_y),
            'mid_court_2': (center_x + court_width_px//4, center_y),
            'mid_width_1': (center_x, center_y - court_height_px//4),
            'mid_width_2': (center_x, center_y + court_height_px//4),
            'height_ref_1': (center_x + court_width_px//2 + 50, center_y),
            'height_ref_2': (center_x + court_width_px//2 + 50, center_y - 50),
            'height_ref_3': (center_x + court_width_px//2 + 50, center_y - 100),
            'height_ref_4': (center_x + court_width_px//2 + 50, center_y - 150),
            'height_ref_5': (center_x + court_width_px//2 + 50, center_y - 200),
            'height_ref_6': (center_x + court_width_px//2 + 50, center_y - 250),
        }
        
        # Draw points and labels
        for point_name, (x, y) in point_positions.items():
            cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(image, point_name, (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def validate_calibration(self, camera_matrices: Dict[str, np.ndarray], 
                           reference_points: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """
        Validate camera calibration quality
        
        Args:
            camera_matrices: Dictionary of camera projection matrices
            reference_points: Dictionary of camera_id -> reference points
            
        Returns:
            Dict: Validation results
        """
        validation_results = {}
        
        for camera_id, P in camera_matrices.items():
            if camera_id not in reference_points:
                continue
            
            points_2d = reference_points[camera_id]
            world_points = np.array(list(self.court_reference_points.values()), dtype=np.float32)
            
            # Project 3D points to 2D
            projected_points = []
            for point_3d in world_points:
                point_homogeneous = np.append(point_3d, 1)
                projected_homogeneous = P @ point_homogeneous
                projected_2d = projected_homogeneous[:2] / projected_homogeneous[2]
                projected_points.append(projected_2d)
            
            # Calculate reprojection error
            errors = []
            for point_name, point_2d in points_2d.items():
                if point_name in self.court_reference_points:
                    idx = list(self.court_reference_points.keys()).index(point_name)
                    error = np.linalg.norm(projected_points[idx] - point_2d)
                    errors.append(error)
            
            mean_error = np.mean(errors) if errors else float('inf')
            max_error = np.max(errors) if errors else float('inf')
            
            validation_results[camera_id] = {
                'mean_reprojection_error': mean_error,
                'max_reprojection_error': max_error,
                'num_points_used': len(errors),
                'calibration_quality': 'Good' if mean_error < 10 else 'Poor'
            }
        
        return validation_results
    
    def save_calibration_data(self, reference_points: Dict[str, Dict[str, np.ndarray]], 
                            output_path: str) -> None:
        """
        Save calibration data to file
        
        Args:
            reference_points: Dictionary of camera_id -> reference points
            output_path: Path to save calibration data
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            data_to_save = {}
            for camera_id, points in reference_points.items():
                data_to_save[camera_id] = {name: point.tolist() for name, point in points.items()}
            
            with open(output_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            self.logger.info(f"Calibration data saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration data: {str(e)}")
    
    def load_calibration_data(self, file_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load calibration data from file
        
        Args:
            file_path: Path to calibration data file
            
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Loaded calibration data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            reference_points = {}
            for camera_id, points in data.items():
                reference_points[camera_id] = {name: np.array(point) for name, point in points.items()}
            
            self.logger.info(f"Calibration data loaded from {file_path}")
            return reference_points
            
        except Exception as e:
            self.logger.error(f"Failed to load calibration data: {str(e)}")
            return {}
    
    def create_calibration_report(self, validation_results: Dict, output_path: str) -> None:
        """
        Create calibration validation report
        
        Args:
            validation_results: Validation results from validate_calibration
            output_path: Path to save report
        """
        try:
            report = "Volleyball Hawkeye System - Camera Calibration Report\n"
            report += "=" * 50 + "\n\n"
            
            for camera_id, results in validation_results.items():
                report += f"Camera: {camera_id}\n"
                report += f"  Mean Reprojection Error: {results['mean_reprojection_error']:.2f} pixels\n"
                report += f"  Max Reprojection Error: {results['max_reprojection_error']:.2f} pixels\n"
                report += f"  Number of Points Used: {results['num_points_used']}\n"
                report += f"  Calibration Quality: {results['calibration_quality']}\n\n"
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Calibration report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create calibration report: {str(e)}") 