#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from collections import deque
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from typing import List, Tuple, Dict, Optional
import json
import pickle

# Only import YOLOv11Detector
from yolo_detector import YOLOv11Detector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VolleyballHawkeye:
    """
    Volleyball Hawkeye System based on multi-camera 3D reconstruction
    Adapted for volleyball analysis
    Uses YOLOv11 exclusively
    """
    
    def __init__(self, config_path: str = None, yolo_model_path: str = "yolo11n.pt"):
        """
        Initialize Volleyball Hawkeye System
        
        Args:
            config_path (str): Configuration file path
            yolo_model_path (str): Path to YOLOv11 model
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Volleyball Hawkeye System...")
        
        # Camera parameters
        self.camera_matrices = {}  # Camera projection matrices
        self.camera_params = {}    # Intrinsic and extrinsic parameters
        
        # Ball tracking
        self.ball_tracks = {}      # Ball positions from each camera
        self.ball_3d_trajectory = []  # 3D reconstructed trajectory
        self.ball_speed_history = deque(maxlen=30)
        
        # Court dimensions (volleyball court) - Updated to official FIVB standards
        self.court_length = 18.0  # meters (playing court length)
        self.court_width = 9.0    # meters (playing court width)
        self.net_height = 2.43    # meters (men's height)
        
        # Free zone dimensions (FIVB standards)
        self.free_zone_sideline = 5.0    # meters (FIVB World and Official competitions)
        self.free_zone_endline = 6.5     # meters (FIVB World and Official competitions)
        self.free_playing_space = 12.5   # meters (FIVB World and Official competitions)
        
        # Court line dimensions
        self.line_width = 0.05    # meters (5 cm line width)
        self.attack_line_distance = 3.0  # meters (attack line from center line)
        
        # Net dimensions
        self.net_width = 1.0      # meters (net width)
        self.net_length = 9.5     # meters (net length)
        self.antenna_height = 0.8 # meters (antenna extends 80cm above net)
        
        # Service zone dimensions
        self.service_zone_width = 9.0  # meters (service zone width)
        
        # Substitution zone
        self.substitution_zone_width = 3.0  # meters (between attack line extensions)
        
        # Volleyball specific parameters
        self.ball_radius = 0.105  # meters (official volleyball radius)
        self.ball_mass = 0.27     # kg
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Initialize YOLOv11 detector
        try:
            self.ball_detector = YOLOv11Detector(
                model_path=yolo_model_path,
                confidence_threshold=self.confidence_threshold
            )
            self.logger.info(f"Successfully loaded YOLOv11 ball detection model: {yolo_model_path}")
            model_info = self.ball_detector.get_model_info()
            self.logger.info(f"Model info: {model_info}")
        except Exception as e:
            self.logger.error(f"Failed to load ball detection model: {str(e)}")
            raise
        
        # Volleyball court reference points (3D world coordinates)
        self.court_reference_points = self._initialize_court_reference_points()
        
        self.logger.info("Volleyball Hawkeye System initialized successfully")
    
    def _initialize_court_reference_points(self) -> Dict[str, np.ndarray]:
        """
        Initialize volleyball court reference points in 3D world coordinates
        Updated to official FIVB standards with comprehensive court elements
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of reference point names and their 3D coordinates
        """
        # Volleyball court reference points (in meters)
        # Origin at center of court, X-axis along length, Y-axis along width, Z-axis up
        # Updated to include all official court elements
        points = {
            # Playing court corners (18m x 9m)
            'corner_1': np.array([-self.court_length/2, -self.court_width/2, 0]),  # Back left
            'corner_2': np.array([self.court_length/2, -self.court_width/2, 0]),   # Back right
            'corner_3': np.array([self.court_length/2, self.court_width/2, 0]),    # Front right
            'corner_4': np.array([-self.court_length/2, self.court_width/2, 0]),   # Front left
            
            # Net posts (at center line)
            'net_post_1': np.array([0, -self.court_width/2, self.net_height]),    # Left post
            'net_post_2': np.array([0, self.court_width/2, self.net_height]),     # Right post
            
            # Net antennas (80cm above net, positioned on the sides of the net)
            'antenna_1': np.array([0, -self.court_width/2, self.net_height + self.antenna_height]),  # Left antenna
            'antenna_2': np.array([0, self.court_width/2, self.net_height + self.antenna_height]),   # Right antenna
            
            # Attack lines (3m from center line)
            'attack_line_1': np.array([-self.attack_line_distance, -self.court_width/2, 0]),  # Left attack line
            'attack_line_2': np.array([-self.attack_line_distance, self.court_width/2, 0]),   # Right attack line
            'attack_line_3': np.array([self.attack_line_distance, -self.court_width/2, 0]),   # Left attack line
            'attack_line_4': np.array([self.attack_line_distance, self.court_width/2, 0]),    # Right attack line
            
            # Center line points
            'center_line_1': np.array([0, -self.court_width/2, 0]),               # Left center line
            'center_line_2': np.array([0, self.court_width/2, 0]),                # Right center line
            
            # Service zones (9m wide behind end lines)
            'service_zone_1': np.array([-self.court_length/2, -self.service_zone_width/2, 0]),  # Left service zone
            'service_zone_2': np.array([-self.court_length/2, self.service_zone_width/2, 0]),   # Right service zone
            'service_zone_3': np.array([self.court_length/2, -self.service_zone_width/2, 0]),   # Left service zone
            'service_zone_4': np.array([self.court_length/2, self.service_zone_width/2, 0]),    # Right service zone
            
            # Front zone boundaries (between center line and attack lines)
            'front_zone_1': np.array([-self.attack_line_distance/2, -self.court_width/2, 0]),  # Left front zone
            'front_zone_2': np.array([-self.attack_line_distance/2, self.court_width/2, 0]),   # Right front zone
            'front_zone_3': np.array([self.attack_line_distance/2, -self.court_width/2, 0]),   # Left front zone
            'front_zone_4': np.array([self.attack_line_distance/2, self.court_width/2, 0]),    # Right front zone
            
            # Back zone boundaries (between attack lines and end lines)
            'back_zone_1': np.array([-self.court_length/2 + self.attack_line_distance/2, -self.court_width/2, 0]),  # Left back zone
            'back_zone_2': np.array([-self.court_length/2 + self.attack_line_distance/2, self.court_width/2, 0]),   # Right back zone
            'back_zone_3': np.array([self.court_length/2 - self.attack_line_distance/2, -self.court_width/2, 0]),   # Left back zone
            'back_zone_4': np.array([self.court_length/2 - self.attack_line_distance/2, self.court_width/2, 0]),    # Right back zone
            
            # Substitution zone (between attack line extensions)
            'substitution_zone_1': np.array([-self.attack_line_distance, -self.substitution_zone_width/2, 0]),  # Left substitution zone
            'substitution_zone_2': np.array([-self.attack_line_distance, self.substitution_zone_width/2, 0]),   # Right substitution zone
            'substitution_zone_3': np.array([self.attack_line_distance, -self.substitution_zone_width/2, 0]),   # Left substitution zone
            'substitution_zone_4': np.array([self.attack_line_distance, self.substitution_zone_width/2, 0]),    # Right substitution zone
            
            # Free zone boundaries (FIVB standards)
            'free_zone_sideline_1': np.array([-self.court_length/2 - self.free_zone_sideline, -self.court_width/2, 0]),  # Left sideline free zone
            'free_zone_sideline_2': np.array([self.court_length/2 + self.free_zone_sideline, -self.court_width/2, 0]),   # Right sideline free zone
            'free_zone_sideline_3': np.array([-self.court_length/2 - self.free_zone_sideline, self.court_width/2, 0]),   # Left sideline free zone
            'free_zone_sideline_4': np.array([self.court_length/2 + self.free_zone_sideline, self.court_width/2, 0]),    # Right sideline free zone
            'free_zone_endline_1': np.array([-self.court_length/2, -self.court_width/2 - self.free_zone_endline, 0]),    # Back endline free zone
            'free_zone_endline_2': np.array([self.court_length/2, -self.court_width/2 - self.free_zone_endline, 0]),     # Back endline free zone
            'free_zone_endline_3': np.array([-self.court_length/2, self.court_width/2 + self.free_zone_endline, 0]),     # Front endline free zone
            'free_zone_endline_4': np.array([self.court_length/2, self.court_width/2 + self.free_zone_endline, 0]),      # Front endline free zone
            
            # Additional reference points for better calibration
            'mid_court_1': np.array([-self.court_length/4, 0, 0]),                # Mid court left
            'mid_court_2': np.array([self.court_length/4, 0, 0]),                 # Mid court right
            'mid_width_1': np.array([0, -self.court_width/4, 0]),                 # Mid width back
            'mid_width_2': np.array([0, self.court_width/4, 0]),                  # Mid width front
            
            # Height reference points (for free playing space)
            'height_ref_1': np.array([0, 0, 1.0]),                                # 1m height
            'height_ref_2': np.array([0, 0, 2.0]),                                # 2m height
            'height_ref_3': np.array([0, 0, 3.0]),                                # 3m height
            'height_ref_4': np.array([0, 0, 5.0]),                                # 5m height
            'height_ref_5': np.array([0, 0, 7.0]),                                # 7m height (minimum free playing space)
            'height_ref_6': np.array([0, 0, 12.5]),                               # 12.5m height (FIVB free playing space)
        }
        
        return points
    
    def calibrate_cameras(self, calibration_images: Dict[str, List[str]], 
                         reference_points_2d: Dict[str, List[np.ndarray]]) -> bool:
        """
        Calibrate cameras using reference points
        
        Args:
            calibration_images: Dictionary of camera_id -> list of image paths
            reference_points_2d: Dictionary of camera_id -> list of 2D point arrays
            
        Returns:
            bool: True if calibration successful
        """
        self.logger.info("Starting camera calibration...")
        
        try:
            for camera_id, points_2d_list in reference_points_2d.items():
                self.logger.info(f"Calibrating camera {camera_id}...")
                
                # Convert reference points to numpy arrays
                world_points = np.array(list(self.court_reference_points.values()), dtype=np.float32)
                
                # Use neural network approach for camera matrix calculation
                camera_matrix = self._solve_camera_matrix_neural_network(
                    world_points, points_2d_list[0]  # Use first set of points
                )
                
                if camera_matrix is not None:
                    self.camera_matrices[camera_id] = camera_matrix
                    self.logger.info(f"Camera {camera_id} calibrated successfully")
                else:
                    self.logger.error(f"Failed to calibrate camera {camera_id}")
                    return False
            
            self.logger.info("All cameras calibrated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera calibration failed: {str(e)}")
            return False
    
    def _solve_camera_matrix_neural_network(self, world_points: np.ndarray, 
                                          image_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve camera matrix using neural network approach (adapted from tennis system)
        
        Args:
            world_points: 3D world coordinates (N, 3)
            image_points: 2D image coordinates (N, 2)
            
        Returns:
            np.ndarray: Camera projection matrix (3, 4) or None if failed
        """
        try:
            # Normalize points
            world_points_normalized = self._normalize_points_3d(world_points)
            image_points_normalized = self._normalize_points_2d(image_points)
            
            # Create linear system matrix
            N = len(world_points)
            M = np.zeros((2*N, 12))
            
            for i in range(N):
                X, Y, Z = world_points_normalized[i]
                u, v = image_points_normalized[i]
                
                # Fill matrix for this point
                M[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
                M[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
            
            # Solve using SVD
            U, S, Vt = np.linalg.svd(M)
            P = Vt[-1].reshape(3, 4)
            
            # Denormalize
            P = self._denormalize_camera_matrix(P, world_points, image_points)
            
            return P
            
        except Exception as e:
            self.logger.error(f"Neural network camera matrix solution failed: {str(e)}")
            return None
    
    def _normalize_points_3d(self, points: np.ndarray) -> np.ndarray:
        """Normalize 3D points"""
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        scale = np.sqrt(2) / np.std(points_centered)
        return points_centered * scale
    
    def _normalize_points_2d(self, points: np.ndarray) -> np.ndarray:
        """Normalize 2D points"""
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        scale = np.sqrt(2) / np.std(points_centered)
        return points_centered * scale
    
    def _denormalize_camera_matrix(self, P: np.ndarray, world_points: np.ndarray, 
                                  image_points: np.ndarray) -> np.ndarray:
        """Denormalize camera matrix"""
        # This is a simplified denormalization
        # In practice, you would need to apply the inverse normalization transforms
        return P
    
    def detect_ball(self, frame: np.ndarray, camera_id: str) -> List[np.ndarray]:
        """
        Detect volleyball in frame using YOLO detector
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            
        Returns:
            List[np.ndarray]: List of ball detections [x, y, confidence]
        """
        try:
            # Use the YOLO detector
            detections = self.ball_detector.detect_ball(frame, camera_id)
            
            # Convert to expected format [x, y, confidence]
            ball_detections = []
            for detection in detections:
                x, y, confidence, class_id = detection
                ball_detections.append(np.array([x, y, confidence]))
            
            return ball_detections
            
        except Exception as e:
            self.logger.error(f"Ball detection failed: {str(e)}")
            return []
    
    def detect_volleyball_specific(self, frame: np.ndarray, camera_id: str) -> List[np.ndarray]:
        """
        Detect volleyball specifically with enhanced filtering
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            
        Returns:
            List[np.ndarray]: List of volleyball detections [x, y, confidence]
        """
        try:
            # Use volleyball-specific detection
            detections = self.ball_detector.detect_volleyball_specific(frame, camera_id)
            
            # Convert to expected format [x, y, confidence]
            ball_detections = []
            for detection in detections:
                x, y, confidence, class_id = detection
                ball_detections.append(np.array([x, y, confidence]))
            
            return ball_detections
            
        except Exception as e:
            self.logger.error(f"Volleyball-specific detection failed: {str(e)}")
            return []
    
    def track_ball_multi_camera(self, frames: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Track ball across multiple cameras
        
        Args:
            frames: Dictionary of camera_id -> frame
            
        Returns:
            Dict[str, List[np.ndarray]]: Ball tracks for each camera
        """
        ball_tracks = {}
        
        for camera_id, frame in frames.items():
            detections = self.detect_ball(frame, camera_id)
            ball_tracks[camera_id] = detections
        
        return ball_tracks
    
    def reconstruct_3d_trajectory(self, ball_tracks: Dict[str, List[np.ndarray]], 
                                 frame_timestamps: List[float]) -> List[np.ndarray]:
        """
        Reconstruct 3D ball trajectory from multi-camera tracks
        
        Args:
            ball_tracks: Dictionary of camera_id -> list of ball positions
            frame_timestamps: List of frame timestamps
            
        Returns:
            List[np.ndarray]: 3D trajectory points
        """
        self.logger.info("Starting 3D trajectory reconstruction...")
        
        try:
            # Check if we have camera matrices for triangulation
            if not self.camera_matrices:
                self.logger.warning("No camera matrices available. Creating dummy 3D trajectory.")
                # Create a simple 3D trajectory from 2D tracks
                return self._create_dummy_3d_trajectory(ball_tracks, frame_timestamps)
            
            # Align tracks across cameras
            aligned_tracks = self._align_tracks(ball_tracks, frame_timestamps)
            
            # Triangulate 3D points
            trajectory_3d = []
            
            for frame_idx in range(len(frame_timestamps)):
                frame_points = {}
                
                # Collect 2D points from all cameras for this frame
                for camera_id, track in aligned_tracks.items():
                    if frame_idx < len(track) and len(track[frame_idx]) > 0:
                        # Use the detection with highest confidence
                        best_detection = max(track[frame_idx], key=lambda x: x[2])
                        frame_points[camera_id] = best_detection[:2]  # Only x, y
                
                # Triangulate if we have at least 2 cameras
                if len(frame_points) >= 2:
                    point_3d = self._triangulate_point(frame_points)
                    if point_3d is not None:
                        trajectory_3d.append(point_3d)
            
            # Filter outliers
            trajectory_3d = self._filter_outliers(trajectory_3d)
            
            # Fit smooth trajectory
            trajectory_3d = self._fit_trajectory(trajectory_3d)
            
            self.ball_3d_trajectory = trajectory_3d
            self.logger.info(f"3D trajectory reconstructed with {len(trajectory_3d)} points")
            
            return trajectory_3d
            
        except Exception as e:
            self.logger.error(f"3D trajectory reconstruction failed: {str(e)}")
            return []
    
    def _create_dummy_3d_trajectory(self, ball_tracks: Dict[str, List[np.ndarray]], 
                                   frame_timestamps: List[float]) -> List[np.ndarray]:
        """
        Create a dummy 3D trajectory when camera matrices are not available
        
        Args:
            ball_tracks: Dictionary of camera_id -> list of ball positions
            frame_timestamps: List of frame timestamps
            
        Returns:
            List[np.ndarray]: Dummy 3D trajectory points
        """
        trajectory_3d = []
        
        # Use the first camera's tracks to create a simple 3D trajectory
        if ball_tracks:
            first_camera = list(ball_tracks.keys())[0]
            tracks = ball_tracks[first_camera]
            
            for i, frame_tracks in enumerate(tracks):
                if frame_tracks and len(frame_tracks) > 0:
                    # Use the first detection in this frame
                    detection = frame_tracks[0]
                    x, y, conf = detection
                    
                    # Create a simple 3D point
                    # X: horizontal position (scaled)
                    # Y: vertical position (scaled) 
                    # Z: time-based height (simulating ball movement) - minimum 0.1m
                    x_3d = (x - 320) / 100.0  # Center and scale
                    y_3d = (y - 240) / 100.0  # Center and scale
                    z_3d = 0.1 + 2.0 * np.exp(-i/30.0) * np.sin(i/10.0)  # Simulated height with minimum 0.1m
                    
                    # Ensure Z is never negative
                    z_3d = max(0.1, z_3d)
                    
                    trajectory_3d.append(np.array([x_3d, y_3d, z_3d]))
        
        self.logger.info(f"Created dummy 3D trajectory with {len(trajectory_3d)} points")
        return trajectory_3d
    
    def _align_tracks(self, ball_tracks: Dict[str, List[np.ndarray]], 
                     frame_timestamps: List[float]) -> Dict[str, List[List[np.ndarray]]]:
        """
        Align ball tracks across cameras using timestamps
        
        Args:
            ball_tracks: Dictionary of camera_id -> list of ball positions
            frame_timestamps: List of frame timestamps
            
        Returns:
            Dict[str, List[List[np.ndarray]]]: Aligned tracks
        """
        # This is a simplified alignment
        # In practice, you would need more sophisticated temporal alignment
        return ball_tracks
    
    def _triangulate_point(self, frame_points: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Triangulate 3D point from 2D points in multiple cameras
        
        Args:
            frame_points: Dictionary of camera_id -> 2D point [x, y]
            
        Returns:
            np.ndarray: 3D point [x, y, z] or None if failed
        """
        try:
            if len(frame_points) < 2:
                return None
            
            # Prepare matrices for triangulation
            points_2d = []
            projection_matrices = []
            
            for camera_id, point_2d in frame_points.items():
                if camera_id in self.camera_matrices:
                    points_2d.append(point_2d)
                    projection_matrices.append(self.camera_matrices[camera_id])
            
            if len(points_2d) < 2:
                return None
            
            # Convert to numpy arrays
            points_2d = np.array(points_2d, dtype=np.float32)
            projection_matrices = np.array(projection_matrices, dtype=np.float32)
            
            # Triangulate using OpenCV
            point_3d = cv2.triangulatePoints(
                projection_matrices[0], projection_matrices[1],
                points_2d[0].reshape(1, 1, 2), points_2d[1].reshape(1, 1, 2)
            )
            
            # Convert from homogeneous coordinates
            point_3d = point_3d[:3] / point_3d[3]
            
            return point_3d.flatten()
            
        except Exception as e:
            self.logger.error(f"Triangulation failed: {str(e)}")
            return None
    
    def _filter_outliers(self, trajectory: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter outliers from 3D trajectory
        
        Args:
            trajectory: List of 3D points
            
        Returns:
            List[np.ndarray]: Filtered trajectory
        """
        if len(trajectory) < 3:
            return trajectory
        
        try:
            # Convert to numpy array
            trajectory_array = np.array(trajectory)
            
            # Calculate distances between consecutive points
            distances = np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1)
            
            # Remove points with unusually large jumps
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            threshold = mean_distance + 2 * std_distance
            
            filtered_trajectory = [trajectory[0]]  # Keep first point
            
            for i in range(1, len(trajectory)):
                if i < len(distances) and distances[i-1] <= threshold:
                    filtered_trajectory.append(trajectory[i])
            
            return filtered_trajectory
            
        except Exception as e:
            self.logger.error(f"Outlier filtering failed: {str(e)}")
            return trajectory
    
    def _fit_trajectory(self, trajectory: List[np.ndarray]) -> List[np.ndarray]:
        """
        Fit smooth trajectory to 3D points
        
        Args:
            trajectory: List of 3D points
            
        Returns:
            List[np.ndarray]: Fitted trajectory
        """
        if len(trajectory) < 3:
            return trajectory
        
        try:
            # Convert to numpy array
            trajectory_array = np.array(trajectory)
            
            # Fit polynomial to each dimension
            t = np.arange(len(trajectory))
            
            # Fit 3rd degree polynomial to each coordinate
            x_coeffs = np.polyfit(t, trajectory_array[:, 0], min(3, len(trajectory)-1))
            y_coeffs = np.polyfit(t, trajectory_array[:, 1], min(3, len(trajectory)-1))
            z_coeffs = np.polyfit(t, trajectory_array[:, 2], min(3, len(trajectory)-1))
            
            # Generate smooth trajectory
            t_smooth = np.linspace(0, len(trajectory)-1, len(trajectory))
            
            x_smooth = np.polyval(x_coeffs, t_smooth)
            y_smooth = np.polyval(y_coeffs, t_smooth)
            z_smooth = np.polyval(z_coeffs, t_smooth)
            
            # Ensure Z values (heights) are non-negative
            z_smooth = np.maximum(z_smooth, 0.1)  # Minimum height 0.1m
            
            # Combine into trajectory
            fitted_trajectory = []
            for i in range(len(t_smooth)):
                fitted_trajectory.append(np.array([x_smooth[i], y_smooth[i], z_smooth[i]]))
            
            return fitted_trajectory
            
        except Exception as e:
            self.logger.error(f"Trajectory fitting failed: {str(e)}")
            return trajectory
    
    def analyze_trajectory(self, trajectory: List[np.ndarray]) -> Dict:
        """
        Analyze volleyball trajectory for game analysis
        
        Args:
            trajectory: 3D trajectory points
            
        Returns:
            Dict: Analysis results
        """
        if len(trajectory) < 2:
            return {}
        
        try:
            trajectory_array = np.array(trajectory)
            
            # Validate trajectory - ensure all Z values (heights) are non-negative
            if np.any(trajectory_array[:, 2] < 0):
                self.logger.warning("Found negative Z values in trajectory, clamping to minimum height")
                trajectory_array[:, 2] = np.maximum(trajectory_array[:, 2], 0.1)  # Minimum height 0.1m
            
            # Calculate basic statistics
            max_height = np.max(trajectory_array[:, 2])
            min_height = np.min(trajectory_array[:, 2])
            total_distance = np.sum(np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1))
            
            # Calculate speed
            speeds = []
            for i in range(1, len(trajectory)):
                distance = np.linalg.norm(trajectory_array[i] - trajectory_array[i-1])
                speed = distance * 30  # Assuming 30 fps
                speeds.append(speed)
            
            avg_speed = np.mean(speeds) if speeds else 0
            max_speed = np.max(speeds) if speeds else 0
            
            # Detect bounces
            bounces = self._detect_bounces(trajectory_array)
            
            # Determine if ball is in/out
            is_in_bounds = self._check_bounds(trajectory_array)
            
            # Analyze trajectory type
            trajectory_type = self._classify_trajectory(trajectory_array)
            
            analysis = {
                'max_height': max_height,
                'min_height': min_height,
                'total_distance': total_distance,
                'average_speed': avg_speed,
                'max_speed': max_speed,
                'bounces': bounces,
                'is_in_bounds': is_in_bounds,
                'trajectory_type': trajectory_type,
                'trajectory_points': len(trajectory)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Trajectory analysis failed: {str(e)}")
            return {}
    
    def _detect_bounces(self, trajectory: np.ndarray) -> List[int]:
        """
        Detect bounces in trajectory
        
        Args:
            trajectory: 3D trajectory array
            
        Returns:
            List[int]: Indices of bounce points
        """
        bounces = []
        
        # Look for points where ball hits ground (z close to 0)
        ground_threshold = 0.1  # meters
        
        for i in range(1, len(trajectory)-1):
            if (trajectory[i, 2] <= ground_threshold and 
                trajectory[i-1, 2] > ground_threshold and 
                trajectory[i+1, 2] > ground_threshold):
                bounces.append(i)
        
        return bounces
    
    def _check_bounds(self, trajectory: np.ndarray) -> bool:
        """
        Check if ball trajectory is within court bounds including free zone
        
        Args:
            trajectory: 3D trajectory array
            
        Returns:
            bool: True if in bounds (including free zone)
        """
        # Check if any point is outside court boundaries including free zone
        for point in trajectory:
            x, y, z = point
            
            # Check court boundaries including free zone
            # Free zone extends 5m from sidelines and 6.5m from end lines (FIVB standards)
            if (abs(x) > self.court_length/2 + self.free_zone_endline or 
                abs(y) > self.court_width/2 + self.free_zone_sideline):
                return False
            
            # Check height limit (free playing space)
            if z > self.free_playing_space:
                return False
        
        return True
    
    def _classify_trajectory(self, trajectory: np.ndarray) -> str:
        """
        Classify trajectory type
        
        Args:
            trajectory: 3D trajectory array
            
        Returns:
            str: Trajectory classification
        """
        # Simple classification based on trajectory characteristics
        max_height = np.max(trajectory[:, 2])
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        
        if max_height > 4.0:
            return "spike"
        elif max_height > 2.5:
            return "set"
        elif total_distance > 10.0:
            return "serve"
        else:
            return "pass"
    
    def visualize_trajectory(self, trajectory: List[np.ndarray], 
                           analysis: Dict = None) -> None:
        """
        Visualize 3D trajectory
        
        Args:
            trajectory: 3D trajectory points
            analysis: Trajectory analysis results
        """
        if not trajectory:
            return
        
        try:
            trajectory_array = np.array(trajectory)
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot trajectory
            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                   'b-', linewidth=2, label='Ball Trajectory')
            
            # Plot trajectory points
            ax.scatter(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                      c='red', s=20, label='Trajectory Points')
            
            # Plot court boundaries
            self._plot_court_boundaries(ax)
            
            # Add analysis information
            if analysis:
                title = f"Volleyball Trajectory Analysis\n"
                title += f"Max Height: {analysis.get('max_height', 0):.2f}m, "
                title += f"Type: {analysis.get('trajectory_type', 'unknown')}, "
                title += f"In Bounds: {analysis.get('is_in_bounds', True)}"
                ax.set_title(title)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            # Set Z-axis limit to not show negative values
            ax.set_zlim(bottom=0)
            
            ax.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Trajectory visualization failed: {str(e)}")
    
    def _plot_court_boundaries(self, ax) -> None:
        """Plot volleyball court boundaries including all official elements"""
        # Playing court outline (18m x 9m)
        court_x = [-self.court_length/2, self.court_length/2, self.court_length/2, -self.court_length/2, -self.court_length/2]
        court_y = [-self.court_width/2, -self.court_width/2, self.court_width/2, self.court_width/2, -self.court_width/2]
        court_z = [0, 0, 0, 0, 0]
        
        ax.plot(court_x, court_y, court_z, 'k-', linewidth=2, label='Playing Court Boundary')
        
        # Free zone outline (FIVB standards)
        free_zone_x = [-self.court_length/2 - self.free_zone_endline, self.court_length/2 + self.free_zone_endline, 
                       self.court_length/2 + self.free_zone_endline, -self.court_length/2 - self.free_zone_endline, 
                       -self.court_length/2 - self.free_zone_endline]
        free_zone_y = [-self.court_width/2 - self.free_zone_sideline, -self.court_width/2 - self.free_zone_sideline, 
                       self.court_width/2 + self.free_zone_sideline, self.court_width/2 + self.free_zone_sideline, 
                       -self.court_width/2 - self.free_zone_sideline]
        free_zone_z = [0, 0, 0, 0, 0]
        
        ax.plot(free_zone_x, free_zone_y, free_zone_z, 'g--', linewidth=1, label='Free Zone Boundary')
        
        # Net (at center line)
        net_x = [0, 0]
        net_y = [-self.court_width/2, self.court_width/2]
        net_z = [self.net_height, self.net_height]
        
        ax.plot(net_x, net_y, net_z, 'r-', linewidth=3, label='Net')
        
        # Net antennas (80cm above net, positioned on the sides of the net)
        # Left antenna (vertical line from net to antenna height)
        antenna_left_x = [0, 0]
        antenna_left_y = [-self.court_width/2, -self.court_width/2]
        antenna_left_z = [self.net_height, self.net_height + self.antenna_height]
        
        # Right antenna (vertical line from net to antenna height)
        antenna_right_x = [0, 0]
        antenna_right_y = [self.court_width/2, self.court_width/2]
        antenna_right_z = [self.net_height, self.net_height + self.antenna_height]
        
        ax.plot(antenna_left_x, antenna_left_y, antenna_left_z, 'r-', linewidth=2, label='Antennas')
        ax.plot(antenna_right_x, antenna_right_y, antenna_right_z, 'r-', linewidth=2)
        
        # Attack lines (3m from center line)
        attack_line_x1 = [-self.attack_line_distance, -self.attack_line_distance]
        attack_line_y1 = [-self.court_width/2, self.court_width/2]
        attack_line_z1 = [0, 0]
        
        attack_line_x2 = [self.attack_line_distance, self.attack_line_distance]
        attack_line_y2 = [-self.court_width/2, self.court_width/2]
        attack_line_z2 = [0, 0]
        
        ax.plot(attack_line_x1, attack_line_y1, attack_line_z1, 'b-', linewidth=1, label='Attack Lines')
        ax.plot(attack_line_x2, attack_line_y2, attack_line_z2, 'b-', linewidth=1)
        
        # Center line
        center_line_x = [0, 0]
        center_line_y = [-self.court_width/2, self.court_width/2]
        center_line_z = [0, 0]
        
        ax.plot(center_line_x, center_line_y, center_line_z, 'k-', linewidth=2, label='Center Line')
        
        # Service zones (9m wide behind end lines)
        service_zone_x1 = [-self.court_length/2, -self.court_length/2]
        service_zone_y1 = [-self.service_zone_width/2, self.service_zone_width/2]
        service_zone_z1 = [0, 0]
        
        service_zone_x2 = [self.court_length/2, self.court_length/2]
        service_zone_y2 = [-self.service_zone_width/2, self.service_zone_width/2]
        service_zone_z2 = [0, 0]
        
        ax.plot(service_zone_x1, service_zone_y1, service_zone_z1, 'y-', linewidth=1, label='Service Zones')
        ax.plot(service_zone_x2, service_zone_y2, service_zone_z2, 'y-', linewidth=1)
        
        # Set proper 3D scaling for better visualization
        # Calculate the range for each axis
        x_range = self.court_length/2 + self.free_zone_endline
        y_range = self.court_width/2 + self.free_zone_sideline
        z_range = self.free_playing_space
        
        # Set equal aspect ratio for all axes
        max_range = max(x_range, y_range, z_range)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(0, max_range)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
    
    def save_results(self, output_path: str) -> None:
        """
        Save analysis results
        
        Args:
            output_path: Output file path
        """
        try:
            results = {
                'trajectory_3d': [point.tolist() for point in self.ball_3d_trajectory],
                'analysis': self.analyze_trajectory(self.ball_3d_trajectory),
                'camera_matrices': {k: v.tolist() for k, v in self.camera_matrices.items()},
                'court_dimensions': {
                    'length': self.court_length,
                    'width': self.court_width,
                    'net_height': self.net_height,
                    'free_zone_sideline': self.free_zone_sideline,
                    'free_zone_endline': self.free_zone_endline,
                    'free_playing_space': self.free_playing_space,
                    'attack_line_distance': self.attack_line_distance,
                    'service_zone_width': self.service_zone_width,
                    'line_width': self.line_width,
                    'net_width': self.net_width,
                    'net_length': self.net_length,
                    'antenna_height': self.antenna_height
                },
                'yolo_info': self.ball_detector.get_model_info()
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
    
    def load_camera_matrices(self, file_path: str) -> bool:
        """
        Load pre-calibrated camera matrices
        
        Args:
            file_path: Path to camera matrices file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(file_path, 'rb') as f:
                self.camera_matrices = pickle.load(f)
            
            self.logger.info(f"Camera matrices loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load camera matrices: {str(e)}")
            return False
    
    def save_camera_matrices(self, file_path: str) -> bool:
        """
        Save camera matrices
        
        Args:
            file_path: Path to save camera matrices
            
        Returns:
            bool: True if saved successfully
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.camera_matrices, f)
            
            self.logger.info(f"Camera matrices saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save camera matrices: {str(e)}")
            return False
    
    def get_yolo_info(self) -> Dict:
        """
        Get YOLO model information
        
        Returns:
            Dict: YOLO model information
        """
        return self.ball_detector.get_model_info() 