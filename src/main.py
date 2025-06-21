#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import logging
from pathlib import Path
from collections import deque
import math
import mediapipe as mp
from torchvision import transforms
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VolleyballTracker:

    def __init__(self, config_path: str = None):
        """
        初始化排球追蹤系統
        
        參數:
            config_path (str): 設定檔路徑
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化排球追蹤系統...")
        
        try:
            # 載入球員偵測模型
            self.player_detector = YOLO('yolov8n.pt')
            self.logger.info("成功載入球員偵測模型")
            
            # 初始化球員追蹤
            self.player_tracks = {}  # 儲存球員追蹤資訊
            self.next_track_id = 0
            self.track_history = {}  # 儲存球員軌跡
            self.max_history_length = 30  # 最大軌跡長度
            
            # 初始化 MediaPipe Pose
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.logger.info("成功載入姿勢追蹤模型")
            
            # 初始化球追蹤參數
            self.ball_trajectory = deque(maxlen=30) 
            self.ball_speed = deque(maxlen=10)  
            self.last_ball_position = None
            self.frame_count = 0
            
            # 初始化動作識別參數
            self.current_action = None
            self.action_frames = 0
            self.action_threshold = 5  # 降低需要連續的幀數
            self.last_action_time = 0
            self.action_cooldown = 15  # 減少冷卻時間
            
            # 初始化動作序列追蹤
            self.pose_history = deque(maxlen=10)  # 保存最近10幀的姿勢
            self.action_sequences = {
                'spike': [],    # 扣球序列
                'set': [],      # 舉球序列
                'receive': [],  # 接球序列
                'block': []     # 攔網序列
            }
            
            # 設定球的顏色範圍（HSV）- 針對 MIKASA V300 調整
            # 藍色球的顏色範圍
            self.blue_ball_lower = np.array([100, 150, 100])
            self.blue_ball_upper = np.array([130, 255, 255])
            
            # 白色球的顏色範圍
            self.ball_color_lower = np.array([0, 0, 200])
            self.ball_color_upper = np.array([180, 30, 255])
            
            # 黃色球的顏色範圍
            self.yellow_ball_lower = np.array([20, 100, 100])
            self.yellow_ball_upper = np.array([30, 255, 255])
            
            self.logger.info("系統初始化完成")
            
        except Exception as e:
            self.logger.error(f"模型載入失敗: {str(e)}")
            raise
    
    def _calculate_ball_speed(self, current_pos, last_pos, fps=30):
        """
        計算球的速度
        
        參數:
            current_pos: 當前位置
            last_pos: 上一個位置
            fps: 影格率
            
        回傳:
            float: 球的速度（像素/秒）
        """
        if last_pos is None:
            return 0
        
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        speed = distance * fps
        
        return speed
    
    def _analyze_trajectory(self, trajectory):
        """
        分析球的軌跡
        
        參數:
            trajectory: 球的軌跡列表
            
        回傳:
            dict: 軌跡分析結果
        """
        if len(trajectory) < 2:
            return None
        
        # 計算軌跡的方向
        start_point = trajectory[0]
        end_point = trajectory[-1]
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        # 計算軌跡的曲率
        total_distance = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        straight_line_distance = math.sqrt(
            (end_point[0] - start_point[0])**2 + 
            (end_point[1] - start_point[1])**2
        )
        
        curvature = total_distance / straight_line_distance if straight_line_distance > 0 else 1
        
        return {
            'angle': angle,
            'curvature': curvature,
            'distance': total_distance
        }
    
    def _track_pose(self, frame):
        """
        追蹤人體姿勢
        
        參數:
            frame: 輸入影格
            
        回傳:
            list: 姿勢關鍵點列表
        """
        # 轉換顏色空間
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 進行姿勢偵測
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            return landmarks
        return None

    def _draw_skeleton(self, frame, landmarks):
        """
        繪製人體骨架
        
        參數:
            frame: 輸入影格
            landmarks: 姿勢關鍵點列表
        """
        if not landmarks:
            return frame
        
        # 定義骨架連接
        connections = [
            # 臉部
            (0, 1), (1, 2), (2, 3), (3, 7),  # 右眼
            (0, 4), (4, 5), (5, 6), (6, 8),  # 左眼
            (9, 10),  # 嘴巴
            
            # 身體
            (11, 12),  # 肩膀
            (11, 13), (13, 15),  # 右手臂
            (12, 14), (14, 16),  # 左手臂
            
            # 軀幹
            (11, 23), (12, 24),  # 上半身
            (23, 24),  # 臀部
            
            # 腿部
            (23, 25), (25, 27), (27, 29), (29, 31),  # 右腿
            (24, 26), (26, 28), (28, 30), (30, 32),  # 左腿
        ]
        
        # 繪製骨架連接
        for connection in connections:
            start_idx, end_idx = connection
            if (landmarks[start_idx]['visibility'] > 0.5 and 
                landmarks[end_idx]['visibility'] > 0.5):
                start_point = (
                    int(landmarks[start_idx]['x'] * frame.shape[1]),
                    int(landmarks[start_idx]['y'] * frame.shape[0])
                )
                end_point = (
                    int(landmarks[end_idx]['x'] * frame.shape[1]),
                    int(landmarks[end_idx]['y'] * frame.shape[0])
                )
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                
                # 在關鍵點上顯示座標
                if start_idx in [11, 12, 13, 14, 15, 16]:  # 只顯示手臂相關的關鍵點
                    cv2.circle(frame, start_point, 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"({start_point[0]}, {start_point[1]})", 
                              (start_point[0] + 5, start_point[1] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _track_ball(self, frame: np.ndarray) -> tuple:
        """
        追蹤球的位置
        
        參數:
            frame (np.ndarray): 輸入的影格
            
        回傳:
            tuple: 球的位置 (x, y) 或 None
        """
        try:
            # 轉換到 HSV 色彩空間
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 建立不同顏色球的遮罩
            mask_white = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
            mask_blue = cv2.inRange(hsv, self.blue_ball_lower, self.blue_ball_upper)
            mask_yellow = cv2.inRange(hsv, self.yellow_ball_lower, self.yellow_ball_upper)
            
            # 合併所有遮罩
            mask = cv2.bitwise_or(mask_white, mask_blue)
            mask = cv2.bitwise_or(mask, mask_yellow)
            
            # 進行形態學操作來減少雜訊
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 尋找輪廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的輪廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 計算輪廓的面積和周長
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # 計算圓形度（4π * 面積 / 周長^2）
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # 調整面積和圓形度的閾值以適應 MIKASA V300
                if 30 < area < 2000 and circularity > 0.6:  # 放寬條件
                    # 計算輪廓的中心點
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 計算最小外接圓
                        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                        center = (int(x), int(y))
                        radius = int(radius)
                        
                        # 調整半徑範圍
                        if 3 < radius < 40:  # 放寬半徑範圍
                            return (cx, cy)
            
            return None
            
        except Exception as e:
            self.logger.error(f"球追蹤失敗: {str(e)}")
            return None
    
    def _detect_volleyball_action(self, pose_landmarks, ball_position, ball_speed):
        """
        識別排球動作，考慮連續幀的動作序列
        
        參數:
            pose_landmarks: 姿勢關鍵點
            ball_position: 球的位置
            ball_speed: 球的速度
            
        回傳:
            str: 識別出的動作名稱
        """
        if not pose_landmarks:
            return None
            
        # 保存當前姿勢到歷史記錄
        self.pose_history.append(pose_landmarks)
        
        # 取得關鍵點位置
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        left_elbow = pose_landmarks[13]
        right_elbow = pose_landmarks[14]
        left_wrist = pose_landmarks[15]
        right_wrist = pose_landmarks[16]
        
        # 計算手臂角度
        left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # 計算手腕相對於肩膀的位置
        left_wrist_above_shoulder = left_wrist['y'] < left_shoulder['y']
        right_wrist_above_shoulder = right_wrist['y'] < right_shoulder['y']
        
        # 計算手腕的水平距離
        wrist_distance = abs(left_wrist['x'] - right_wrist['x'])
        
        # 輸出調試信息
        print("\n=== 手臂動作分析 ===")
        print(f"左手臂角度: {left_arm_angle:.2f}°")
        print(f"右手臂角度: {right_arm_angle:.2f}°")
        print(f"左手腕位置: ({left_wrist['x']:.2f}, {left_wrist['y']:.2f})")
        print(f"右手腕位置: ({right_wrist['x']:.2f}, {right_wrist['y']:.2f})")
        print(f"雙手水平距離: {wrist_distance:.2f}")
        
        # 檢查所有動作序列
        if len(self.pose_history) >= 3:  # 至少需要3幀來判斷動作
            # 扣球動作序列
            if (left_arm_angle > 100 and right_arm_angle > 100 and
                left_wrist_above_shoulder and right_wrist_above_shoulder):
                self.action_sequences['spike'].append("prepare")
            elif (left_arm_angle < 150 and right_arm_angle < 150 and
                  not left_wrist_above_shoulder and not right_wrist_above_shoulder):
                self.action_sequences['spike'].append("swing")
            
            # 舉球動作序列
            if (left_arm_angle > 20 and right_arm_angle > 20 and
                left_wrist_above_shoulder and right_wrist_above_shoulder):
                self.action_sequences['set'].append("ready")
            elif (left_arm_angle > 20 and right_arm_angle > 20 and
                  wrist_distance < 0.8):
                self.action_sequences['set'].append("contact")
            
            # 接球動作序列
            if (left_arm_angle < 160 and right_arm_angle < 160 and
                (left_wrist_above_shoulder or right_wrist_above_shoulder)):
                self.action_sequences['receive'].append("ready")
            elif (left_arm_angle < 160 and right_arm_angle < 160 and
                  not left_wrist_above_shoulder and not right_wrist_above_shoulder):
                self.action_sequences['receive'].append("contact")
            
            # 攔網動作序列
            if (left_arm_angle > 90 and right_arm_angle > 90 and
                left_wrist_above_shoulder and right_wrist_above_shoulder):
                self.action_sequences['block'].append("ready")
            elif (left_arm_angle > 90 and right_arm_angle > 90 and
                  wrist_distance > 0.2):
                self.action_sequences['block'].append("jump")
            
            # 檢查動作序列是否完成
            # 扣球動作
            if len(self.action_sequences['spike']) >= 2:
                if self.action_sequences['spike'][-2:] == ["prepare", "swing"]:
                    self._clear_all_sequences()
                    return "扣球 (Spiking)"
            
            # 舉球動作
            if len(self.action_sequences['set']) >= 2:
                if self.action_sequences['set'][-2:] == ["ready", "contact"]:
                    self._clear_all_sequences()
                    return "舉球 (Setting)"
            
            # 接球動作
            if len(self.action_sequences['receive']) >= 2:
                if self.action_sequences['receive'][-2:] == ["ready", "contact"]:
                    self._clear_all_sequences()
                    return "接球 (Receiving)"
            
            # 攔網動作
            if len(self.action_sequences['block']) >= 2:
                if self.action_sequences['block'][-2:] == ["ready", "jump"]:
                    self._clear_all_sequences()
                    return "攔網 (Blocking)"
            
            # 如果序列太長，清除舊的動作
            for action_type in self.action_sequences:
                if len(self.action_sequences[action_type]) > 5:
                    self.action_sequences[action_type] = self.action_sequences[action_type][-5:]
        
        return None

    def _clear_all_sequences(self):
        """
        清除所有動作序列
        """
        for action_type in self.action_sequences:
            self.action_sequences[action_type] = []

    def _calculate_angle(self, a, b, c):
        """
        計算三個點形成的角度
        
        參數:
            a, b, c: 三個點的座標
            
        回傳:
            float: 角度（度）
        """
        a = np.array([a['x'], a['y']])
        b = np.array([b['x'], b['y']])
        c = np.array([c['x'], c['y']])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def _extract_features(self, frame, bbox):
        """
        從球員區域提取特徵
        
        參數:
            frame: 輸入影格
            bbox: 邊界框 [x1, y1, x2, y2]
            
        回傳:
            torch.Tensor: 特徵向量
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            # 確保邊界框在影格範圍內
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # 提取球員區域
            player_roi = frame[y1:y2, x1:x2]
            if player_roi.size == 0:
                return None
                
            # 轉換為 PIL 圖像
            player_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB)
            player_roi = Image.fromarray(player_roi)
            
            # 應用轉換
            player_roi = self.transform(player_roi)
            player_roi = player_roi.unsqueeze(0)
            
            # 如果有 GPU，將數據移到 GPU
            if torch.cuda.is_available():
                player_roi = player_roi.cuda()
            
            # 提取特徵
            with torch.no_grad():
                features = self.reid_model(player_roi)
            
            return features
            
        except Exception as e:
            self.logger.error(f"特徵提取失敗: {str(e)}")
            return None

    def _update_tracks(self, detections, frame):
        """
        更新球員追蹤
        
        參數:
            detections: 偵測結果列表
            frame: 當前影格
        """
        if not detections:
            return
            
        # 如果沒有現有的追蹤，為所有偵測創建新的追蹤
        if not self.player_tracks:
            for det in detections:
                if det['class'] == 'person':
                    self.player_tracks[self.next_track_id] = {
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'lost_count': 0
                    }
                    self.track_history[self.next_track_id] = deque(maxlen=self.max_history_length)
                    self.track_history[self.next_track_id].append(det['bbox'])
                    self.next_track_id += 1
            return
        
        # 使用簡單的 IoU 匹配
        for track_id, track_info in list(self.player_tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if det['class'] != 'person':
                    continue
                    
                # 計算 IoU
                track_bbox = track_info['bbox']
                det_bbox = det['bbox']
                
                # 計算交集區域
                x1 = max(track_bbox[0], det_bbox[0])
                y1 = max(track_bbox[1], det_bbox[1])
                x2 = min(track_bbox[2], det_bbox[2])
                y2 = min(track_bbox[3], det_bbox[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                    union = track_area + det_area - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = i
            
            # 更新追蹤
            if best_iou > 0.3:  # IoU 閾值
                det = detections[best_det_idx]
                self.player_tracks[track_id]['bbox'] = det['bbox']
                self.player_tracks[track_id]['confidence'] = det['confidence']
                self.player_tracks[track_id]['lost_count'] = 0
                self.track_history[track_id].append(det['bbox'])
            else:
                self.player_tracks[track_id]['lost_count'] += 1
                if self.player_tracks[track_id]['lost_count'] > 30:  # 30幀未匹配則刪除
                    del self.player_tracks[track_id]
                    del self.track_history[track_id]
        
        # 為未匹配的偵測創建新的追蹤
        for det in detections:
            if det['class'] != 'person':
                continue
                
            # 檢查是否已經被匹配
            matched = False
            for track_info in self.player_tracks.values():
                if self._calculate_iou(track_info['bbox'], det['bbox']) > 0.3:
                    matched = True
                    break
            
            if not matched:
                self.player_tracks[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'lost_count': 0
                }
                self.track_history[self.next_track_id] = deque(maxlen=self.max_history_length)
                self.track_history[self.next_track_id].append(det['bbox'])
                self.next_track_id += 1

    def _calculate_iou(self, bbox1, bbox2):
        """
        計算兩個邊界框的 IoU
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 > x1 and y2 > y1:
            intersection = (x2 - x1) * (y2 - y1)
            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = bbox1_area + bbox2_area - intersection
            return intersection / union if union > 0 else 0
        return 0

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        處理單一影格
        
        參數:
            frame (np.ndarray): 輸入的影格
            
        回傳:
            dict: 包含偵測結果的字典
        """
        try:
            # 偵測球員
            player_results = self.player_detector(frame)
            player_detections = self._process_detections(player_results)
            
            # 更新球員追蹤
            self._update_tracks(player_detections, frame)
            
            # 追蹤姿勢
            pose_landmarks = self._track_pose(frame)
            
            # 追蹤球
            ball_position = self._track_ball(frame)
            if ball_position:
                # 計算球的速度
                speed = self._calculate_ball_speed(ball_position, self.last_ball_position)
                self.ball_speed.append(speed)
                
                # 更新球的軌跡
                self.ball_trajectory.append(ball_position)
                self.last_ball_position = ball_position
            
            # 識別排球動作
            detected_action = self._detect_volleyball_action(
                pose_landmarks, 
                ball_position, 
                list(self.ball_speed)
            )
            
            # 更新動作狀態
            if detected_action:
                if detected_action == self.current_action:
                    self.action_frames += 1
                    if self.action_frames >= self.action_threshold:
                        self.last_action_time = self.frame_count
                        self.current_action = detected_action  # 確保動作被設置
                else:
                    self.current_action = detected_action
                    self.action_frames = 1
            else:
                if self.frame_count - self.last_action_time > self.action_cooldown:
                    self.current_action = None
                    self.action_frames = 0
            
            self.frame_count += 1
            
            # 分析球的軌跡
            trajectory_analysis = self._analyze_trajectory(list(self.ball_trajectory)) if len(self.ball_trajectory) > 1 else None
            
            # 在畫面上顯示偵測結果
            debug_frame = frame.copy()
            
            # 顯示 HSV 遮罩（用於除錯）
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_white = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
            mask_blue = cv2.inRange(hsv, self.blue_ball_lower, self.blue_ball_upper)
            mask_yellow = cv2.inRange(hsv, self.yellow_ball_lower, self.yellow_ball_upper)
            mask = cv2.bitwise_or(mask_white, mask_blue)
            mask = cv2.bitwise_or(mask, mask_yellow)
            
            # 將遮罩轉換為彩色以便顯示
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # 在畫面上顯示遮罩
            cv2.imshow('Ball Detection Mask', mask_display)
            
            # 在返回結果中加入追蹤資訊
            return {
                'players': player_detections,
                'tracks': self.player_tracks,
                'track_history': self.track_history,
                'pose': pose_landmarks,
                'ball': ball_position,
                'ball_trajectory': list(self.ball_trajectory),
                'ball_speed': list(self.ball_speed),
                'trajectory_analysis': trajectory_analysis,
                'current_action': self.current_action if self.action_frames >= self.action_threshold else None,
                'frame': frame,
                'debug_frame': debug_frame
            }
            
        except Exception as e:
            self.logger.error(f"影格處理失敗: {str(e)}")
            return {'error': str(e)}
    
    def _process_detections(self, results) -> list:
        """
        處理偵測結果
        
        參數:
            results: YOLO 偵測結果
            
        回傳:
            list: 處理後的偵測結果列表
        """
        processed_detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 取得邊界框座標
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # 只處理人物和可能的球類物件
                if class_name in ['person', 'sports ball', 'ball']:
                    processed_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_name
                    })
        
        return processed_detections

def main():
    """
    主程式入口點
    """
    try:
        # 初始化追蹤器
        tracker = VolleyballTracker()
        logger.info("排球追蹤系統已啟動")
        
        # 嘗試開啟攝影機
        logger.info("正在嘗試開啟攝影機...")
        cap = cv2.VideoCapture(0)
        
        # 檢查攝影機是否成功開啟
        if not cap.isOpened():
            logger.error("無法開啟攝影機。請確認：")
            logger.error("1. 攝影機已正確連接")
            logger.error("2. 已授予攝影機存取權限")
            logger.error("3. 沒有其他程式正在使用攝影機")
            return
        
        logger.info("攝影機已成功開啟")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("無法讀取影格")
                break
            
            # 處理影格
            results = tracker.process_frame(frame)
            
            # 顯示結果
            if 'error' not in results:
                # 在畫面上繪製球員追蹤結果
                for track_id, track_info in results['tracks'].items():
                    x1, y1, x2, y2 = map(int, track_info['bbox'])
                    # 繪製邊界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 顯示追蹤 ID
                    label = f"Player {track_id}"
                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 繪製軌跡
                    if track_id in results['track_history']:
                        history = results['track_history'][track_id]
                        for i in range(1, len(history)):
                            if history[i-1] is None or history[i] is None:
                                continue
                            pt1 = (int((history[i-1][0] + history[i-1][2])/2),
                                  int((history[i-1][1] + history[i-1][3])/2))
                            pt2 = (int((history[i][0] + history[i][2])/2),
                                  int((history[i][1] + history[i][3])/2))
                            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                
                # 在畫面上繪製球員偵測結果
                for player in results['players']:
                    x1, y1, x2, y2 = player['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 顯示偵測類別和信心度
                    label = f"{player['class']}: {player['confidence']:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 在畫面上顯示動作標籤
                if results['current_action']:
                    action = results['current_action']
                    # 在畫面頂部中央顯示動作標籤
                    action_label = f"動作: {action}"
                    (text_width, text_height), _ = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
                    label_x = (frame.shape[1] - text_width) // 2
                    label_y = 80  # 距離頂部80像素
                    
                    # 繪製標籤背景
                    cv2.rectangle(frame, 
                                (label_x - 20, label_y - text_height - 20),
                                (label_x + text_width + 20, label_y + 20),
                                (0, 0, 0), -1)
                    
                    # 繪製標籤文字
                    cv2.putText(frame, action_label, (label_x, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
                    
                    # 在終端輸出當前動作
                    print(f"\n當前動作: {action}")
                
                # 在畫面上繪製骨架
                if results['pose']:
                    frame = tracker._draw_skeleton(frame, results['pose'])
                
                # 繪製球的軌跡
                for i in range(1, len(results['ball_trajectory'])):
                    if results['ball_trajectory'][i-1] is None or results['ball_trajectory'][i] is None:
                        continue
                    cv2.line(frame, results['ball_trajectory'][i-1], results['ball_trajectory'][i], (0, 0, 255), 2)
                
                # 如果偵測到球，顯示標籤
                if results['ball']:
                    x, y = results['ball']
                    # 設定標籤文字和位置
                    label = "MIKASA V300"
                    # 取得文字大小
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    # 計算標籤背景的位置
                    label_x = x - text_width // 2
                    label_y = y - 20  # 在球的上方顯示標籤
                    # 繪製標籤背景
                    cv2.rectangle(frame, 
                                (label_x - 5, label_y - text_height - 5),
                                (label_x + text_width + 5, label_y + 5),
                                (0, 0, 0), -1)
                    # 繪製標籤文字
                    cv2.putText(frame, label, (label_x, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 顯示軌跡分析結果
                if results['trajectory_analysis']:
                    analysis = results['trajectory_analysis']
                    cv2.putText(frame, f"Angle: {analysis['angle']:.1f}°", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Curvature: {analysis['curvature']:.2f}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 顯示影格
            cv2.imshow('Volleyball Tracker', frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("使用者按下 'q' 鍵，程式結束")
                break
        
        # 釋放資源
        cap.release()
        cv2.destroyAllWindows()
        logger.info("程式正常結束")
        
    except Exception as e:
        logger.error(f"程式執行錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 