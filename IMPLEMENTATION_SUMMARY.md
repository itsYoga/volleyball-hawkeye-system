# Volleyball Hawkeye System - Implementation Summary

## Overview

This project successfully adapts the [SEU-Robot-Vision-Project-tennis-Hawkeye-system](https://github.com/GehaoZhang6/SEU-Robot-Vision-Project-tennis-Hawkeye-system) from tennis to volleyball, implementing a multi-camera 3D reconstruction system for volleyball analysis using YOLOv8 (ready for YOLOv11 when available). The system now fully complies with FIVB (Fédération Internationale de Volleyball) standards.

## Key Adaptations from Tennis to Volleyball

### 1. Court Dimensions and Reference Points (FIVB Standards)
- **Tennis Court**: 23.77m × 10.97m
- **Volleyball Court**: 18m × 9m (playing court)
- **Free Zone**: 5m sidelines, 6.5m end lines (FIVB World and Official competitions)
- **Free Playing Space**: 12.5m minimum height (FIVB standards)
- **Reference Points**: 47 volleyball-specific reference points including:
  - Playing court corners (4 points)
  - Net posts (2 points)
  - Net antennas (2 points)
  - Attack lines (4 points)
  - Center lines (2 points)
  - Service zones (4 points)
  - Front zone boundaries (4 points)
  - Back zone boundaries (4 points)
  - Substitution zones (4 points)
  - Free zone boundaries (8 points)
  - Mid court/width points (4 points)
  - Height reference points (6 points)

### 2. Court Structure Elements
- **Playing Court**: 18m × 9m rectangle
- **Center Line**: Divides court into equal halves
- **Attack Lines**: 3m from center line
- **Front Zone**: Between center line and attack lines
- **Back Zone**: Between attack lines and end lines
- **Service Zones**: 9m wide behind end lines
- **Substitution Zone**: Between attack line extensions
- **Free Zone**: 5m sidelines, 6.5m end lines
- **Net**: 2.43m height (men), 2.24m (women)
- **Antennas**: 80cm above net
- **Line Width**: 5cm

### 3. Ball Detection
- **Tennis**: Yellow/white tennis ball
- **Volleyball**: Multiple colors (blue, white, yellow) with YOLOv8 detection
- **Model**: Using YOLOv8n.pt (ready for YOLOv11 upgrade)

### 4. Trajectory Analysis
- **Tennis**: Focus on in/out calls
- **Volleyball**: Comprehensive analysis including:
  - Spike analysis (height, speed, performance assessment)
  - Set analysis (control, accuracy)
  - Serve analysis (distance, trajectory)
  - Pass analysis (reception quality)
  - Bounce detection
  - Performance metrics
  - Court zone-specific analysis

### 5. Boundary Checking
- **Tennis**: Court boundaries only
- **Volleyball**: Includes free zone boundaries and height limits
  - Playing court: 18m × 9m
  - Free zone: 5m sidelines, 6.5m end lines
  - Free playing space: 12.5m height limit

## System Architecture

### Core Components

1. **VolleyballHawkeye** (`src/volleyball_hawkeye.py`)
   - Main system class
   - 3D trajectory reconstruction
   - Ball detection with YOLOv8
   - Trajectory analysis and classification
   - 3D visualization with FIVB court elements
   - Free zone boundary checking

2. **VolleyballCalibrationTools** (`src/calibration_tools.py`)
   - Interactive camera calibration
   - Reference point selection (47 points)
   - Calibration validation
   - Calibration image generation with FIVB court structure

3. **Run Script** (`src/run_hawkeye.py`)
   - Command-line interface
   - Three modes: calibrate, analyze, realtime
   - Multi-camera support

### Key Features Implemented

1. **FIVB-Compliant Court Structure**
   - Complete court dimensions and zones
   - Free zone boundaries
   - Height limits for free playing space
   - All official court elements

2. **Enhanced 3D Visualization**
   - Playing court outline
   - Free zone boundaries
   - Net and antennas
   - Attack lines
   - Center line
   - Service zones

3. **Comprehensive Reference Points**
   - 47 calibration points
   - All court zones represented
   - Height reference points up to 12.5m

4. **Advanced Boundary Checking**
   - Free zone inclusion
   - Height limit enforcement
   - Zone-specific analysis

## Technical Implementation

### Court Dimensions Configuration
```python
# FIVB Standards
court_length = 18.0  # meters (playing court length)
court_width = 9.0    # meters (playing court width)
net_height = 2.43    # meters (men's height)
free_zone_sideline = 5.0    # meters
free_zone_endline = 6.5     # meters
free_playing_space = 12.5   # meters
attack_line_distance = 3.0  # meters
service_zone_width = 9.0    # meters
line_width = 0.05           # meters (5cm)
```

### Reference Points Structure
The system now includes comprehensive reference points for:
- Playing court boundaries
- Free zone boundaries
- All court zones (front, back, service, substitution)
- Height references for free playing space
- Net and antenna positions

### Boundary Checking Algorithm
```python
def _check_bounds(self, trajectory):
    for point in trajectory:
        x, y, z = point
        # Check court boundaries including free zone
        if (abs(x) > court_length/2 + free_zone_endline or 
            abs(y) > court_width/2 + free_zone_sideline):
            return False
        # Check height limit
        if z > free_playing_space:
            return False
    return True
```

## Configuration Files

### hawkeye_config.yaml
Updated to include all FIVB court dimensions:
- Court dimensions and zones
- Free zone specifications
- Net and antenna dimensions
- Line specifications
- Complete reference point list

## Output Enhancements

### Court Dimensions in Results
```json
{
  "court_dimensions": {
    "length": 18.0,
    "width": 9.0,
    "net_height": 2.43,
    "free_zone_sideline": 5.0,
    "free_zone_endline": 6.5,
    "free_playing_space": 12.5,
    "attack_line_distance": 3.0,
    "service_zone_width": 9.0,
    "line_width": 0.05,
    "net_width": 1.0,
    "net_length": 9.5,
    "antenna_height": 0.8
  }
}
```

## Testing and Validation

### Example Usage
Updated `examples/example_usage.py` includes:
- FIVB court structure demonstration
- Zone-specific trajectory analysis
- Free zone boundary testing
- Height limit validation

### Demo Script
Enhanced `demo.py` with:
- FIVB standards information
- Complete court structure display
- 47 reference point calibration
- Free zone and height limit examples

## Future Enhancements

1. **Gender-Specific Net Heights**
   - Automatic detection of men's vs women's games
   - Dynamic net height adjustment

2. **Zone-Specific Analysis**
   - Front zone vs back zone performance metrics
   - Service zone accuracy analysis
   - Substitution zone monitoring

3. **FIVB Rule Integration**
   - Automatic rule violation detection
   - Free zone play analysis
   - Height limit enforcement

4. **Advanced Visualization**
   - Interactive 3D court model
   - Real-time zone highlighting
   - Player position tracking

## Conclusion

The Volleyball Hawkeye System has been successfully updated to fully comply with FIVB standards, providing a comprehensive and accurate analysis platform for volleyball matches. The system now includes all official court elements, proper boundary checking, and enhanced visualization capabilities that reflect the true structure of a volleyball court as defined by international standards.

## References

- **Original Tennis System**: [SEU-Robot-Vision-Project-tennis-Hawkeye-system](https://github.com/GehaoZhang6/SEU-Robot-Vision-Project-tennis-Hawkeye-system)
- **YOLOv8**: Ultralytics YOLOv8 implementation
- **Computer Vision**: OpenCV and NumPy for image processing
- **3D Reconstruction**: Multi-camera triangulation techniques 