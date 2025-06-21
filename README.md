# 排球鷹眼系統 (Volleyball Hawkeye System)

這是一個基於多攝影機3D重建的排球鷹眼系統，改編自網球鷹眼系統，專門用於排球比賽分析。系統採用國際排球總會(FIVB)標準的場地規格。

## 功能特點 (Features)

- **多攝影機3D重建**: 使用多個攝影機進行球的3D軌跡重建
- **YOLOv11球體偵測**: 使用最新的YOLOv11模型進行排球偵測
- **攝影機校準工具**: 互動式攝影機校準和驗證
- **軌跡分析**: 分析球的軌跡、速度、彈跳點等
- **即時分析**: 支援即時多攝影機分析
- **視覺化**: 3D軌跡視覺化和分析報告
- **FIVB標準**: 完全符合國際排球總會場地規格

## 排球場地結構 (Volleyball Court Structure)

### 比賽場地 (Playing Court)
- **尺寸**: 18公尺 × 9公尺
- **中心線**: 將場地分為兩個相等的半場
- **線寬**: 5公分

### 自由區 (Free Zone)
- **邊線自由區**: 5公尺 (FIVB世界級和正式比賽)
- **端線自由區**: 6.5公尺 (FIVB世界級和正式比賽)
- **自由比賽空間**: 12.5公尺最小高度

### 球網 (Net)
- **高度**: 男子2.43公尺，女子2.24公尺
- **寬度**: 1公尺
- **長度**: 9.5-10公尺
- **天線**: 80公分高於網頂

### 場地線條和區域 (Lines and Zones)
- **攻擊線**: 距離中心線3公尺
- **前區**: 中心線與攻擊線之間的區域
- **後區**: 攻擊線與端線之間的區域
- **發球區**: 端線後方9公尺寬
- **替換區**: 攻擊線延伸線之間

## 系統需求 (Requirements)

- Python 3.8+
- CUDA 支援的 GPU（建議）
- 多個攝影機或影片檔案
- 排球場地參考點

## 安裝步驟 (Installation)

1. 克隆專案：
```bash
git clone [repository_url]
cd volleyball_tracker
```

2. 建立虛擬環境（建議）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

## 使用方法 (Usage)

### 1. 攝影機校準 (Camera Calibration)

首先需要校準攝影機以獲得準確的3D重建：

```bash
# 創建校準模板
python src/run_hawkeye.py --mode calibrate --create_calibration_image

# 校準攝影機（使用影片檔案）
python src/run_hawkeye.py --mode calibrate --cameras video1.mp4 video2.mp4 video3.mp4

# 校準攝影機（使用攝影機設備）
python src/run_hawkeye.py --mode calibrate --cameras 0 1 2
```

### 2. 軌跡分析 (Trajectory Analysis)

分析錄製的影片：

```bash
# 分析多攝影機影片
python src/run_hawkeye.py --mode analyze --cameras video1.mp4 video2.mp4 --camera_matrices camera_matrices.pkl

# 指定輸出目錄
python src/run_hawkeye.py --mode analyze --cameras video1.mp4 video2.mp4 --output_dir results/
```

### 3. 即時分析 (Real-time Analysis)

即時分析多攝影機串流：

```bash
# 使用攝影機設備
python src/run_hawkeye.py --mode realtime --cameras 0 1 2 --camera_matrices camera_matrices.pkl

# 使用影片檔案（用於測試）
python src/run_hawkeye.py --mode realtime --cameras video1.mp4 video2.mp4
```

## 專案結構 (Project Structure)

```
volleyball_tracker/
├── src/                    # 原始碼
│   ├── volleyball_hawkeye.py    # 主要鷹眼系統
│   ├── calibration_tools.py     # 攝影機校準工具
│   ├── run_hawkeye.py          # 主執行腳本
│   └── main.py                 # 原始追蹤系統
├── output/                 # 輸出結果
│   ├── calibration_data.json   # 校準資料
│   ├── camera_matrices.pkl     # 攝影機矩陣
│   ├── trajectory_analysis.json # 軌跡分析結果
│   └── calibration_report.txt  # 校準報告
├── data/                   # 資料集
├── models/                 # 訓練好的模型
└── requirements.txt        # 依賴套件列表
```

## 技術細節 (Technical Details)

### 攝影機校準 (Camera Calibration)

系統使用47個排球場地參考點進行攝影機校準，完全符合FIVB標準：

- 比賽場地角落 (4個點)
- 網柱 (2個點)
- 天線 (2個點)
- 攻擊線 (4個點)
- 中心線 (2個點)
- 發球區 (4個點)
- 前區邊界 (4個點)
- 後區邊界 (4個點)
- 替換區 (4個點)
- 自由區邊界 (8個點)
- 中場點 (2個點)
- 中寬點 (2個點)
- 高度參考點 (6個點)

### 球體偵測 (Ball Detection)

使用YOLOv11模型進行排球偵測，支援：
- 多種排球顏色（藍色、白色、黃色）
- 高精度偵測
- 即時處理

### 3D重建 (3D Reconstruction)

- 使用三角測量法重建3D軌跡
- 多攝影機時間同步
- 異常值過濾
- 軌跡平滑化

### 軌跡分析 (Trajectory Analysis)

分析結果包括：
- 最大高度
- 總距離
- 平均速度
- 最大速度
- 彈跳點
- 是否在界內（包含自由區）
- 軌跡類型（扣球、舉球、發球、傳球）

### 場地邊界檢查 (Court Boundary Checking)

系統檢查球是否在以下範圍內：
- 比賽場地：18m × 9m
- 自由區：邊線5m，端線6.5m
- 自由比賽空間：12.5m高度限制

## 輸出格式 (Output Format)

### 軌跡分析結果 (JSON)

```json
{
  "trajectory_3d": [[x, y, z], ...],
  "analysis": {
    "max_height": 4.5,
    "min_height": 0.1,
    "total_distance": 15.2,
    "average_speed": 12.3,
    "max_speed": 25.6,
    "bounces": [15, 45],
    "is_in_bounds": true,
    "trajectory_type": "spike"
  },
  "camera_matrices": {...},
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

## 開發計劃 (Development Plan)

- [x] 多攝影機3D重建
- [x] YOLOv11球體偵測
- [x] 攝影機校準工具
- [x] 軌跡分析
- [x] 即時分析
- [x] FIVB標準場地規格
- [ ] 球員偵測整合
- [ ] 動作識別
- [ ] 戰術分析
- [ ] 效能指標計算
- [ ] Web介面

## 授權 (License)

MIT License

## 貢獻 (Contributing)

歡迎提交 Pull Request 或開 Issue 來協助改進這個專案。

## 參考資料 (References)

本專案改編自 [SEU-Robot-Vision-Project-tennis-Hawkeye-system](https://github.com/GehaoZhang6/SEU-Robot-Vision-Project-tennis-Hawkeye-system)，並針對排球運動進行了優化，完全符合國際排球總會(FIVB)標準。 