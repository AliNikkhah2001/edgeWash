# WashWise - Real-Time Handwashing Step Tracker

## Overview
YOLOv8-based demo from Roboflow for real-time WHO step classification on sink camera footage with Streamlit GUI.

## Code Structure
- **Type**: **Uses Roboflow SDK** (cloud-based inference)
- **Implementation**: Client application using Roboflow InferencePipeline
- **Code Included**: Yes - inference and GUI code
- **Dependencies**: roboflow-inference, opencv-python, python-dotenv

## Model Architecture
- **Model**: **YOLOv8** via Roboflow Workflows
- **Approach**: Real-time video classification (frame-by-frame)
- **Training Location**: Roboflow cloud platform (not in codebase)
- **Deployment**: InferencePipeline (streaming inference)

## Video/Temporal Handling
- **Temporal Model**: None - purely frame-by-frame predictions
- **3D Convolutions**: No
- **Sequence Modeling**: No (RNN/LSTM/GRU)
- **Duration Tracking**: External logic via `StepTracker` class
- **Time Series Steps**: Accumulates time per step (10 seconds target per step)
- **Real-time Processing**: Up to 30 FPS via Roboflow pipeline

## Classes & WHO Steps
- **Total Classes**: 8 WHO handwashing steps
  - Palms together
  - Right palm on left dorsum
  - Left palm on right dorsum
  - Fingers interlaced
  - Right nails
  - Left nails
  - Right thumb
  - Left thumb
- **WHO Coverage**: All 8 core movements

## Datasets Used
- **Training Data**: Custom dataset trained on Roboflow platform
- **Format**: Uploaded to Roboflow (not included in repository)
- **Public Availability**: No (user must provide own Roboflow workflow)

## Training Details
- **Training Script**: None (training done on Roboflow platform)
- **Hyperparameters**: Configured via Roboflow UI (not exposed in code)
- **Model Access**: Requires Roboflow API credentials
- **Pre-trained Weights**: Hosted on Roboflow (not downloadable directly)

## Running Instructions

### 1. Setup Environment
```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Roboflow Credentials
Edit `.env` file:
```bash
API_KEY=your_roboflow_api_key
WORKSPACE_NAME=your_workspace_name
WORKFLOW_ID=your_deployed_workflow_id
```

### 3. Run Application
```bash
python main.py
```

### 4. Usage
- Application opens two OpenCV windows:
  - **Live Feed**: Shows webcam with current step prediction
  - **WashWise Status**: Progress bars for each step (10-second target)
- Perform handwashing steps in any order
- Each step needs 10 seconds of accumulated time
- Completion message displays when all steps are done

## Key Features
- **Real-time inference** (up to 30 FPS)
- **Cloud-based model** (no local training required)
- **Flexible order** (steps can be completed non-sequentially)
- **Visual progress tracking** (OpenCV GUI with progress bars)
- **Configurable duration** (target_duration_seconds parameter)

## Limitations
- **Cloud-dependent** (requires Roboflow API and internet)
- **No temporal modeling** - frame-by-frame predictions only
- **No local training** - must use Roboflow platform
- **API costs** - Roboflow inference charges apply
- **No offline mode** - requires constant API connection

## Customization Options
- **Step Duration**: Change `target_duration_seconds` in `main.py`
- **Step Names**: Update `steps` set in `step_tracker.py`
- **UI Colors**: Modify color values in `status_display.py`
- **FPS**: Adjust `max_fps` parameter in `main.py`

## Availability
- **Code**: Open (license not specified)
- **Dataset**: Not included (must provide own)
- **Trained Weights**: Hosted on Roboflow (requires API access)
- **External API**: Yes (Roboflow)
