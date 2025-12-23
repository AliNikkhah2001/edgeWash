
# WashWise - Handwashing Step Tracker

WashWise is a real-time handwashing step tracker built with Python and Roboflow. It uses your webcam and a custom-trained Roboflow workflow to recognize 8 handwashing steps, track the time spent on each, and display progress in a modern OpenCV GUI.


## Features

- **8 Handwashing Steps Tracking**: Monitors all essential handwashing techniques
- **Real-time Video Processing**: Uses Roboflow Inference Pipeline for efficient video processing
- **Workflow Integration**: Connects to your deployed Roboflow workflow
- **Progress Tracking**: Visual progress bars and time accumulation for each step
- **Flexible Order**: Steps can be completed in any order
- **Completion Message**: "You are good to go!" when all steps are completed


## Setup Instructions

### 1. Install Dependencies

All required packages are listed in `requirements.txt`:

```
roboflow-inference
opencv-python
python-dotenv
numpy
```

Install everything with:
```bash
pip install -r requirements.txt
```


### 2. Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and replace the placeholder values with your actual Roboflow details:
   - `API_KEY`: Your Roboflow API key (from account settings)
   - `WORKSPACE_NAME`: Your workspace name
   - `WORKFLOW_ID`: Your deployed workflow ID

3. **Important**: Never commit the `.env` file to version control as it contains sensitive API credentials.


### 3. Run the Application

```bash
python main.py
```


## How to Use

1. **Start the Application**: Run `python main.py` to start the tracker
2. **Camera Window**: An OpenCV window will show the live camera feed with the current step prediction
3. **Status Window**: A separate OpenCV window displays progress bars for each step
4. **Perform Handwashing Steps**: The app will detect which step you're performing in real time
5. **Watch Progress**: Each step needs 10 seconds of accumulated time
6. **Complete All Steps**: When all steps reach 10 seconds, you'll see a completion message


## Technical Details

### Inference Pipeline
- Uses Roboflow's InferencePipeline for efficient real-time video processing
- Processes video frames at up to 30 FPS
- Automatically handles camera input and prediction workflow
- Displays results in OpenCV windows

### Architecture
- **main.py**: Entry point, sets up the pipeline and GUI
- **step_tracker.py**: Tracks cumulative time for each handwashing step
- **status_display.py**: Generates the progress bar GUI
- **Prediction Callback**: Updates tracker and GUI for each frame


## Customization

- **Step Duration**: Change the `target_duration_seconds` argument in `StepTracker` (in `main.py`)
- **Step Names**: Update the `steps` set in `step_tracker.py` and the display mapping in `main.py`/`status_display.py`
- **UI Colors**: Modify color values in `status_display.py`


## Troubleshooting

- **Environment Issues**: Ensure your `.env` file exists and contains all required variables
- **Camera Issues**: Ensure your camera is not being used by another application
- **Inference Pipeline Errors**: Check your API key, workspace name, and workflow ID in the `.env` file
- **Performance**: Lower the FPS in `main.py` if experiencing lag
- **OpenCV Window**: The camera feed and status appear in separate windows - this is normal behavior
- **Missing .env file**: Copy `.env.example` to `.env` and configure your credentials


## Security Notes

- **Never commit your `.env` file** to version control (it's included in `.gitignore`)
- Keep your API key secure and don't share it publicly
- Use environment variables for all sensitive configuration


## Requirements

- Python 3.8+
- Webcam/Camera
- Internet connection (for Roboflow API calls)
- Roboflow account with deployed workflow
