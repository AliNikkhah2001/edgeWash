# Import the InferencePipeline object
from inference import InferencePipeline
import cv2
import os
from dotenv import load_dotenv
from step_tracker import StepTracker
from status_display import create_status_image

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
WORKFLOW_ID = os.getenv("WORKFLOW_ID")

# Initialize the step tracker
tracker = StepTracker(target_duration_seconds=10)

def my_sink(result, video_frame):
    # The 'result' dict contains a 'predictions' key, which holds another dict.
    # This inner dict has a 'top' key with the most likely class.
    predicted_class = "None"
    if "predictions" in result:
        predicted_class = result["predictions"].get("top", "None")

    # Update the tracker with the latest prediction
    tracker.update(predicted_class)

    # Get the latest status from the tracker
    durations, target = tracker.get_status()
    all_done = tracker.all_steps_complete()

    # Create and display the status GUI
    status_gui = create_status_image(durations, target, all_done)
    cv2.imshow("WashWise Status", status_gui)

    # --- Display the live camera feed with user-friendly prediction text ---
    # Mapping from internal step names to user-friendly display names
    display_names = {
        "palms togethers": "Palms Together",
        "right palm on left dorsum": "Right Palm on Left Dorsum",
        "left palm on right dorsum": "Left Palm on Right Dorsum",
        "fingers interlaced": "Fingers Interlaced",
        "right nails": "Right Nails",
        "left nails": "Left Nails",
        "right thumb": "Right Thumb",
        "left thumb": "Left Thumb"
    }

    # Normalize the predicted class for display
    display_label = display_names.get(predicted_class.lower(), predicted_class.replace('_', ' ').title())

    # Make a copy to draw on
    frame_to_display = video_frame.image.copy()

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Prediction: {display_label}"
    text_color = (255, 255, 255) # White
    bg_color = (0, 0, 0) # Black
    
    # Get text size to draw a background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.8, 2)
    
    # Draw background rectangle and text
    cv2.rectangle(frame_to_display, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 5), bg_color, -1)
    cv2.putText(frame_to_display, text, (15, 10 + text_height + 5), font, 0.8, text_color, 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Live Feed", frame_to_display)
    cv2.waitKey(1)


# initialize a pipeline object
print("Initializing pipeline...")
pipeline = InferencePipeline.init_with_workflow(
    api_key=API_KEY,
    workspace_name=WORKSPACE_NAME,
    workflow_id=WORKFLOW_ID,
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)

print("Starting pipeline...")
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
print("Pipeline finished.")
cv2.destroyAllWindows()



