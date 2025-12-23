import cv2
import numpy as np

def create_status_image(durations, target_duration, all_complete):
    """
    Creates an image to display the status of the handwashing steps.
    """
    # Image dimensions and colors
    width, height = 800, 600
    background_color = (20, 20, 20)  # Dark grey
    text_color = (230, 230, 230)     # Light grey
    bar_background = (50, 50, 50)
    bar_color_incomplete = (220, 180, 0) # Yellow
    bar_color_complete = (0, 200, 80)     # Green
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Create a black image
    status_img = np.full((height, width, 3), background_color, dtype=np.uint8)

    # --- Title ---
    cv2.putText(status_img, "Handwashing Progress", (50, 60), font, 1.5, text_color, 3, cv2.LINE_AA)

    if all_complete:
        # --- Completion Message ---
        cv2.putText(status_img, "All Steps Complete!", (150, 250), font, 2, bar_color_complete, 4, cv2.LINE_AA)
        cv2.putText(status_img, "You are good to go!", (200, 350), font, 1.5, text_color, 3, cv2.LINE_AA)
    else:
        # --- Progress Bars ---
        y_pos = 120
        for step, duration in durations.items():
            # Draw the background of the progress bar
            cv2.rectangle(status_img, (50, y_pos), (width - 50, y_pos + 40), bar_background, -1)

            # Calculate progress
            progress = min(duration / target_duration, 1.0) # Cap at 100%
            bar_width = int(progress * (width - 100))
            
            # Determine bar color
            bar_color = bar_color_complete if progress >= 1.0 else bar_color_incomplete

            # Draw the progress bar
            if bar_width > 0:
                cv2.rectangle(status_img, (50, y_pos), (50 + bar_width, y_pos + 40), bar_color, -1)

            # Draw the text label for the step
            label = f"{step.replace('_', ' ').title()}: {duration:.1f}s / {target_duration}s"
            cv2.putText(status_img, label, (60, y_pos + 28), font, 0.7, text_color, 2, cv2.LINE_AA)
            
            y_pos += 60

    return status_img
