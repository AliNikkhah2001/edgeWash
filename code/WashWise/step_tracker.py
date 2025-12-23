import time

class StepTracker:
    """
    A class to track the progress of a multi-step process, like handwashing.
    It measures the cumulative duration each step is performed and checks against a target time.
    """
    def __init__(self, target_duration_seconds=10):
        """
        Initializes the StepTracker.
        """
        self.steps = {
            "palms togethers",
            "right palm on left dorsum",
            "left palm on right dorsum",
            "fingers interlaced",
            "right nails",
            "left nails",
            "right thumb",
            "left thumb"
        }
        self.target_duration = target_duration_seconds
        self.durations = {step: 0 for step in self.steps}
        
        self.current_step = None
        self.current_step_start_time = None

    def update(self, detected_step):
        """
        Updates the duration for a detected step.
        """
        # Roboflow might return class names with underscores or be 'None'
        normalized_step = "none"
        if detected_step:
            normalized_step = detected_step.replace("_", " ").lower()

        # If the detected step is not one we are tracking, or it's "None", it marks the end of a pose.
        if normalized_step not in self.steps:
            if self.current_step is not None:
                # Add the duration of the last pose to its total
                duration = time.time() - self.current_step_start_time
                self.durations[self.current_step] += duration
                print(f"Stopped tracking '{self.current_step}'. Total duration: {self.durations[self.current_step]:.2f}s")
            
            # Reset current step since no valid pose is detected
            self.current_step = None
            self.current_step_start_time = None
            return

        # If this is a new, valid step being detected
        if normalized_step != self.current_step:
            # First, log the time for the previous step if there was one
            if self.current_step is not None:
                duration = time.time() - self.current_step_start_time
                self.durations[self.current_step] += duration
                print(f"Stopped tracking '{self.current_step}'. Total duration: {self.durations[self.current_step]:.2f}s")

            # Then, start the timer for the new step
            self.current_step = normalized_step
            self.current_step_start_time = time.time()
            print(f"Started tracking: '{self.current_step}'")

    def get_status(self):
        """
        Returns the current durations and target for display purposes.
        Includes the actively accumulating time for the current step.
        """
        display_durations = self.durations.copy()
        if self.current_step is not None:
            elapsed = time.time() - self.current_step_start_time
            display_durations[self.current_step] += elapsed
        return display_durations, self.target_duration

    def all_steps_complete(self):
        """
        Checks if all steps have reached their target duration.
        """
        for step in self.steps:
            if self.durations[step] < self.target_duration:
                return False
        return True
