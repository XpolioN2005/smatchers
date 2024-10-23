import cv2

def create_background_subtractor():
    """Initialize and return a background subtractor."""
    return cv2.createBackgroundSubtractorMOG2()
