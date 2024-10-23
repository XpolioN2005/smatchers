import cv2

def detect_contours(fgmask):
    """Find and return contours from the foreground mask."""
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) > 500]
