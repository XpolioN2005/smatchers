#main

import cv2
from util.background_subtractor import create_background_subtractor
from util.contour_detection import detect_contours
from util.display import display_frame

def main(vid=0):
    # Initialize video capture
    cap = cv2.VideoCapture(vid)

    # Create background subtractor
    fgbg = create_background_subtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction and get contours
        fgmask = fgbg.apply(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0))
        contours = detect_contours(fgmask)

        # Draw rectangles around detected objects and display the frame
        frame = display_frame(frame, contours)
        cv2.imshow('Flying Object Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)
