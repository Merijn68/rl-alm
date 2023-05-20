import cv2
import numpy as np

# Specify the video source (0 for webcam, or provide a video file path)
video_source = 0
cap = cv2.VideoCapture(video_source)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform any necessary processing on the frame
    # For example, you can resize, draw overlays, or apply filters

    # Display the resulting frame
    cv2.imshow('System State', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
