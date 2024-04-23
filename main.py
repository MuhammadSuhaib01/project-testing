import cv2
from ultralytics import YOLO
import torch

torch.cuda.set_device(0) # Set to your desired GPU number

# Load your model
model = YOLO("yolov8s-pose.pt") 


# Open the video
cap = cv2.VideoCapture("gymvideo.mp4")

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run model prediction
    results =  model('gymvideo.mp4',stream=False)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs      
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

    # Extract keypoints
    keypoints = results[0].keypoints

    print(keypoints)

    # Draw keypoints on the frame
    # for kpt in keypoints.xyxy:
    #     x = int(kpt[0][0]) 
    #     y = int(kpt[0][1])
    #     cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # Show the frame with keypoints
    cv2.imshow('Keypoints without lines', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
