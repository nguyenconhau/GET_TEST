import cv2
import torch
from models.model_utils import create_model
from utils.detect import detect_objects

# Load the model
configs = {'pretrained_path': 'E:\Project\pythonProject\Objec_De\SFA3D\checkpoints\fpn_resnet_18\fpn_resnet_18_epoch_300.pth'}
model = create_model(configs)
model.eval()

# Open the video file
cap = cv2.VideoCapture('test.avi')

while(cap.isOpened()):
    # Read a frame from the video
    ret, frame = cap.read()

    # Convert the frame to a tensor
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()

    # Detect objects in the frame
    detections = detect_objects(model, frame_tensor)

    # Draw bounding boxes around the detected objects
    for detection in detections:
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()