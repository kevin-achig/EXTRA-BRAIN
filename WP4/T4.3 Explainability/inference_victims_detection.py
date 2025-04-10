import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('/home/christyan/EXTRA/KTH/WP4_explainab/Entrenamiento-300-px-12min/weights/best.pt')

# Path to the .mp4 video
video_path = '/home/christyan/EXTRA/KTH/WP4_explainab/video_CAR-ARENA.mp4'

# Open the video with OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create an object to write the processed video
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference with the YOLOv8 model
    results = model.predict(frame)

    # Draw detection boxes and segmentations on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame (optional)
    cv2.imshow('Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

