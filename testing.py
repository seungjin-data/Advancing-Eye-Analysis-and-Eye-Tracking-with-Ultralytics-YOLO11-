import cv2 
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO(r'Eye analysis.pt')

# Path to the video file
video_path = r'eye.mp4'

# Start video capture
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Desired processing size
target_width = 1280  # Set for HD resolution
target_height = 720

# Output video properties
fps = cap.get(cv2.CAP_PROP_FPS)
output_filename = 'output_video.mp4'

# Define the video writer for .mp4 format with H.264 codec
output_video = cv2.VideoWriter(
    output_filename,
    cv2.VideoWriter_fourcc(*'mp4v'),  # Change to 'H264' if needed
    fps,
    (target_width, target_height)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read the frame.")
        break

    # Resize the frame to HD (1280x720) resolution
    frame = cv2.resize(frame, (target_width, target_height))

    # Perform detection
    try:
        results = model.predict(frame, verbose=False)
        annotated_frame = frame.copy()

        # Add color annotations for iris and pupil
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == 0:  # Assuming class 0 is iris
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Yellow
                    cv2.putText(annotated_frame, 'Iris', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 4)
                elif cls == 1:  # Assuming class 1 is pupil
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 0, 128), 4)  # Purple
                    cv2.putText(annotated_frame, 'Pupil', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 4)
                    
                    # Segmentation for the pupil
                    pupil_region = frame[y1:y2, x1:x2]
                    mask = np.zeros_like(pupil_region)
                    mask[:, :] = (255, 0, 255)  # Magenta color for segmentation
                    annotated_frame[y1:y2, x1:x2] = cv2.addWeighted(pupil_region, 0.5, mask, 0.5, 0)

    except Exception as e:
        print(f"Error during model prediction: {e}")
        break

    # Display the frame
    cv2.imshow('Eye analysis', annotated_frame)

    # Write the frame to the output video
    output_video.write(annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Video processing completed. Output saved as {output_filename}")
