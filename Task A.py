import cv2
import numpy as np
import matplotlib.pyplot as plt

# === DISABLE OpenCL to avoid memory errors ===
cv2.ocl.setUseOpenCL(False)

# === PARAMETERS ===
BRIGHTNESS_THRESHOLD = 100

# === USER INPUT ===
#video_path = input("Enter video path (e.g., 'street.mp4'): ").strip()

# === OPEN VIDEO ===
#cap = cv2.VideoCapture(video_path)
#if not cap.isOpened():
 #   print("Error: Could not open video file.")
  #  exit()

# Input video that need blurring effect
input_file = "street.mp4"
video = cv2.VideoCapture(input_file)

# Set Output video properties
output_file =("processed_video.avi",
              cv2.VideoWriter_fourcc(*"MJPG"),
              30.0,
              (1280, 720))

# Create the actual VideoWriter using the tuple
out = cv2.VideoWriter(*output_file)

# === READ FIRST 50 FRAMES TO CHECK BRIGHTNESS ===
brightness_values = []
for _ in range(50):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    brightness_values.append(brightness)

cap.release()

# === CHECK IF FRAMES WERE READ ===
if not brightness_values:
    print("Error: No frames read from the video.")
    exit()

# === CALCULATE AVERAGE BRIGHTNESS ===
average_brightness = np.mean(brightness_values)

print("Average Brightness:", average_brightness)

# === CLASSIFY DAY OR NIGHT ===
if average_brightness >= BRIGHTNESS_THRESHOLD:
    print("This is likely a daytime video.")
else:
    print("This is likely a nighttime video.")

# === PLOT HISTOGRAM OF BRIGHTNESS ===
plt.figure(figsize=(8, 5))
plt.hist(brightness_values, bins=20, color='blue', edgecolor='black')
plt.title('Brightness Distribution (First 50 Frames)')
plt.xlabel('Average Brightness per Frame')
plt.ylabel('Number of Frames')
plt.grid(True)
plt.tight_layout()
plt.show()

#Blur Effect
# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("face_detector.xml")

# Get total number of frames
total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

print("Processing video frame by frame... Press 'q' to quit early.")

# === Frame-by-frame processing ===
for frame_count in range(0, int(total_frames)):
    success, frame = video.read()
    if not success:
        print(f"Failed to read frame {frame_count}")
        continue

    # Resize frame to match output resolution
    frame = cv2.resize(frame, (1280, 720))

    # Convert frame to grayscale for face detection (Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # Blur all detected face regions
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
        frame[y:y+h, x:x+w] = blurred_face

    # Save the processed frame to output video
    out.write(frame)

    # Show the frame live while processing (optional)
    cv2.imshow("Face Blurring Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

# Release all resources
video.release()
out.release()
cv2.destroyAllWindows()

print(f"Face blurring complete. Output saved as '{output_file}'")
