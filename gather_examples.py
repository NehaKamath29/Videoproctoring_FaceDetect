import cv2
import os
import numpy as np

print("Start gathering")

def generate_images(input_live_video_dir, input_non_live_video_dir, output_live_dir, output_non_live_dir, detector_path, confidence=0.5, skip=0):
    # Load face detector model
    protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
    modelPath = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Function to process video files and generate images
    def process_video(input_video_dir, output_image_dir, category):
        os.makedirs(output_image_dir, exist_ok=True)

        for video_file in os.listdir(input_video_dir):
            video_path = os.path.join(input_video_dir, video_file)
            vs = cv2.VideoCapture(video_path)
            saved = 0
            frame_number = 0
            print("Starting")

            while True:
                grabbed, frame = vs.read()
                if not grabbed:
                    break

                frame_number += 1
                if skip > 0 and frame_number % skip != 0:
                    continue

                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()

                if len(detections) > 0:
                    i = np.argmax(detections[0, 0, :, 2])
                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_threshold:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        face = frame[startY:endY, startX:endX]

                        # Check if the face region is valid
                        if face is not None and face.size > 0:
                            p = os.path.sep.join([output_image_dir, f"{category}_{video_file}_frame_{saved}.png"])
                            cv2.imwrite(p, face)
                            saved += 1

            vs.release()
            print("Done")

    # Process live videos
    process_video(input_live_video_dir, output_live_dir, "live")

    # Process non-live videos
    process_video(input_non_live_video_dir, output_non_live_dir, "non_live")

# Define paths to input videos, output directories, and face detector model
input_live_video_dir = "liveness_dataset/live"
input_non_live_video_dir = "liveness_dataset/non_live"
output_live_dir = "liveness_output_img/live"
output_non_live_dir = "liveness_output_img/non_live"
detector_path = "detector"

# Set confidence threshold and skip parameters
confidence_threshold = 0.5
skip = 0  # Set to 0 to process all frames

# Call the function to generate images
generate_images(input_live_video_dir, input_non_live_video_dir, output_live_dir, output_non_live_dir, detector_path, confidence_threshold, skip)
print("Done gathering!")
