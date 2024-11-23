import os
import cv2 as cv
import yt_dlp as youtube_dl
from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import concurrent.futures
import imageio
import re

class VideoProcessor:
    def __init__(self, face_model, body_model):
        self.face_net = cv.dnn.readNetFromCaffe(face_model[0], face_model[1])
        self.body_net = cv.dnn.readNetFromDarknet(body_model[0], body_model[1])

        # Initialize smoothing storage
        self.previous_faces = []
        self.previous_bodies = []

    def download_video(self, video_url, output_filename='temp_video.mp4'):
        # Using yt-dlp to download the video
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_filename,  # Save as 'temp_video.mp4'
            'quiet': True,
            'noplaylist': True,
            'merge_output_format': 'mp4',
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=True)
        
        # Check if the file has been downloaded and exists
        if not os.path.exists(output_filename):
            raise Exception("Video download failed.")
        
        # Open the video file with OpenCV
        cap = cv.VideoCapture(output_filename)
        return cap

    def download_image(self, image_url, output_filename='temp_image.jpg'):
        # Download the image from the URL
        response = requests.get(image_url)
        if response.status_code == 200:
            img_data = response.content
            with open(output_filename, 'wb') as f:
                f.write(img_data)
            img = cv.imread(output_filename)

            # Check if the image is empty
            if img is None:
                raise Exception("Failed to read image from URL.")
            return img
        else:
            raise Exception("Failed to download image.")

    def download_gif(self, gif_url, output_filename='temp_gif.gif'):
        # Download the GIF from the URL
        response = requests.get(gif_url)
        if response.status_code == 200:
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            # Load GIF as a sequence of images
            gif = imageio.mimread(output_filename)
            return gif
        else:
            raise Exception("Failed to download GIF.")

    def process_frame(self, img_np):
        if img_np is None or img_np.size == 0:
            print("Error: Empty image passed to process_frame.")
            return img_np  # Return the original image if it's empty

        try:
            # Step 1: Analyze brightness to adjust preprocessing dynamically
            brightness = cv.mean(cv.cvtColor(img_np, cv.COLOR_BGR2GRAY))[0]
            adaptive_threshold = max(0.11, min(0.6, brightness / 255))

            # Step 2: Prepare input blobs for face and body detection
            face_blob = cv.dnn.blobFromImage(
                img_np, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False
            )
            body_blob = cv.dnn.blobFromImage(
                img_np, 1.0 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False
            )

            # Step 3: Set input for networks and get detections
            self.face_net.setInput(face_blob)
            face_detections = self.face_net.forward()

            self.body_net.setInput(body_blob)
            body_detections = self.body_net.forward()

            img_h, img_w = img_np.shape[:2]  # Image dimensions

            # Step 4: Detect faces (draw bounding boxes on original image)
            detected_faces = []
            for i in range(face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]
                if confidence >= 0.25:
                    x1 = max(0, int(face_detections[0, 0, i, 3] * img_w))
                    y1 = max(0, int(face_detections[0, 0, i, 4] * img_h))
                    x2 = min(img_w, int(face_detections[0, 0, i, 5] * img_w))
                    y2 = min(img_h, int(face_detections[0, 0, i, 6] * img_h))
                    detected_faces.append((x1, y1, x2, y2))

            # Step 5: Detect bodies (draw bounding boxes on original image)
            detected_bodies = []
            for detection in body_detections:
                confidence = detection[4]
                if confidence >= 0.25:
                    center_x = int(detection[0] * img_w)
                    center_y = int(detection[1] * img_h)
                    width = int(detection[2] * img_w)
                    height = int(detection[3] * img_h)

                    x1 = max(0, center_x - width // 2)
                    y1 = max(0, center_y - height // 2)
                    x2 = min(img_w, center_x + width // 2)
                    y2 = min(img_h, center_y + height // 2)

                    aspect_ratio = height / width
                    if aspect_ratio >= 1.2:  # Exclude non-body objects
                        detected_bodies.append((x1, y1, x2, y2))

            # Step 6: Apply Non-Maximum Suppression (NMS) to reduce overlapping body detections
            nms_indices = cv.dnn.NMSBoxes(
                [box for box in detected_bodies], [1] * len(detected_bodies), 0.50, 0.4
            )

            # Step 7: Smooth bounding boxes and predict zones
            smoothed_faces = self.smooth_detections(self.previous_faces, detected_faces)
            smoothed_bodies = self.smooth_detections(self.previous_bodies, detected_bodies)

            # Step 8: Draw bounding boxes on faces and bodies (colored for both)
            for (x1, y1, x2, y2) in smoothed_faces:
                cv.rectangle(img_np, (x1, y1), (x2, y2), (200, 0, 0), 2)
                cv.putText(img_np, 'Face', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if len(nms_indices) > 0:
                for i in nms_indices.flatten():
                    (x1, y1, x2, y2) = smoothed_bodies[i]
                    cv.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for bodies
                    cv.putText(img_np, 'Body', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Store the current detections for smoothing in the next frame
            self.previous_faces = smoothed_faces
            self.previous_bodies = smoothed_bodies

            # Return the processed frame
            return img_np

        except cv.error as e:
            print(f"OpenCV error in process_frame: {e}")
        except Exception as e:
            print(f"Unexpected error in process_frame: {e}")

        # Return the original frame in case of error
        return img_np

    def smooth_detections(self, previous_detections, current_detections, smoothing_factor=0.5):
        smoothed_detections = []
        if len(previous_detections) != len(current_detections):
            smoothed_detections = current_detections
        else:
            for prev, curr in zip(previous_detections, current_detections):
                x1 = int(smoothing_factor * prev[0] + (1 - smoothing_factor) * curr[0])
                y1 = int(smoothing_factor * prev[1] + (1 - smoothing_factor) * curr[1])
                x2 = int(smoothing_factor * prev[2] + (1 - smoothing_factor) * curr[2])
                y2 = int(smoothing_factor * prev[3] + (1 - smoothing_factor) * curr[3])
                smoothed_detections.append((x1, y1, x2, y2))
        return smoothed_detections

def is_youtube_url(url):
    # Regular expression for YouTube URL
    return re.match(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.*', url)

def is_image_url(url):
    # Check if URL ends with an image extension
    return url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))

def main():
    # Ask for user input
    url = input("Please enter a video, image URL, or leave blank for no URL: ").strip()

    # Model paths (assumed you have downloaded the models already)
    face_model = ('hidden/deploy.prototxt', 'hidden/res10_300x300_ssd_iter_140000.caffemodel')
    body_model = ('hidden/yolov4.cfg', 'hidden/yolov4.weights')

    processor = VideoProcessor(face_model, body_model)

    if url == "":
        print("No URL entered. Exiting.")
        return

    try:
        if is_youtube_url(url):
            print("YouTube URL detected. Downloading video...")
            cap = processor.download_video(url)
        elif is_image_url(url):
            print("Image URL detected. Downloading image...")
            img = processor.download_image(url)
            
            # Process the image to draw bounding boxes
            processed_image = processor.process_frame(img)
            
            # Show the processed image with bounding boxes
            cv.imshow("Processed Image", processed_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
            return  # Exit after showing the processed image
        else:
            print("Unsupported URL format.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = processor.process_frame(frame)
            cv.imshow("Processed Video", processed_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()
