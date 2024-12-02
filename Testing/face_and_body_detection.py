import os
import cv2 as cv
import yt_dlp as youtube_dl
from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
import asyncio
import logging


class Processor:
    def __init__(self, face_model, body_model):
        self.face_net = cv.dnn.readNetFromCaffe(face_model[0], face_model[1])
        self.body_net = cv.dnn.readNetFromDarknet(body_model[0], body_model[1])
        self.previous_faces = []
        self.previous_bodies = []

    async def download_media(self, media_url, output_filename):
        """Download any media type (image, GIF, or video)."""
        try:
            response = await asyncio.to_thread(requests.get, media_url, stream=True)
            if response.status_code == 200:
                with open(output_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Media downloaded: {output_filename}")
            else:
                raise Exception(f"Failed to download media: {media_url} (HTTP {response.status_code})")
        except Exception as e:
            raise ValueError(f"Error downloading media: {e}")

    async def process_gif(self, frames, durations):
        """Process each frame of the GIF and return the processed frames."""
        processed_frames = []
        # Use tqdm to show a progress bar while iterating through frames
        for frame in tqdm(frames, desc="Drawling on frames", unit="frame"):
            processed_frame = await asyncio.to_thread(self.process_frame, frame)
            processed_frames.append(processed_frame)
        await self.save_gif_with_detections(processed_frames, durations)
        return processed_frames, durations

    def process_frame(self, img_np):
     #print(img_np)
     if img_np is None or img_np.size == 0:
        print("Error: Empty image passed to process_frame.")
        return img_np  # Return the original image if it's empty

     try:
        # Ensure the image has 3 channels (RGB)
        if len(img_np.shape) == 2:  # Grayscale image
            img_np = cv.cvtColor(img_np, cv.COLOR_GRAY2BGR)

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
            if confidence >= 0.25:  # Adjust confidence threshold for face detection
                x1 = max(0, int(face_detections[0, 0, i, 3] * img_w))
                y1 = max(0, int(face_detections[0, 0, i, 4] * img_h))
                x2 = min(img_w, int(face_detections[0, 0, i, 5] * img_w))
                y2 = min(img_h, int(face_detections[0, 0, i, 6] * img_h))
                detected_faces.append((x1, y1, x2, y2))

        # Step 5: Detect bodies (draw bounding boxes on original image)
        detected_bodies = []
        for detection in body_detections:
            confidence = detection[4]
            if confidence >= 0.3:  # Adjust confidence threshold for body detection
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
            cv.rectangle(img_np, (x1, y1), (x2, y2), (200, 0, 0), 2)  # Red for faces
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
    
    def process_image(self, image_path):
     """
     Process an image, perform face and body detection, and save the result as 'detected.jpeg'.
     """
     try:
        # Load the image from the provided path
        img_np = cv.imread(image_path)
        if img_np is None or img_np.size == 0:
            print("Error: Failed to load image or empty image passed.")
            return None  # Return None to indicate failure

        # Ensure the image has 3 channels (convert grayscale to RGB if necessary)
        if len(img_np.shape) == 2:
            img_np = cv.cvtColor(img_np, cv.COLOR_GRAY2BGR)

        # Analyze brightness for preprocessing
        brightness = cv.mean(cv.cvtColor(img_np, cv.COLOR_BGR2GRAY))[0]
        adaptive_threshold = max(0.109, min(0.6, brightness / 255))  # Adjust confidence based on brightness


        # Prepare input blobs for detection
        face_blob = cv.dnn.blobFromImage(
            img_np, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False
        )
        body_blob = cv.dnn.blobFromImage(
            img_np, 1.0 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False
        )

        # Run face detection
        self.face_net.setInput(face_blob)
        face_detections = self.face_net.forward()

        # Run body detection
        self.body_net.setInput(body_blob)
        body_detections = self.body_net.forward()

        img_h, img_w = img_np.shape[:2]

        # Detect faces
        detected_faces = []
        for i in range(face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence < 0:  # Skip weak detections
                continue

            # Compute bounding box
            x1 = max(0, int(face_detections[0, 0, i, 3] * img_w))
            y1 = max(0, int(face_detections[0, 0, i, 4] * img_h))
            x2 = min(img_w, int(face_detections[0, 0, i, 5] * img_w))
            y2 = min(img_h, int(face_detections[0, 0, i, 6] * img_h))

            # Draw initial bounding box
            cv.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(img_np, 'Face Detected', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Detect bodies
        detected_bodies = []
        for detection in body_detections:
            confidence = detection[4]
            if confidence >= 0.3:
                center_x = int(detection[0] * img_w)
                center_y = int(detection[1] * img_h)
                width = int(detection[2] * img_w)
                height = int(detection[3] * img_h)

                x1 = max(0, center_x - width // 2)
                y1 = max(0, center_y - height // 2)
                x2 = min(img_w, center_x + width // 2)
                y2 = min(img_h, center_y + height // 2)

                aspect_ratio = height / width
                if aspect_ratio >= 1.2:
                    detected_bodies.append((x1, y1, x2, y2))

        # Smooth detections
        smoothed_faces = self.smooth_detections(self.previous_faces, detected_faces)
        smoothed_bodies = self.smooth_detections(self.previous_bodies, detected_bodies)

        # Draw bounding boxes
        for (x1, y1, x2, y2) in smoothed_faces:
            cv.rectangle(img_np, (x1, y1), (x2, y2), (200, 0, 0), 2)
            cv.putText(img_np, 'Face', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for (x1, y1, x2, y2) in smoothed_bodies:
            cv.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(img_np, 'Body', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the processed image
        output_path = "output/detected.jpeg"
        cv.imwrite(output_path, img_np)

        # Display the saved image in a window
        cv.imshow("Detected Image", img_np)
        cv.waitKey(0)  # Wait for a key press to close the window
        cv.destroyAllWindows()  # Close the window after the key press

        # Store detections for smoothing in the next frame
        self.previous_faces = smoothed_faces
        self.previous_bodies = smoothed_bodies

        return img_np

     except cv.error as e:
        print(f"OpenCV error in process_image: {e}")
     except Exception as e:
        print(f"Unexpected error in process_image: {e}")

     return None
   
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

    async def save_gif_with_detections(self, frames, durations):
     """Save the processed frames back into a GIF with detection boxes."""
     processed_frames = []
    
     # Use tqdm to show a progress bar while iterating through frames
     for frame in tqdm(frames, desc="Saving frames", unit="frame"):
        # Convert the frame from a NumPy array to a PIL Image
        pil_frame = Image.fromarray(frame)

        # Optionally, save each processed frame or add to a list
        processed_frames.append(pil_frame)

     # Save the processed GIF
     output_gif = "output/processed_media.gif"
     processed_frames[0].save(output_gif, save_all=True, append_images=processed_frames[1:], loop=0, duration=durations)

     print(f"Processed GIF saved as {output_gif}")
     return output_gif
    async def process_media(self, media_url):
        """Main function to detect and process media from a URL."""
        try:
            # Detect the media type (image, gif, video)
            media_type = await asyncio.to_thread(self.detect_media_type, media_url)
            print(f"Detected media type: {media_type}")

            # Output filename, using .jpg extension for images
            output_filename = f"temp.{media_type if media_type != 'image' else 'jpg'}"

            # Create an instance of Processor (ensure this is correctly set up)
            processor = Processor(
                face_model=('hidden/deploy.prototxt', 'hidden/res10_300x300_ssd_iter_140000.caffemodel'),
                body_model=('hidden/yolov4.cfg', 'hidden/yolov4.weights')
            )

            # Download the media
            await processor.download_media(media_url, output_filename)

            if media_type == 'image':
                # Process the image using OpenCV or PIL to ensure it's correctly loaded
                img = await asyncio.to_thread(processor.process_frame, output_filename)
                #print(f"Image type after processing: {type(img)}")

                if isinstance(img, str):
                    raise ValueError("Image processing returned a string instead of a NumPy array.")
                if img is None:
                    raise Exception("Image processing failed.")

                img_np = np.array(img) if not isinstance(img, np.ndarray) else img
                #print(f"Image data as NumPy array: {img_np.shape}")

                return img_np

            elif media_type == 'gif':
                # Handle GIF-specific logic
                print("Processing GIF...")
                gif = Image.open(output_filename)
                frames = []
                durations = []  # To store the durations of each frame

                # Use tqdm to show a progress bar while iterating through frames
                for frame in tqdm(range(gif.n_frames), desc="Processing frames", unit="frame"):
                    gif.seek(frame)
                    frame_image = np.array(gif.convert('RGB'))  # Convert to RGB for further processing
                    frames.append(frame_image)
                    durations.append(gif.info['duration'])

                processed_frames, durations = await processor.process_gif(frames, durations)
                return processed_frames, durations

            elif media_type == 'video':
                # Handle video-specific logic
                print("Processing video...")
                # Implement the video processing logic here as per your requirements
                pass

        except Exception as e:
            print(f"Error processing media: {e}")

    def detect_media_type(self, media_url):
        """Detects the media type from the URL or file extension."""
        if media_url.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            return 'image'
        elif media_url.lower().endswith('.gif'):
            return 'gif'
        elif media_url.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            return 'video'
        else:
            raise ValueError("Unsupported media type")

def detect_media_type(media_url):
    """Detect the type of media (image, gif, or video) based on the file extension."""
    # List of supported image, gif, and video extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    gif_extensions = ['.gif']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    # Get the file extension from the URL
    file_extension = os.path.splitext(media_url)[-1].lower()

    if file_extension in image_extensions:
        return 'image'
    elif file_extension in gif_extensions:
        return 'gif'
    elif file_extension in video_extensions:
        return 'video'
    else:
        raise ValueError(f"Unsupported media type: {file_extension}")
    
async def main():
    while True:
     media_url = input("Enter the media URL ('exit' to quit): ")
     if media_url != 'exit':
      processor = Processor(face_model=('hidden/deploy.prototxt', 'hidden/res10_300x300_ssd_iter_140000.caffemodel'), body_model=('hidden/yolov4.cfg', 'hidden/yolov4.weights'))
      result = await processor.process_media(media_url)
     else:
         break
     # print("Processed Media Result:", result)

# Execute main asynchronously
if __name__ == '__main__':
    asyncio.run(main())
