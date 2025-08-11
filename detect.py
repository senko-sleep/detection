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
import tempfile
import pickle
import concurrent.futures


class Processor:
    def __init__(self, face_model, body_model):
        self.face_net = cv.dnn.readNetFromCaffe(face_model[0], face_model[1])
        self.body_net = cv.dnn.readNetFromDarknet(body_model[0], body_model[1])
        self.previous_faces = []
        self.previous_bodies = []
        self.face_features = []  # Store face features for learning
        self.tracked_face = None
        self.tracked_body = None

        # Load Haar Cascade Classifiers for face, body, eyes, and head
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Face detection
        self.body_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')  # Body detection
        self.eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')  # Eye detection
        self.head_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Head detection (same as face)
        # Directory to save face images
        self.saved_faces_dir = "saved_faces"
        if not os.path.exists(self.saved_faces_dir):
            os.makedirs(self.saved_faces_dir)
            
        # Initialize variables for tracking face coordinates
        self.tracked_face = None  # This will store the coordinates of the fixed face
        self.saved_faces = self.load_saved_faces()

            
    def save_face_image(self, img_np, face_coords, face_index):
     x1, y1, x2, y2 = face_coords
     face_image = img_np[y1:y2, x1:x2]
    
     # Resize face image to a consistent size before saving
     face_image_resized = cv.resize(face_image, (128, 128))  # Resize to a standard size
    
     filename = f"{self.saved_faces_dir}/face_{face_index}.jpg"
     cv.imwrite(filename, face_image_resized)
     return face_image_resized
 
    def load_saved_faces(self):
        # Attempt to load saved faces from file, if it exists
        saved_faces = []
        try:
            with open('saved_faces.pkl', 'rb') as f:
                saved_faces = pickle.load(f)
        except FileNotFoundError:
            pass
        return saved_faces
    
    def save_faces_to_file(self):
        # Save the list of faces to a file using pickle
        with open('saved_faces.pkl', 'wb') as f:
            pickle.dump(self.saved_faces, f)

    
    
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

    def compare_faces(self, face_img1, face_img2):
     # Ensure the images are the same size
     if face_img1.shape != face_img2.shape:
        print("Error: Images must have the same dimensions for comparison.")
        return None
    
     # Compute the mean squared error (MSE) between two images
     diff = cv.absdiff(face_img1, face_img2)  # Absolute difference between images
     diff_squared = np.square(diff)  # Square the differences to get MSE
     mse = np.mean(diff_squared)  # Mean of squared differences
    
     return mse
 
    def process_frame(self, img_np):
     if img_np is None or img_np.size == 0:
        print("Error: Empty image passed to process_frame.")
        return img_np

     try:
        if len(img_np.shape) == 2:
            img_np = cv.cvtColor(img_np, cv.COLOR_GRAY2BGR)

        img_h, img_w = img_np.shape[:2]

        face_blob = cv.dnn.blobFromImage(img_np, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
        body_blob = cv.dnn.blobFromImage(img_np, 1.0 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.face_net.setInput, face_blob)
            executor.submit(self.body_net.setInput, body_blob)
            face_detections = self.face_net.forward()
            body_detections = self.body_net.forward()

        min_confidence = 0.1

        # Find best face detection (highest confidence ≥ min_confidence)
        best_face = None
        best_face_conf = 0
        if face_detections is not None and face_detections.shape[2] > 0:
            for i in range(face_detections.shape[2]):
                conf = face_detections[0, 0, i, 2]
                if conf >= min_confidence and conf > best_face_conf:
                    x1 = int(face_detections[0, 0, i, 3] * img_w)
                    y1 = int(face_detections[0, 0, i, 4] * img_h)
                    x2 = int(face_detections[0, 0, i, 5] * img_w)
                    y2 = int(face_detections[0, 0, i, 6] * img_h)
                    best_face = (x1, y1, x2, y2)
                    best_face_conf = conf

        # Find best body detection (highest confidence ≥ min_confidence)
        best_body = None
        best_body_conf = 0
        if body_detections is not None and len(body_detections) > 0:
            for detection in body_detections:
                conf = detection[4]
                if conf >= min_confidence and conf > best_body_conf:
                    center_x = int(detection[0] * img_w)
                    center_y = int(detection[1] * img_h)
                    width = int(detection[2] * img_w)
                    height = int(detection[3] * img_h)
                    x1 = max(0, center_x - width // 2)
                    y1 = max(0, center_y - height // 2)
                    x2 = min(img_w, center_x + width // 2)
                    y2 = min(img_h, center_y + height // 2)
                    aspect_ratio = (y2 - y1) / (x2 - x1 + 1e-5)
                    if aspect_ratio >= 1.2:
                        best_body = (x1, y1, x2, y2)
                        best_body_conf = conf

        smoothing_factor = 0.6  # How fast to adapt (0=only old, 1=only new)

        def smooth_box(old_box, new_box, alpha):
            if old_box is None:
                return new_box
            ox1, oy1, ox2, oy2 = old_box
            nx1, ny1, nx2, ny2 = new_box
            sx1 = int(alpha * nx1 + (1 - alpha) * ox1)
            sy1 = int(alpha * ny1 + (1 - alpha) * oy1)
            sx2 = int(alpha * nx2 + (1 - alpha) * ox2)
            sy2 = int(alpha * ny2 + (1 - alpha) * oy2)
            # Ensure box does not grow unnecessarily: shrink only if new smaller, else keep old bigger size
            sx1 = max(sx1, min(ox1, nx1))
            sy1 = max(sy1, min(oy1, ny1))
            sx2 = min(sx2, max(ox2, nx2))
            sy2 = min(sy2, max(oy2, ny2))
            return (sx1, sy1, sx2, sy2)

        # Update tracked face box
        if best_face:
            self.tracked_face = smooth_box(self.tracked_face, best_face, smoothing_factor)
        else:
            # No detection this frame: keep previous tracked face unchanged
            if self.tracked_face is None:
                # No previous tracked face either; do nothing
                pass

        # Update tracked body box
        if best_body:
            self.tracked_body = smooth_box(self.tracked_body, best_body, smoothing_factor)
        else:
            if self.tracked_body is None:
                pass  # no prior tracked body

        # Draw tracked boxes
        if self.tracked_face:
            x1, y1, x2, y2 = self.tracked_face
            cv.rectangle(img_np, (x1, y1), (x2, y2), (200, 0, 0), 2)
            cv.putText(img_np, 'Tracked Face', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if self.tracked_body:
            x1, y1, x2, y2 = self.tracked_body
            cv.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(img_np, 'Tracked Body', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save faces if needed
        self.save_faces_to_file()

        return img_np

     except cv.error as e:
        print(f"OpenCV error in process_frame: {e}")
     except Exception as e:
        print(f"Unexpected error in process_frame: {e}")

     return img_np
 
    def smooth_detections(self, previous_detections, current_detections):
        if len(previous_detections) == 0:
            return current_detections

        # Smooth the current detections with previous ones
        smoothed_detections = []
        for curr in current_detections:
            closest_previous = min(previous_detections, key=lambda prev: np.linalg.norm(np.array(curr) - np.array(prev)))
            smoothed_detections.append(self.average_coordinates(curr, closest_previous))

        return smoothed_detections

    def average_coordinates(self, coord1, coord2):
        return tuple((np.array(coord1) + np.array(coord2)) // 2)
    
    def process_image(self, image_path):
     # Load the image from the provided path
     img_np = cv.imread(image_path)
     if img_np is None or img_np.size == 0:
        print("Error: Empty image passed to process_frame.")
        return img_np  # Return the original image if it's empty

     try:
        # Ensure the image has 3 channels (RGB)
        if len(img_np.shape) == 2:  # Grayscale image
            img_np = cv.cvtColor(img_np, cv.COLOR_GRAY2BGR)

        img_h, img_w = img_np.shape[:2]  # Image dimensions

        # Prepare input blobs for face and body detection using DNN models
        face_blob = cv.dnn.blobFromImage(img_np, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
        body_blob = cv.dnn.blobFromImage(img_np, 1.0 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        # Run face and body detection in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            face_future = executor.submit(self.face_net.setInput, face_blob)
            body_future = executor.submit(self.body_net.setInput, body_blob)

            # Get face and body detections
            face_detections = self.face_net.forward()
            body_detections = self.body_net.forward()

        # Check if face_detections is None or empty
        if face_detections is None or face_detections.shape[2] == 0:
            print("No faces detected.")
            face_detections = np.array([])  # Avoid further errors

        # Detect faces and keep the highest confidence face
        highest_confidence = 0
        best_face_coords = None
        if face_detections.size > 0:  # Proceed only if face detections are available
            for i in range(face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]
                if confidence > 0.15:  # Confidence threshold (e.g., 0.5)
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        x1 = int(face_detections[0, 0, i, 3] * img_w)
                        y1 = int(face_detections[0, 0, i, 4] * img_h)
                        x2 = int(face_detections[0, 0, i, 5] * img_w)
                        y2 = int(face_detections[0, 0, i, 6] * img_h)
                        best_face_coords = (x1, y1, x2, y2)

        # If face detected, process it
        if best_face_coords:
            x1, y1, x2, y2 = best_face_coords
            cv.rectangle(img_np, (x1, y1), (x2, y2), (200, 0, 0), 2)  # Red for face
            cv.putText(img_np, f'Face: {highest_confidence:.2f}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Check if body_detections is None or empty
        if body_detections is None or body_detections.shape[0] == 0:
            print("No bodies detected.")
            body_detections = np.array([])  # Avoid further errors

        # Detect and draw bodies with NMS
        detected_bodies = []
        if body_detections.size > 0:  # Proceed only if body detections are available
            for detection in body_detections:
                confidence = detection[4]
                if confidence >= 0.3:  # Confidence threshold for body detection
                    center_x, center_y, width, height = map(int, (detection[0] * img_w, detection[1] * img_h, detection[2] * img_w, detection[3] * img_h))
                    x1, y1, x2, y2 = max(0, center_x - width // 2), max(0, center_y - height // 2), min(img_w, center_x + width // 2), min(img_h, center_y + height // 2)
                    aspect_ratio = height / width
                    if aspect_ratio >= 1.2:
                        detected_bodies.append((x1, y1, x2, y2))

        nms_indices = cv.dnn.NMSBoxes([box for box in detected_bodies], [1] * len(detected_bodies), 0.50, 0.4)
        if len(nms_indices) > 0:
            for i in nms_indices.flatten():
                (x1, y1, x2, y2) = detected_bodies[i]
                cv.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue for bodies
                cv.putText(img_np, 'Body', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the list of faces to file for persistence (if needed)
        self.save_faces_to_file()

      
        # Show the processed image in a window
        cv.imshow("Processed Image", img_np)

        # Save the processed image as output/processed_media.png
        cv.imwrite("output/processed_media.png", img_np)

        # Wait for the user to press any key to close the window
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Return the processed image
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
                img = await asyncio.to_thread(processor.process_image, output_filename)
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
        if any(media_url.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            return 'image'
        elif media_url.lower().endswith('.gif'):
            return 'gif'
        elif media_url.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            return 'video'
        else:
            return 'image'
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
     if media_url not in ['exit', 'quit','stop','end']:
      processor = Processor(face_model=('hidden/deploy.prototxt', 'hidden/res10_300x300_ssd_iter_140000.caffemodel'), body_model=('hidden/yolov4.cfg', 'hidden/yolov4.weights'))
      result = await processor.process_media(media_url)
     else:
         break
     # print("Processed Media Result:", result)

# Execute main asynchronously
if __name__ == '__main__':
    asyncio.run(main())
