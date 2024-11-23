import os
import cv2 as cv
import numpy as np
import time
import aiohttp
import asyncio
import io
from sklearn.metrics.pairwise import cosine_similarity
import logging
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageSequence
from tqdm import tqdm  # Import tqdm for progress tracking
import mediapipe as mp
import cmake
import dlib

# Configure logging
logging.basicConfig(filename='pokemon_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PokemonPredictor:
    def __init__(self, dataset_folder="Data/pokemon/pokemon_images", 
                 dataset_file="dataset.npy", max_workers=4, num_keypoints=100):
        # Initialize key parameters
        self.dataset_file = dataset_file
        self.dataset_folder = dataset_folder
        self.max_workers = max_workers
        self.num_keypoints = num_keypoints
        
        # Initialize FLANN-based matcher for feature comparison
        self.flann = cv.FlannBasedMatcher(
            dict(algorithm=6, table_number=9, key_size=9, multi_probe_level=1), 
            dict(checks=1, fast=True)
        )

        # Cache for storing descriptors
        self.cache = {}  

        # Initialize ORB feature detector
        self.orb = cv.ORB_create(nfeatures=num_keypoints)  

        # Initialize face recognizer (for optional face learning)
        self.face_recognizer = cv.face.LBPHFaceRecognizer_create()  

        # Load pre-trained models for face and body detection
        self.face_net = cv.dnn.readNetFromCaffe("hidden/deploy.prototxt", "hidden/res10_300x300_ssd_iter_140000.caffemodel")
        self.landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.body_net = cv.dnn.readNetFromDarknet("hidden/yolov4.cfg", "hidden/yolov4.weights")
        self.body_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.body_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        # Load Haar Cascade Classifiers for face and body detection
        self.head_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Face detection
        self.body_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')  # Full-body detection
        self.eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')  # Eye detection

        # Initialize face detection (Haar Cascade)
        self.face = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Previous bounding box history for face tracking
        self.previous_faces = []  # Store previous face detections (for smoothing)
        self.previous_bodies = []  # Store previous body detections (for smoothing)
        self.face_memory = []  # Memory for past N frames of face detections
        self.body_memory = []  # Memory for past N frames of body detections
        self.smoothing_window = 5  # Number of past frames to consider for smoothing





   

      

    async def load_dataset(self):
        """Load the dataset from npy file or process the image folder asynchronously."""
        if os.path.exists(self.dataset_file):
            self.load_from_npy(self.dataset_file)
        else:
            await self.load_from_images()

    def load_from_npy(self, dataset_file):
        """Load precomputed descriptors from npy."""
        data = np.load(dataset_file, allow_pickle=True).item()
        self.cache = data
        logging.info(f"Loaded dataset from {dataset_file}. Total images: {len(data)}")

    async def load_from_images(self):
        """Process images in the folder using asyncio."""
        tasks = [
            self.process_image(os.path.join(self.dataset_folder, filename), filename)
            for filename in os.listdir(self.dataset_folder)
            if os.path.isfile(os.path.join(self.dataset_folder, filename))
        ]
        await asyncio.gather(*tasks)  # Gather all tasks and run them concurrently
        await self.save_dataset_concurrently()

    async def save_dataset_concurrently(self):
        """Save the dataset to npy file asynchronously."""
        if self.cache:  # Only save if there are descriptors
            try:
                await asyncio.to_thread(self.save_to_npy)
            except Exception as e:
                logging.error(f"Error saving dataset: {e}")
        else:
            logging.info("No descriptors to save.")

    def save_to_npy(self):
        """Actual function to save the cache to a npy file."""
        np.save(self.dataset_file, self.cache)
        logging.info(f"Saved dataset to {self.dataset_file}.")

    async def process_image(self, path, filename):
        """Process a single image asynchronously."""
        img = await asyncio.get_event_loop().run_in_executor(self.executor, cv.imread, path)
        if img is not None:
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray_img, None)
            if descriptors is not None:
                self.cache[filename] = descriptors.astype(np.uint8)
                await self.cache_flipped_image(img, filename)

    async def cache_flipped_image(self, img, filename):
        """Cache the flipped version of the image and its descriptors."""
        flipped_img = cv.flip(img, 1)
        gray_flipped_img = cv.cvtColor(flipped_img, cv.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_flipped_img, None)
        if descriptors is not None:
            flipped_filename = filename.replace(".png", "_flipped.png")
            self.cache[flipped_filename] = descriptors.astype(np.uint8)

    async def load_image_from_url(self, url):
        """Asynchronously fetch and decode an image from a URL."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        img_data = await response.read()
                        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
                        return cv.imdecode(img_array, cv.IMREAD_COLOR)
                    else:
                        logging.error(f"Failed to fetch image, status code: {response.status}")
            except aiohttp.ClientError as e:
                logging.error(f"Error fetching image: {e}")
        return None

    async def load_gif_frames(self, url):
     """Load frames from a GIF URL and capture their durations."""
     async with aiohttp.ClientSession() as session:
         try:
            async with session.get(url) as response:
                if response.status == 200:
                    img_data = await response.read()
                    img_gif = Image.open(io.BytesIO(img_data))
                    frames = [np.array(frame.convert('RGB')) for frame in ImageSequence.Iterator(img_gif)]
                    
                    # Extract durations (in milliseconds) for each frame
                    durations = [frame.info['duration'] for frame in ImageSequence.Iterator(img_gif)]
                    return frames, durations
                else:
                    logging.error(f"Failed to fetch GIF, status code: {response.status}")
         except aiohttp.ClientError as e:
            logging.error(f"Error fetching GIF: {e}")
     return None, None

    async def cross_match(self, desB):
        """Match the descriptors with the dataset using FLANN."""
        if desB is None or desB.size == 0:
            logging.warning("No descriptors available for matching.")
            return None, 0.0

        best_match = None
        best_score = float('-inf')

        for filename, descriptor in self.cache.items():
            matches = self.flann.knnMatch(desB, descriptor, k=2)

            # Filter out good matches
            good_matches = []
            for match_pair in matches:
                # Ensure we have at least 2 matches in the pair
                if len(match_pair) >= 2:
                    m, n = match_pair  # Unpack the matches
                    # Compare distances and filter matches
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            score = len(good_matches)

            if score > best_score:
                best_score = score
                best_match = filename

        return best_match, best_score

    async def predict_pokemon(self, img):
     """Predict the closest matching Pokémon from the dataset using contour detection and feature extraction."""
     start_time = time.time()
     best_match = None
     highest_score = float('-inf')  # Initialize with very low similarity score
     frames_with_detections = []  # List to store frames with bounding boxes
     durations = []  # List to store frame durations (for GIFs)

     # Process as a list of frames (animated GIF) or as a static image
     if isinstance(img, list):
        preloaded_frames = img.copy()
        frame_count = len(preloaded_frames)

        # Use tqdm to track progress
        for frame_idx, img_np in tqdm(enumerate(preloaded_frames), total=frame_count, desc="Processing frames"):
            img_np = cv.cvtColor(img_np, cv.IMREAD_COLOR)  # Convert to BGR for OpenCV
            highest_score, best_match, processed_frame = await self.process_frame(img_np, frame_idx, highest_score, best_match)
            frames_with_detections.append(processed_frame)  # Store the processed frame with detections

        # Return both frames and durations if it's a GIF
        return best_match, time.time() - start_time, frames_with_detections, durations

     else:
        img_np = np.array(img)
        img_np = img_np.astype(np.uint8) if img_np.dtype != np.uint8 else img_np
        highest_score, best_match, processed_frame = await self.process_frame(img_np, 0, highest_score, best_match)
        frames_with_detections.append(processed_frame)  # Store the processed frame with detections

     logging.info(f"Processed image in {time.time() - start_time:.2f} seconds. Best match: {best_match} with score {highest_score}.")
     return best_match, time.time() - start_time, frames_with_detections, None  # No durations for static images
    
    
    async def process_frame(self, img_np, frame_idx, highest_score, best_match):
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
                cv.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv.putText(img_np, 'Face', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if len(nms_indices) > 0:
                for i in nms_indices.flatten():
                    (x1, y1, x2, y2) = smoothed_bodies[i]
                    cv.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for bodies
                    cv.putText(img_np, 'Body', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Store the current detections for smoothing in the next frame
            self.previous_faces = smoothed_faces
            self.previous_bodies = smoothed_bodies

            # Step 9: Return the highest match and processed frame
            return highest_score, best_match, img_np

        except cv.error as e:
            print(f"OpenCV error in process_frame: {e}")
        except Exception as e:
            print(f"Unexpected error in process_frame: {e}")

        # Return original values in case of error
        return highest_score, best_match, img_np

    def smooth_detections(self, previous_detections, current_detections, smoothing_factor=0.5):
        """
        Smooth bounding boxes based on previous detections and current detections
        using a simple average method for each (x1, y1, x2, y2) bounding box.
        """
        smoothed_detections = []

        if len(previous_detections) != len(current_detections):
            # Handle the case where the number of previous and current detections doesn't match
            # In such cases, just return the current detections (no smoothing)
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
     """Save the processed frames as a new GIF with bounding boxes, maintaining the original speed."""
     save_path = 'detected_face.gif'
     frames_to_save = [Image.fromarray(frame) for frame in frames]
    
     # Save the frames with the original frame durations
     frames_to_save[0].save(save_path, save_all=True, append_images=frames_to_save[1:], loop=0, duration=durations)
     logging.info(f"Saved processed GIF with detections to {save_path}.")
     
async def main():
    predictor = PokemonPredictor()
    await predictor.load_dataset()

    while True:
        img_url = input("Enter an image URL (or type 'quit' to exit): ")
        if img_url.lower() == 'quit':
            break

        # Load GIF or static image
        if img_url.lower().endswith('.gif'):
            frames, durations = await predictor.load_gif_frames(img_url)
            if frames:
                best_match, _, processed_frames, _ = await predictor.predict_pokemon(frames)
                print(f"Best match for GIF: {best_match}")
                await predictor.save_gif_with_detections(processed_frames, durations)

        else:
            img = await predictor.load_image_from_url(img_url)
            if img is not None:
                best_match, _, processed_frames, _ = await predictor.predict_pokemon(img)
                print(f"Best match for image: {best_match}")

asyncio.run(main())
