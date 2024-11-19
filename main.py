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


# Configure logging
logging.basicConfig(filename='pokemon_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PokemonPredictor:
    def __init__(self, dataset_folder="Data/pokemon/pokemon_images", 
                 dataset_file="dataset.npy", max_workers=4, num_keypoints=100):
        self.flann = cv.FlannBasedMatcher(
            dict(algorithm=6, table_number=9, key_size=9, multi_probe_level=1), 
            dict(checks=1, fast=True)
        )
        self.cache = {}  # Store descriptors
        self.dataset_file = dataset_file
        self.dataset_folder = dataset_folder
        self.orb = cv.ORB_create(nfeatures=num_keypoints)  # Initialize ORB detector
        # Initialize cascades for face and body detection
        self.head_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')  # or use another head detection model
        self.body_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')  # Load a body detection model
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')
        self.eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
        self.orb = cv.ORB_create()  # ORB feature detector
        self.cache = {}  # Store descriptors for similarity comparison
        self.face = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.face_recognizer =  cv.face.LBPHFaceRecognizer_create()  # Optional: For learning faces
        self.face_net = cv.dnn.readNetFromCaffe("hidden/deploy.prototxt", "hidden/res10_300x300_ssd_iter_140000.caffemodel")





   

      

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
     """Face detection with landmark refinement and tightly fitted bounding boxes for multiple faces."""
     try:
        # Convert image to blob for face detection
        blob = cv.dnn.blobFromImage(img_np, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # Iterate through all detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Compute the bounding box of the detected face
            x1 = int(detections[0, 0, i, 3] * img_np.shape[1])
            y1 = int(detections[0, 0, i, 4] * img_np.shape[0])
            x2 = int(detections[0, 0, i, 5] * img_np.shape[1])
            y2 = int(detections[0, 0, i, 6] * img_np.shape[0])

            # Ensure bounding box is within the image boundaries
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_np.shape[1], x2), min(img_np.shape[0], y2)

            # Draw the initial face bounding box
            cv.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(img_np, 'Face Detected', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Use landmarks to refine the bounding box
            face_roi = img_np[y1:y2, x1:x2]
            landmarks = self.landmark_detector.detectLandmarks(face_roi)

            if landmarks:
                # Calculate the refined bounding box based on landmarks
                min_x = min([pt[0] for pt in landmarks])
                max_x = max([pt[0] for pt in landmarks])
                min_y = min([pt[1] for pt in landmarks])
                max_y = max([pt[1] for pt in landmarks])

                # Apply a shrink factor to make the bounding box tighter around the face
                shrink_factor = 0.1
                x1_refined = max(0, int(x1 + min_x - (max_x - min_x) * shrink_factor))
                y1_refined = max(0, int(y1 + min_y - (max_y - min_y) * shrink_factor))
                x2_refined = min(img_np.shape[1], int(x1 + max_x + (max_x - min_x) * shrink_factor))
                y2_refined = min(img_np.shape[0], int(y1 + max_y + (max_y - min_y) * shrink_factor))

                # Draw the refined bounding box
                cv.rectangle(img_np, (x1_refined, y1_refined), (x2_refined, y2_refined), (0, 255, 0), 2)
                cv.putText(img_np, 'Refined Face', (x1_refined, y1_refined - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv.putText(img_np, 'No Landmarks Detected', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Return the highest match score, best match, and the processed image
        return highest_score, best_match, img_np

     except cv.error as e:
        print(f"OpenCV error in process_frame: {e}")
     except Exception as e:
        print(f"Unexpected error in process_frame: {e}")

     # Return the original values in case of an error
     return highest_score, best_match, img_np
 
 
 
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
