import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageSequence
import imageio
import os
import tempfile
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Function to download file from URL
def download_file_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download file from URL: {url}")
        print(f"Exception details: {e}")
        return None

# Function to check if file is a GIF
def is_gif(file_bytes):
    try:
        img = Image.open(file_bytes)
        return img.format == 'GIF'
    except Exception as e:
        print(f"Error checking file format: {e}")
        return False

# Function to process a single frame with Detectron2
def process_frame(frame, predictor, cfg):
    # Convert PIL Image to OpenCV format (RGB to BGR)
    frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    
    # Run object detection
    outputs = predictor(frame_cv)
    
    # Get metadata for visualization
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    # Create visualizer
    v = Visualizer(frame_cv[:, :, ::-1],  # Convert BGR to RGB for visualizer
                  metadata=metadata,
                  scale=1.0,
                  instance_mode=ColorMode.IMAGE)
    
    # Draw instance predictions
    vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Get the image with predictions (already in RGB format)
    visualized_frame = vis_output.get_image()
    
    # Convert back to PIL Image
    return Image.fromarray(visualized_frame)

# Function to initialize the model for object detection
def init_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for object detection
    
    # Set device explicitly to CPU
    cfg.MODEL.DEVICE = "cpu"
    
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

# Function to process GIF file
def process_gif(gif_data, output_path="detected_output.gif"):
    # Initialize model
    print("Initializing model (CPU mode)...")
    predictor, cfg = init_model()
    
    # Open GIF file
    gif = Image.open(gif_data)
    
    # Get GIF metadata
    duration = gif.info.get('duration', 100)  # Default 100ms if not specified
    loop = gif.info.get('loop', 0)  # Default to loop forever
    
    # List to store processed frames
    processed_frames = []
    
    # Process each frame
    frame_count = 0
    total_frames = sum(1 for _ in ImageSequence.Iterator(gif))
    
    print(f"Processing GIF with {total_frames} frames...")
    
    for frame in ImageSequence.Iterator(gif):
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")
        
        # Convert frame to RGB (needed for some GIFs)
        rgb_frame = frame.convert('RGB')
        
        # Process the frame with object detection
        processed_frame = process_frame(rgb_frame, predictor, cfg)
        
        # Append to list
        processed_frames.append(processed_frame)
    
    # Save as GIF
    print(f"Saving processed GIF to {output_path}")
    processed_frames[0].save(
        output_path,
        save_all=True,
        append_images=processed_frames[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )
    
    print(f"Processing complete! Output saved to {output_path}")
    return output_path

# Function to process a single image
def process_single_image(image_data, output_path="detected_output.jpg"):
    # Initialize model
    print("Initializing model (CPU mode)...")
    predictor, cfg = init_model()
    
    # Open and convert image to OpenCV format
    img = Image.open(image_data).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Process the image
    print("Detecting objects and drawing bounding boxes...")
    
    # Run object detection
    outputs = predictor(img_cv)
    
    # Get metadata for visualization
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    # Create visualizer
    v = Visualizer(img_cv[:, :, ::-1],
                  metadata=metadata,
                  scale=1.0,
                  instance_mode=ColorMode.IMAGE)
    
    # Draw instance predictions
    vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Get the image with predictions
    visualized_image = vis_output.get_image()
    
    # Convert back to OpenCV format if needed
    result_img = Image.fromarray(visualized_image)
    
    # Save result
    result_img.save(output_path)
    print(f"Processing complete! Output saved to {output_path}")
    
    # Print detection info
    print_detection_info(outputs, metadata)
    
    return output_path

# Print detection information
def print_detection_info(outputs, metadata):
    instances = outputs["instances"]
    pred_classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()
    
    print("\nDetection Results:")
    print("-----------------")
    print(f"Total objects detected: {len(instances)}")
    print("\nDetailed detections:")
    
    for i, (class_idx, score) in enumerate(zip(pred_classes, scores)):
        class_name = metadata.thing_classes[class_idx]
        print(f"Object {i+1}: {class_name} (confidence: {score:.2f})")

# Main function to execute the entire process
def process_file_from_url(url):
    print(f"Downloading file from: {url}")
    file_data = download_file_from_url(url)
    
    if file_data is None:
        print("ERROR: File could not be downloaded.")
        return None
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "detectron_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file is a GIF
    file_data.seek(0)  # Reset file pointer
    if is_gif(file_data):
        print("Detected GIF file. Processing frames...")
        file_data.seek(0)  # Reset file pointer again
        output_path = os.path.join(output_dir, "detected_output.gif")
        return process_gif(file_data, output_path)
    else:
        print("Processing as a single image...")
        file_data.seek(0)  # Reset file pointer
        output_path = os.path.join(output_dir, "detected_output.jpg")
        return process_single_image(file_data, output_path)

# Example usage
if __name__ == "__main__":
    url = input("Enter the URL of the GIF or image: ")
    if not url:
        print("No URL provided. Exiting...")
    else:
        output_file = process_file_from_url(url)
        if output_file:
            print(f"File processed successfully and saved to: {output_file}")
