# Image Detection

`Image Detection` is a Python-based app that detects and highlights key objects in images and GIFs. It uses edge detection, feature matching, and async processing to handle different file types, drawing boxes around detected areas.

![Detected GIF](Testing/images/detected.gif)

### Features:
- **Async Loading**: Quickly loads and caches image data.
- **Edge Detection**: Uses ORB keypoints and FLANN-based matching to find features.
- **GIF Support**: Processes GIFs, adds boxes, and saves the results as new GIFs.
- **URL Support**: Lets you load images from URLs.
- **Progress Tracking**: Shows progress with `tqdm` while processing GIFs.

### Detected Face GIF

![Detected Body and Face GIF](Testing/images/processed_media.gif)

This GIF shows how the app highlights faces in GIFs. It detects faces in each frame using a DNN-based detector and draws a box around each face.

**How It Works:**
- **Face Detection**: Uses OpenCV's DNN-based face detector (Caffe model) to find faces.
- **Body Detection**: Uses YOLOv4 (You Only Look Once version 4) for detecting bodies (even some objects).

- **Box Refinement**: Adjusts boxes with facial landmarks for a better fit.
- **GIF Processing**: Processes each frame and saves the result as a new GIF.

The final result is a GIF with faces accurately highlighted frame by frame.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/cringe-neko-girl/detection.git
   cd Detection
