# ImagePredictor

ImagePredictor is a Python-based application for identifying the closest matching image from a dataset. It uses advanced edge detection, feature matching, and asynchronous processing to handle both static images and animated GIFs, identifying objects based on a precomputed dataset of features.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Setting Up the Dataset](#setting-up-the-dataset)
   - [Running the Predictor](#running-the-predictor)
   - [Processing an Image from a URL](#processing-an-image-from-a-url)
4. [Code Overview](#code-overview)
5. [Logging](#logging)
6. [Contributing](#contributing)
7. [License](#license)

## Features

- **Async Loading**: Efficiently loads and caches image data with asynchronous processing.
- **Edge Detection and Keypoints**: Uses ORB keypoints and FLANN-based matching for accurate feature matching.
- **GIF Support**: Processes GIF frames with bounding boxes and saves results as a new animated GIF.
- **URL Support**: Allows for loading images directly from URLs.
- **Progress Tracking**: Uses `tqdm` to display processing progress for GIF frames.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/cringe-neko-girl/detection.git
   cd ImagePredictor```
