# ImagePredictor

ImagePredictor is a Python-based application for identifying the closest matching image from a dataset. It uses advanced edge detection, feature matching, and asynchronous processing to handle both static images and animated GIFs, identifying objects based on a precomputed dataset of features.

![Detected Pokémon GIF](Testing/detected_pokemon.gif)


- **Async Loading**: Efficiently loads and caches image data with asynchronous processing.
- **Edge Detection and Keypoints**: Uses ORB keypoints and FLANN-based matching for accurate feature matching.
- **GIF Support**: Processes GIF frames with bounding boxes and saves results as a new animated GIF.
- **URL Support**: Allows for loading images directly from URLs.
- **Progress Tracking**: Uses `tqdm` to display processing progress for GIF frames.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/cringe-neko-girl/detection.git
   cd ImagePredictor
