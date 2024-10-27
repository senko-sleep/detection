import os
import logging
import requests
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Configuration class
class Config:
    ORG_DIR = 'pokemon_templates/train/images'
    BATCH_SIZE = 16
    EPOCHS = 5  # Increased for better training
    INITIAL_LEARNING_RATE = 0.01
    MODEL_SAVE_PATH = 'pokemon_detector.pth'
    FEATURE_VECTOR_SIZE = 2048  # ResNet50 final feature layer

# Dataset class
class PokemonDataset:
    def __init__(self, org_dir, transform=None):
        self.org_dir = org_dir
        self.transform = transform
        self.images = self.load_images()
        self.model = self.load_resnet()  # Ensure model is loaded here
        self.feature_vectors = self.load_feature_vectors()

    def load_images(self):
        """Load image filenames from the given directory."""
        return [img for img in os.listdir(self.org_dir) if img.endswith(".png")]

    def load_resnet(self):
        """Load pretrained ResNet50 for feature extraction."""
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Identity()  # Use penultimate layer for feature vectors
        model.eval()
        return model

    def extract_features(self, image):
        """Extract feature vector from an image using ResNet50."""
        with torch.no_grad():
            features = self.model(image)
        return features

    def load_feature_vectors(self):
        """Precompute feature vectors for all Pokémon templates, including flipped versions."""
        features = {}
        for img_name in tqdm(self.images, desc="Extracting features"):
            img_path = os.path.join(self.org_dir, img_name)

            # Load original image and convert to tensor
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image_tensor = self.transform(image).unsqueeze(0)

            # Store feature vector for the original image
            features[img_name] = self.extract_features(image_tensor).cpu().numpy()

            # Store feature vector for the horizontally flipped image
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_tensor = self.transform(flipped_image).unsqueeze(0)
            features[f"{img_name}_flipped"] = self.extract_features(flipped_tensor).cpu().numpy()

        return features

# Main Detector class
class PokemonDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = PokemonDataset(Config.ORG_DIR, transform=self.initialize_transform())
        self.model = self.dataset.load_resnet().to(self.device)

    def initialize_transform(self):
        """Define the transform pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_url):
        """Predict the closest matching Pokémon template from an input image URL."""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.initialize_transform()(image).unsqueeze(0).to(self.device)

            # Extract feature vector from the input image
            input_features = self.dataset.extract_features(image).cpu().numpy()

            # Find the closest match by comparing with template feature vectors
            best_match, best_score = self.find_best_match(input_features)
            if best_score > 0.7:  # Confidence threshold
                return best_match, best_score
            else:
                logging.warning("No confident match found.")
                return None

        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            return None

    def find_best_match(self, input_features):
        """Find the Pokémon template with the highest cosine similarity, considering flipped images."""
        best_match = None
        best_score = -1

        for img_name, feature_vector in self.dataset.feature_vectors.items():
            similarity = cosine_similarity(input_features, feature_vector)[0][0]
            if similarity > best_score:
                best_match = img_name.split('.')[0].replace('_flipped', '')  # Remove flipped marker if present
                best_score = similarity

        return best_match, best_score

# Main Execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = PokemonDetector()

    while True:
        image_url = input("Enter the image URL for prediction (or type 'exit' to quit): ").strip()
        if image_url.lower() == 'exit':
            break

        result = detector.predict(image_url)
        if result:
            prediction, confidence = result
            logging.info(f"Matched Pokémon: {prediction} with confidence: {confidence:.2f}")
        else:
            logging.error("Prediction failed.")
