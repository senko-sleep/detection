import os
import logging
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import torch.nn.functional as F

# Configuration class for easy adjustments
class Config:
    ORG_DIR = 'pokemon_templates/train/images'
    BATCH_SIZE = 16
    EPOCHS = 3
    INITIAL_LEARNING_RATE = 0.001
    LEARNING_RATE_MIN = 1e-6  # Minimum learning rate
    MODEL_SAVE_PATH = 'pokemon_detector.pth'

# Dataset class
class PokemonDataset:
    def __init__(self, org_dir, transform=None):
        self.org_dir = org_dir
        self.transform = transform
        self.images = self.load_images()
        self.classes = self.load_classes()

    def load_images(self):
        images = []
        for img in os.listdir(self.org_dir):
            if img.endswith(".png"):
                images.append(img)
        return images

    def load_classes(self):
        return [img.split('.')[0] for img in self.images]  # Extract class names from filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.org_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.classes[idx]

# Pokemon Detector class
class PokemonDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = PokemonDataset(Config.ORG_DIR, transform=self.initialize_transform())
        self.classes = self.dataset.classes  # Fixed: Set self.classes after dataset initialization
        self.model = self.load_model()

    def initialize_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 for the model
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # More augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        model = models.resnet50(weights='IMAGENET1K_V1')  # Updated way to load weights
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))  # Adjust output layer
        model.to(self.device)
        model.train()  # Set the model to training mode
        return model

    def train(self):
        # DataLoader
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.INITIAL_LEARNING_RATE)  # Using AdamW

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, min_lr=Config.LEARNING_RATE_MIN)

        for epoch in range(Config.EPOCHS):
            running_loss = 0.0
            self.model.train()  # Ensure model is in training mode

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}"):
                images, labels = images.to(self.device), torch.tensor([self.classes.index(label) for label in labels]).to(self.device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Update the learning rate based on the loss
            scheduler.step(running_loss)
            logging.info(f"Epoch [{epoch + 1}/{Config.EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

        # Save the trained model
        torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
        logging.info(f"Model saved to {Config.MODEL_SAVE_PATH}")

    def predict(self, image_url):
        """Predict the Pokémon class from the image URL."""
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an error for bad responses
            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Preprocess the image before prediction
            image = self.initialize_transform()(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(image)
                probabilities = F.softmax(output, dim=1)  # Get probabilities
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()

                # Use a confidence threshold (e.g., 0.5) to determine if prediction is valid
                if confidence > 0.5:  # Adjust threshold as needed
                    return self.classes[predicted_idx], confidence
                else:
                    logging.warning("Confidence too low for prediction.")
                    return None

        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            return None

# Main Execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create a PokemonDetector instance
    detector = PokemonDetector()

    # Train the model and save the weights
    detector.train()

    # Prediction loop
    while True:
        image_url = input("Enter the image URL for prediction (or type 'exit' to quit): ").strip()
        if image_url.lower() == 'exit':
            break

        result = detector.predict(image_url)
        if result is not None:
            prediction, confidence = result
            logging.info(f"Predicted Pokémon: {prediction} with confidence: {confidence:.2f}")
        else:
            logging.error("Prediction failed.")
