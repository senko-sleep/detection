import os
import logging
import requests
from io import BytesIO
from PIL import Image
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import gc
import yaml

class PokemonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for img_file in os.listdir(root_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(root_dir, img_file)
                label_path = img_path.replace('images', 'labels').replace('.png', '.txt')
                self.images.append(img_path)
                self.labels.append(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            boxes = [list(map(float, line.split())) for line in f]

        return image, torch.tensor(boxes)

class PokemonDetector:
    def __init__(self):
        self.pokemon_templates = os.path.join(os.getcwd(), 'pokemon_templates')
        self.train_images_path = os.path.join(self.pokemon_templates, 'train', 'images')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = 'yolov5_pokemon_model.pt'

        # Load YOLOv5 model from ultralytics or custom
        self.model = self.load_or_train_model()

        # Data transformation (resizing for YOLOv5 input)
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    def load_or_train_model(self):
        """Load the YOLOv5 model or train it if no pretrained model exists."""
        if os.path.exists(self.model_path):
            logging.info(f"Loading model from {self.model_path}...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        else:
            logging.info("No saved model found. Training a new model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.train_model(model)
            model.save(self.model_path)
            logging.info(f"Model saved to {self.model_path}")
        return model.to(self.device)

    def train_model(self, model, epochs=5, batch_size=16):
        """Train the YOLOv5 model."""
        # Initialize dataset and dataloader
        dataset = PokemonDataset(self.train_images_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        logging.info("Starting training...")

        for epoch in range(epochs):
            running_loss = 0.0
            with tqdm(total=len(dataloader), desc=f"Epoch [{epoch + 1}/{epochs}]", unit="batch") as pbar:
                for images, boxes in dataloader:
                    images = images.to(self.device)
                    boxes = boxes.to(self.device)

                    optimizer = model.model.module.model[-1].optimizer

                    # Forward pass
                    outputs = model(images, targets=boxes)

                    # Compute loss and update weights
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                    pbar.update(1)

            logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
            gc.collect()

        logging.info("Training completed.")

    def predict(self, image_url):
        """Predict the bounding boxes and classes of Pokémon from an image URL."""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            logging.error(f"Failed to load image from URL: {e}")
            return None

        image = self.transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)

        return outputs

    def run(self):
        """Run the prediction loop."""
        while True:
            img_url = input("Enter the URL to a Pokémon image (or type 'quit' to exit): ")
            if img_url.lower() == 'quit':
                logging.info("Exiting the Pokémon Detector.")
                break

            result = self.predict(img_url)
            if result is not None:
                logging.info(f"Predicted bounding boxes: {result.xyxy}")
            else:
                logging.info("Prediction failed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    detector = PokemonDetector()
    detector.run()
