# Loads the best trained model.
# Randomly selects 25 images from seg_pred.
# Predicts each image and shows the predicted class above it.

import random
import torch
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from torchvision import transforms
from model import SimpleCNN


CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)

checkpoint = torch.load("models/best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image_paths = list(Path("seg_pred\seg_pred").glob("*.jpg"))
image_paths = random.sample(image_paths, 25)

plt.figure(figsize=(10, 10))

for i, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred = model(x).argmax(dim=1).item()

    plt.subplot(5, 5, i + 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(CLASS_NAMES[pred], fontsize=9)


Save_dir = Path(r"C:\Users\HP\Desktop\DL\PyTorch\CNN\archive\results")
plt.savefig(Save_dir / "predictions.png", dpi=200)

plt.tight_layout()
plt.show()
