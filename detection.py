import torch
from torchvision import transforms
from PIL import Image
from model import get_model

# Define the number of classes (adjust based on your dataset)
num_classes = 9  # Example: 9 classes for tomato diseases

# Initialize the model architecture
model = get_model(num_classes)

# Load the model state dictionary with weights_only=True
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the disease names
disease_names = [
    "bacterial spot",
    "early blight",
    "healthy",
    "late blight",
    "leaf mold",
    "septorial leaf spot",
    "spotted spider mite",
    "target spot",
    "yellow leaf curl virus"
]

def predict(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_prob, top_class = torch.max(probabilities, 1)
    disease_name = disease_names[top_class.item()]
    probability = round(top_prob.item() * 100, 1)  # Format probability to one decimal place
    return disease_name, probability