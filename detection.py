import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import boto3
import io
import os

# Load environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')

# Define the number of classes (adjust based on your dataset)
num_classes = 9  # Example: 9 classes for tomato diseases

# Initialize the model architecture
model = get_model(num_classes)

# Download the model from S3 into memory
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
bucket_name = 'machinelearningmodels65675'
model_key = 'model.pth'

model_buffer = io.BytesIO()
s3.download_fileobj(bucket_name, model_key, model_buffer)
model_buffer.seek(0)

# Load the model state dictionary
model.load_state_dict(torch.load(model_buffer, map_location=torch.device('cpu')))
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