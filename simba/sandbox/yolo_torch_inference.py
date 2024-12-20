import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Paths
MDL = r"/mnt/c/troubleshooting/yolo_mdl/train/weights/best.pt"
IMG_PATH = r"/mnt/c/Users/sroni/OneDrive/Desktop/Screenshot 2024-11-15 123805.png"

# Load the model (YOLOv8 model)
checkpoint = torch.load(MDL, map_location=torch.device('cpu'))  # Load model weights
model = checkpoint['model'].eval()  # Set to eval mode

# Load and preprocess the image
image = Image.open(IMG_PATH)  # PIL image loading
if image.mode == 'RGBA':
    image = image.convert('RGB')

image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV


transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to 640x640
    transforms.ToTensor(),  # Convert to tensor
])

# Convert image to tensor and add batch dimension
image_tensor = transform(image).unsqueeze(0).half()

# Run inference on the image
with torch.no_grad():  # Disable gradient calculation
    predictions = model(image_tensor)  # Inference

detections = predictions[0].squeeze(0)


predictions = predictions[0][0]
predictions = predictions.permute(1, 0)
boxes = predictions[:, :4]  # First 4 columns: [x_center, y_center, width, height]
confidences = predictions[:, 4]  # 5th column: confidence score
top_conf = torch.argmax(confidences)
top_box = boxes[top_conf, :]

print(top_box)

xmin = top_box[0] - top_box[2] / 2
ymin = top_box[1] - top_box[3] / 2
xmax = top_box[0] + top_box[2] / 2
ymax = top_box[1] + top_box[3] / 2


print(xmin)

original_height, original_width = image_np.shape[:2]  # Get the original image dimensions

# Scale the bounding box back to original image size
scale_x = original_width / 640
scale_y = original_height / 640

xmin = int(xmin * scale_x)
ymin = int(ymin * scale_y)
xmax = int(xmax * scale_x)
ymax = int(ymax * scale_y)

print(xmin)
#
# # Draw the bounding box on the original image
# cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green box, thickness=2
#
# # Display the image with the bounding box
# cv2.imshow("Image with Bounding Box", image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

