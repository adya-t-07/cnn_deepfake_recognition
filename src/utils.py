import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.model import CNN

# 1. Setup Device & Transforms
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Simple Grad-CAM Hook Implementation
class ForensicGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture forward activations and backward gradients
        self.forward_hook = target_layer.register_forward_hook(self.save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradients)
        
    def save_activations(self, model, input, output):
        self.activations = output.detach()
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor, class_idx=0):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass target
        loss = output[0] if output.shape[1] == 1 else output[0, class_idx]
        loss.backward()
        
        # Compute weight factors from pooled gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        
        # Apply ReLU to retain only features that positively contribute to the class
        cam = torch.clamp(cam, min=0)
        cam = cam / (torch.max(cam) + 1e-7)
        return cam.cpu().numpy()
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

# 3. Execution Wrapper
def run_forensic_analysis(img_path, title_prefix="Image"):
    model = CNN(pretrained=False)
    weights_path = os.path.join('weights', 'cnn_baseline.pth')
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    
    # 1. Load image with OpenCV to perform the face crop
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
        
    # Load OpenCV's built-in face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If a face is found, crop it. Otherwise, fallback to the full image.
    if len(faces) > 0:
        print(f"🎯 Face detected! Cropping image before Grad-CAM calculation...")
        x, y, w, h = faces[0]
        # Crop the face region
        cropped_img_bgr = img_bgr[y:y+h, x:x+w]
    else:
        print(f"⚠️ No face detected by OpenCV cascade. Using raw image fallback.")
        cropped_img_bgr = img_bgr

    # 2. Convert the cropped OpenCV image (BGR) to a PIL Image (RGB) for PyTorch
    cropped_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
    raw_img_pil = Image.fromarray(cropped_rgb)
    
    # Transform tensor for the network
    input_tensor = transform(raw_img_pil).unsqueeze(0).to(DEVICE)
    
    # 3. Compute Grad-CAM
    target_layer = model.network.layer4[-1]
    cam_tool = ForensicGradCAM(model, target_layer)
    heatmap = cam_tool.generate_heatmap(input_tensor)
    cam_tool.remove_hooks()
    
    # 4. Prepare visualization layout
    display_img = cv2.resize(cropped_img_bgr, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(display_img, 0.6, heatmap_colored, 0.4, 0)
    
    # Plot side by side
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{title_prefix} (Cropped Input)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"{title_prefix} CNN Attention Hotspot")
    plt.axis('off')
    plt.show()

# --- LOCAL EXECUTION ---
if __name__ == "__main__":
    # 🎯 PASTE YOUR COPIED PATH INSIDE THE QUOTES BELOW:
    # The 'r' before the string handles Windows backslashes perfectly!
    manual_image_path = r"C:\Users\tnkis\projects_adya\ViTvsCNN\data\external\train\fake\0EHGGMDEDA.jpg"
    
    print(f"🧠 Local execution starting: Analyzing target image...")
    if os.path.exists(manual_image_path):
        run_forensic_analysis(manual_image_path)
    else:
        print(f"❌ Error: The path you pasted does not exist:\n{manual_image_path}")
        print("Please double check your file name and file extension (.jpg vs .png)!")