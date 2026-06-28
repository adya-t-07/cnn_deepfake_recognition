import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
from src.model import DeepfakeViTWrapper # Ensure this matches your wrapper's file location

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import types

class ViTAttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.original_forwards = {}
        
        # Patch each encoder block's forward pass dynamically
        for i, block in enumerate(self.model.network.encoder.layers):
            # Save a backup of the true forward method
            self.original_forwards[i] = block.forward
            
            # Create an adjusted forward pass that binds to the instance
            # and extracts attention weights explicitly
            block.forward = types.MethodType(self._create_patched_forward(i, block), block)
            
    def _create_patched_forward(self, layer_idx, block_instance):
        def patched_forward(self, input_tensor):
            # 1. Mirror the normal LayerNorm preprocessing steps
            x = self.ln_1(input_tensor)
            
            # 2. FORCE need_weights=True to intercept the matrix
            x_attn, weights = self.self_attention(x, x, x, need_weights=True)
            
            # Capture the weights matrix securely
            self_visualizer = visualizer # Access from external scoping
            self_visualizer.attentions.append(weights.detach())
            
            # 3. Complete the standard residual stream
            x = self.dropout(x_attn)
            x = x + input_tensor
            
            y = self.ln_2(x)
            y = self.mlp(y)
            return x + y
            
        return patched_forward
        
    def remove_hooks(self):
        # Restore the original unpatched methods clean and tidy
        for i, block in enumerate(self.model.network.encoder.layers):
            block.forward = self.original_forwards[i]

    def compute_rollout(self, input_tensor):
        self.attentions = []
        
        # Crucial configuration to override tracking scoping globally
        global visualizer
        visualizer = self
        
        # Execute the patched network forward path
        _ = self.model(input_tensor)
            
        if not self.attentions:
            raise RuntimeError("No attention matrices were captured! Patching sequence failed.")
            
        # Build the attention rollout matrix across all layer blocks
        result = torch.eye(self.attentions[0].size(-1)).to(DEVICE)
        with torch.no_grad():
            for attention in self.attentions:
                attn_matrix = attention.squeeze(0)
                
                # Handle matrix shapes safely across varying PyTorch backend layouts
                if len(attn_matrix.shape) == 3:
                    attn_matrix = torch.mean(attn_matrix, dim=0)
                
                I = torch.eye(attn_matrix.size(-1)).to(DEVICE)
                a = (attn_matrix + I) / 2
                a = a / a.sum(dim=-1, keepdim=True)
                result = torch.matmul(a, result)
        
        # Extract attention mapping for the [CLS] token (14x14 patch layout)
        mask = result[0, 1:].reshape(14, 14)
        x, y = torch.meshgrid(torch.linspace(-1, 1, 14), torch.linspace(-1, 1, 14), indexing='ij')
        dst = torch.sqrt(x**2 + y**2).to(DEVICE)
        sigma = 0.75
        gauss = torch.exp(-(dst**2 / (2.0 * sigma**2))).cpu().numpy()
        
        # Apply the mask to dampen the outer border cells smoothly
        mask = mask.cpu().numpy() * gauss
        
        # Re-normalize the heat distribution map
        mask = mask / (mask.max() + 1e-7)
        return mask

def visualize_vit_attention(img_path):
    model = DeepfakeViTWrapper(pretrained=False)
    weights_path = os.path.join('weights', 'vit_baseline.pth')
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing ViT weights at: {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    
    visualizer = ViTAttentionVisualizer(model)
    
    # Load and crop face (matching your CNN optimization steps)
    img_bgr = cv2.imread(img_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_img_bgr = img_bgr[y:y+h, x:x+w]
    else:
        cropped_img_bgr = img_bgr

    cropped_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
    raw_img_pil = Image.fromarray(cropped_rgb)
    input_tensor = transform(raw_img_pil).unsqueeze(0).to(DEVICE)
    
    attn_mask = visualizer.compute_rollout(input_tensor)
    visualizer.remove_hooks()
    
    # Process overlay maps
    display_img = cv2.resize(cropped_img_bgr, (224, 224))
    attn_resized = cv2.resize(attn_mask, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(display_img, 0.6, heatmap, 0.4, 0)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Input")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("ViT Self-Attention Map")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Point this to your test image path
    target_image = r"C:\Users\tnkis\projects_adya\ViTvsCNN\data\external\valid\fake\0AIFZB4IE6.jpg"
    visualize_vit_attention(target_image)