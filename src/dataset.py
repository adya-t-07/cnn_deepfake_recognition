import os
import cv2
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(post_process=False, keep_all=False, device=device)

class Data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_samples = []
        self.class_to_idx = {"real":1, "fake":0} #map each class of image to a label

        for class_name in ["real", "fake"]:
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_folder, img_name)
                    self.all_samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, index):
        img_path, label = self.all_samples[index]
        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #reverse color channel swapping

        #face cropping:
        try: 
            boxes, _ = detector.detect(image)
            if boxes is not None:
                box = boxes[0].astype(int) # [x1, y1, x2, y2]
                h, w, _ = image.shape
                #add 10% padding around the face box
                x1 = max(0, box[0] - int(0.1*(box[2]-box[0])))
                y1 = max(0, box[1] - int(0.1*(box[3]-box[1])))
                x2 = min(w, box[2] + int(0.1*(box[2]-box[0])))
                y2 = min(h, box[1] + int(0.1*(box[3]-box[1])))

                face = image[y1:y2, x1:x2]
                if face.size > 0:
                    image=face

        except Exception:
            pass
        
        image = Image.fromarray(image) #conver to PIL for torchvision transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
