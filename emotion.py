import os
import cv2
import torch
from torchvision import transforms
from FERMolel import FERModel
# 載入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FERModel().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

classes = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

faces_dir = "faces"
for face_file in os.listdir(faces_dir):
    face_path = os.path.join(faces_dir, face_file)
    face_img = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
    face_resized = cv2.resize(face_img, (48, 48))
    img = transform(face_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    label = classes[predicted.item()]
    print(f"{face_file}: {label}")