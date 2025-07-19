import cv2
import torch
import time
import logging
from torchvision import transforms
from FERMolel import FERModel
from mail_util import send_email

# 日誌設定（加入0.py的格式）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 載入模型與設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FERModel().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
classes = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
NEGATIVE_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Sad']
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

negative_start_time = None
negative_duration = 0
non_negative_duration = 0
INTERRUPT_LIMIT = 1
NEGATIVE_DURATION = 10

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("無法讀取攝影機影像！")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 直接標準化整張影像
    face_resized = cv2.resize(gray, (48, 48))
    img = transform(face_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    label = classes[predicted.item()]
    cv2.putText(frame, f'Emotion: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 只在負面情緒時記錄 log
    current_time = time.time()
    frame_duration = 0.1

    if label in NEGATIVE_EMOTIONS:
        logging.info(f"偵測到負面情緒：{label}")
        if negative_start_time is None:
            logging.info(f"開始累積負面情緒：{label}")
            negative_start_time = current_time
            negative_duration = 0
        negative_duration += frame_duration
        non_negative_duration = 0

        # 累積超過10秒發送郵件
        if negative_start_time is not None and negative_duration >= NEGATIVE_DURATION and non_negative_duration <= INTERRUPT_LIMIT:
            logging.info(f"偵測到負面情緒累積 {negative_duration:.2f} 秒，發送郵件")
            send_email(
                subject="Emotion Alert",
                body=f"Detected negative emotion for {negative_duration:.2f} seconds!",
                client_email="huanghongjunh2@gmail.com"
            )
            negative_start_time = None
            negative_duration = 0
            non_negative_duration = 0
    else:
        if negative_start_time is not None:
            non_negative_duration += frame_duration
            if non_negative_duration > INTERRUPT_LIMIT:
                negative_start_time = None
                negative_duration = 0
                non_negative_duration = 0

    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
logging.info("程式正常結束。")