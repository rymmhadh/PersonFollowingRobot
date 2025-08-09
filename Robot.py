import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

import time
import cv2
import torch
import numpy as np
from PIL import Image

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchreid.utils.feature_extractor import FeatureExtractor
from my_strongsort import StrongSORT
from RobotDB import save_profile_to_db

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_yolo = YOLO("yolov8n.pt").to(device)

# Create a dummy image (HxWxC) with random pixels, dtype uint8
dummy_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

# Predict using the model on dummy image, device as string
results = model_yolo.predict(source=dummy_img, device=device, verbose=False)
print("Inference done")

model_yolo = YOLO("yolov8x.pt").to(device)
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=20, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
extractor_front = FeatureExtractor(model_name='osnet_x1_0', device=device)
extractor_rear = FeatureExtractor(model_name='osnet_x1_0', device=device)
tracker = StrongSORT(model_weights='osnet_x0_25_market1501.pt', device=device, fp16=False)

selected_id = None
boxes = []
recording = False
record_start_time = None
record_duration = 20

embeddings_face = []
embeddings_body_front = []
embeddings_body_rear = []

target_embedding = None
target_name = None
target_track_id = None

def cosine_similarity(a, b):
    a = a.view(-1).cpu().numpy()
    b = b.view(-1).cpu().numpy()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_closest_bbox(click, bboxes):
    x_click, y_click = click
    min_dist = float('inf')
    best_idx = -1
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        dist = (x_click - x_center) ** 2 + (y_click - y_center) ** 2
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

def mouse_callback(event, x, y, flags, param):
    global selected_id, boxes, recording, record_start_time
    if event == cv2.EVENT_LBUTTONDOWN and boxes:
        selected_id = get_closest_bbox((x, y), boxes)
        recording = True
        record_start_time = time.time()
        print(f"[INFO] Selected bbox {selected_id} for recording")

def tensor_to_list(tensor):
    if tensor is None:
        return None
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().tolist()
    elif isinstance(tensor, (list, np.ndarray)):
        return tensor
    else:
        raise TypeError(f"Unsupported type for tensor_to_list: {type(tensor)}")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model_yolo.predict(source=frame, classes=[0], device=0 if device == 'cuda' else 'cpu', verbose=False)[0]
    boxes = [list(map(int, box.xyxy[0].tolist())) for box in results.boxes]

    detections = []
    for (x1, y1, x2, y2), box in zip(boxes, results.boxes):
        conf = float(box.conf[0])
        detections.append([x1, y1, x2, y2, conf, 0])
    detections = np.array(detections)

    tracks = tracker.update(detections, frame)

    best_score = -1
    current_target_box = None
    target_track_id = None

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        x, y, w, h = track.to_tlwh()
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        x1, x2 = max(0, x1), min(x2, frame.shape[1] - 1)
        y1, y2 = max(0, y1), min(y2, frame.shape[0] - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if target_embedding is not None:
            emb = extractor_front(crop_rgb)
            emb = torch.tensor(emb).to(device).float()
            emb = emb / torch.norm(emb)
            sim = cosine_similarity(target_embedding, emb)

            if sim > best_score:
                best_score = sim
                current_target_box = (x1, y1, x2, y2)
                target_track_id = track_id

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        x, y, w, h = track.to_tlwh()
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        if track_id == target_track_id and best_score > 0.75:
            color = (0, 255, 0)
            label = f"{target_name} ({best_score:.2f})" if target_name else f"Target ({best_score:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif target_embedding is None:
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if recording:
        elapsed = time.time() - record_start_time if record_start_time is not None else 0
        cv2.putText(frame, f"Recording: {elapsed:.1f}/{record_duration}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif target_embedding is None:
        cv2.putText(frame, "Click person to start 20s auto-recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    else:
        cv2.putText(frame, "Tracking target based on embedding similarity", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    if recording and selected_id is not None and selected_id < len(boxes):
        x1, y1, x2, y2 = boxes[selected_id]
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_rgb_pil = Image.fromarray(crop_rgb)

        faces = mtcnn(crop_rgb_pil)
        if faces is not None and len(faces) > 0:
            face_tensor = faces[0].to(device)  # Prendre premier visage détecté
            face_tensor = face_tensor.unsqueeze(0)  # Ajouter dimension batch
            face_emb = resnet(face_tensor).squeeze(0)
            face_emb = face_emb / torch.norm(face_emb)
            embeddings_face.append(face_emb.detach().cpu())

        body_emb_front = extractor_front(crop_rgb)
        if len(body_emb_front.shape) == 2 and body_emb_front.shape[0] == 1:
            body_emb_front = body_emb_front.squeeze(0)
        body_emb_front = body_emb_front / torch.norm(body_emb_front)
        embeddings_body_front.append(body_emb_front.detach().cpu())

        body_emb_rear = extractor_rear(crop_rgb)
        if len(body_emb_rear.shape) == 2 and body_emb_rear.shape[0] == 1:
            body_emb_rear = body_emb_rear.squeeze(0)
        body_emb_rear = body_emb_rear / torch.norm(body_emb_rear)
        embeddings_body_rear.append(body_emb_rear.detach().cpu())

    if recording and record_start_time is not None and time.time() - record_start_time >= record_duration:
        avg_face_emb = torch.mean(torch.stack(embeddings_face), dim=0) if embeddings_face else None
        avg_body_front_emb = torch.mean(torch.stack(embeddings_body_front), dim=0) if embeddings_body_front else None
        avg_body_rear_emb = torch.mean(torch.stack(embeddings_body_rear), dim=0) if embeddings_body_rear else None

        avg_face_emb = tensor_to_list(avg_face_emb)
        avg_body_front_emb = tensor_to_list(avg_body_front_emb)
        avg_body_rear_emb = tensor_to_list(avg_body_rear_emb)

        name = input("Enter profile name to save: ")
        save_profile_to_db(name, avg_face_emb, avg_body_front_emb, avg_body_rear_emb)
        print(f"[SUCCESS] Profile '{name}' saved with averaged embeddings.")

        if avg_body_front_emb is not None:
            target_embedding = torch.tensor(avg_body_front_emb).to(device).float()
            target_embedding = target_embedding / torch.norm(target_embedding)
            target_name = name

        recording = False
        selected_id = None
        embeddings_face.clear()
        embeddings_body_front.clear()
        embeddings_body_rear.clear()

cap.release()
cv2.destroyAllWindows()
