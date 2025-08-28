import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import time
from PIL import Image as PILImage
import sys
import termios
import tty
import threading

# Modules IA
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchreid.utils.feature_extractor import FeatureExtractor
from person_following_pkg.my_strongsort import StrongSORT
from person_following_pkg.RobotDB import save_profile_to_db


class PersonFollowingNode(Node):
    def __init__(self):
        super().__init__('person_following_node')

        self.bridge = CvBridge()
        self.frame = None
        self.boxes = []
        self.tracks = []
        self.selected_id = None
        self.recording = False
        self.record_start_time = None
        self.record_duration = 20
        self.video_writer = None

        # Infos tracking
        self.target_embedding = None
        self.target_name = None
        self.target_track_id = None
        self.embeddings_face = []
        self.embeddings_body_front = []

        # Subscriptions et publisher
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        # Modèles IA
        self.model_yolo = YOLO("yolov8n.pt").to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device, min_face_size=20, post_process=False)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.extractor_front = FeatureExtractor(model_name='osnet_x1_0', device='cpu')
        self.tracker = StrongSORT(model_weights='osnet_x0_25_market1501.pt', device=self.device, fp16=False)

        # Fenêtre OpenCV
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.mouse_callback)

        # Threads
        threading.Thread(target=self.keyboard_listener, daemon=True).start()
        threading.Thread(target=self.detection_thread, daemon=True).start()

        # Mode recherche automatique
        self.searching = False
        self.last_seen_time = time.time()   # dernier moment où la personne a été vue

    # ----------------- Utilitaires -----------------
    def cosine_similarity(self, a, b):
        a = a.view(-1).cpu().numpy()
        b = b.view(-1).cpu().numpy()
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def tensor_to_list(self, tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().tolist()

    def get_closest_bbox(self, click, bboxes):
        x_click, y_click = click
        min_dist = float('inf')
        best_idx = -1
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            cx, cy = (x1+x2)/2, (y1+y2)/2
            dist = (x_click-cx)**2 + (y_click-cy)**2
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx

    # ----------------- Callbacks -----------------
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.boxes:
            self.selected_id = self.get_closest_bbox((x, y), self.boxes)
            self.recording = True
            self.record_start_time = time.time()
            self.get_logger().info(f"[INFO] Selected bbox {self.selected_id} for recording")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        self.frame = cv2.resize(frame, (640, 480))

    # ----------------- Thread clavier -----------------
    def keyboard_listener(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                key = sys.stdin.read(1)
                twist = Twist()
                if key == '\x1b':
                    key2 = sys.stdin.read(1)
                    key3 = sys.stdin.read(1)
                    if key2 == '[':
                        if key3 == 'A': twist.linear.x = 0.5
                        elif key3 == 'B': twist.linear.x = -0.5
                        elif key3 == 'C': twist.angular.z = -1.0
                        elif key3 == 'D': twist.angular.z = 1.0
                self.cmd_vel_pub.publish(twist)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # ----------------- Thread détection -----------------
    def detection_thread(self):
        while rclpy.ok():
            if self.frame is None:
                time.sleep(0.01)
                continue
            frame_copy = self.frame.copy()
            results = self.model_yolo.predict(frame_copy, classes=[0], device=self.device, verbose=False)
            self.boxes = [list(map(int, box.xyxy[0].tolist())) for box in results[0].boxes]
            detections = [[x1, y1, x2, y2, 0.99, 0] for (x1, y1, x2, y2) in self.boxes]
            self.tracks = self.tracker.update(np.array(detections), frame_copy)
            time.sleep(0.01)

    # ----------------- Boucle principale -----------------
    def show_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.frame is None:
                continue

            frame = self.frame.copy()
            best_score = -1
            current_target_box = None
            self.target_track_id = None

            # ----------------- Sélection du track cible -----------------
            for track in self.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                x, y, w, h = track.to_tlwh()
                x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                if self.target_embedding is not None:
                    crop = frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    emb = torch.tensor(self.extractor_front(crop_rgb)).float()
                    emb = emb / torch.norm(emb)
                    sim = self.cosine_similarity(self.target_embedding, emb)
                    if sim > best_score:
                        best_score = sim
                        current_target_box = (x1, y1, x2, y2)
                        self.target_track_id = track.track_id

            # ----------------- Déterminer si on doit chercher -----------------
            if current_target_box is not None:
                self.searching = False
                self.last_seen_time = time.time()
            else:
                # Activer le mode recherche seulement si la personne est perdue depuis 2s
                if time.time() - self.last_seen_time > 2.0:
                    self.searching = True

            # ----------------- Commande robot automatique -----------------
            self.publish_cmd_vel(current_target_box, best_score)

            # ----------------- Dessiner les tracks -----------------
            for track in self.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                x, y, w, h = track.to_tlwh()
                x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                color = (0, 255, 0) if track.track_id == self.target_track_id else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = self.target_name if track.track_id == self.target_track_id else f"ID {track.track_id}"
                cv2.putText(frame, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # ----------------- Enregistrement embeddings et vidéo -----------------
            if self.recording and self.selected_id is not None and self.selected_id < len(self.boxes):
                x1, y1, x2, y2 = self.boxes[self.selected_id]
                h, w = frame.shape[:2]
                x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
                y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop is not None and crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_pil = PILImage.fromarray(crop_rgb)
                        try:
                            faces = self.mtcnn(crop_pil)
                        except Exception as e :
                            self.get_logger().warn(f"MTCNN failed: {e}")
                            faces = None    
                        
                        if faces is not None and len(faces) > 0:
                            face_emb = self.resnet(faces[0].unsqueeze(0).to(self.device)).squeeze(0)
                            face_emb = face_emb / face_emb.norm()
                            self.embeddings_face.append(face_emb.detach().cpu())
                           
                        body_emb = self.extractor_front(crop_rgb)
                        body_emb = torch.tensor(body_emb, dtype=torch.float32)
                        body_emb = body_emb / torch.norm(body_emb)
                        self.embeddings_body_front.append(body_emb)

            # Fin enregistrement 20s
            if self.recording and self.record_start_time and time.time() - self.record_start_time >= self.record_duration:
                avg_face = torch.mean(torch.stack(self.embeddings_face), dim=0) if self.embeddings_face else None
                avg_body = torch.mean(torch.stack(self.embeddings_body_front), dim=0) if self.embeddings_body_front else None
                name = f"profile_{int(time.time())}"
                save_profile_to_db(name, self.tensor_to_list(avg_face), self.tensor_to_list(avg_body), None)
                self.get_logger().info(f"[SUCCESS] Profile '{name}' saved.")
                if avg_body is not None:
                    self.target_embedding = avg_body / torch.norm(avg_body)
                    self.target_name = name
                self.recording = False
                self.selected_id = None
                self.embeddings_face.clear()
                self.embeddings_body_front.clear()

            # Enregistrement vidéo
            if self.recording:
                if self.video_writer is None:
                    h, w = frame.shape[:2]
                    self.video_writer = cv2.VideoWriter(f"recording_{int(time.time())}.avi",
                                                        cv2.VideoWriter_fourcc(*"XVID"),
                                                        20.0, (w, h))
                self.video_writer.write(frame)
            elif self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            # Affichage
            status = "Recording" if self.recording else "Tracking" if self.target_embedding is not None else "Click person"
            if self.searching:
                status = "Searching..."
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.get_logger().info("ESC pressed. Shutting down.")
                rclpy.shutdown()
                break

    # ----------------- Commande robot automatique -----------------
    def publish_cmd_vel(self, target_box, score):
        twist = Twist()
        if target_box is not None and score > 0.65 and self.frame is not None:
            x1, y1, x2, y2 = target_box
            cx = (x1+x2)/2
            frame_center = self.frame.shape[1]/2
            error_x = (cx - frame_center)/frame_center
            twist.angular.z = -0.8 * error_x
            twist.linear.x = 0.35 if abs(error_x) < 0.25 else 0.0
        elif self.searching:
            # Mode recherche : tourner sur place
            twist.linear.x = 0.0
            twist.angular.z = 0.8
        self.cmd_vel_pub.publish(twist)


# ----------------- Fonction main -----------------
def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowingNode()
    try:
        node.show_loop()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
