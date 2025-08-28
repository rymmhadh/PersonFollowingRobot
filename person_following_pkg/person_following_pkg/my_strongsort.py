import numpy as np
import torch

from strongsort.reid_multibackend import ReIDDetectMultiBackend
from strongsort.sort.detection import Detection
from strongsort.sort.nn_matching import NearestNeighborDistanceMetric
from strongsort.sort.tracker import Tracker


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # center x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # center y
    y[:, 2] = x[:, 2] - x[:, 0]        # width
    y[:, 3] = x[:, 3] - x[:, 1]        # height
    return y


class StrongSORT:
    def __init__(self, model_weights, device='cpu', fp16=False, max_dist=0.2,
                 max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
        self.device = device
        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance,
                               max_age=max_age, n_init=n_init)
        self.height = None
        self.width = None

    def update(self, dets, ori_img):
        if dets is None or not len(dets) or len(dets.shape) != 2:
            return []

        if isinstance(dets, np.ndarray):
            dets = torch.from_numpy(dets).float().to(self.device)

        xyxys = dets[:, :4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        self.height, self.width = ori_img.shape[:2]

        # Convert to CPU numpy
        xywhs = xyxy2xywh(xyxys).cpu().numpy()
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        confs_np = confs.cpu().numpy()
        classes = clss.cpu().long()

        features = self._get_features(xywhs, ori_img)
        detections = [Detection(bbox_tlwh[i], confs_np[i], features[i]) for i in range(len(features))]

        self.tracker.predict()
        self.tracker.update(detections, classes, confs_np)

        return self.tracker.tracks

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone().cpu().numpy()
        else:
            raise TypeError("bbox_xywh must be np.ndarray or torch.Tensor")

        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, box):
        x, y, w, h = box
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            if y2 <= y1 or x2 <= x1:
                im_crops.append(np.zeros((128, 64, 3), dtype=np.uint8))
            else:
                crop = ori_img[y1:y2, x1:x2]
                im_crops.append(crop)
        return self.model(im_crops) if im_crops else np.array([])