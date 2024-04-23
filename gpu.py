# """Try train the YOLO from scratch."""
# import torch
# from ultralytics import YOLO

# device: str = "mps" if torch.backends.mps.is_available() else "cpu"

# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model.to(device)
# model.train(data="coco.yaml", epochs=5)
# metrics = model.val()

import ultralytics
ultralytics.checks()