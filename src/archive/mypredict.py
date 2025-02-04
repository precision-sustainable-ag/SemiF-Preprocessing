import sys
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.engine.results import Boxes
import json
import pandas as pd
import cv2


def clip_coords(boxes, shape):
    """ Clip bounding xyxy bounding boxes to image shape (height, width) """
    if isinstance(boxes, torch.Tensor):  # faster for tensors
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster for arrays)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """ Rescale coords (xyxy) from img1_shape to img0_shape """
    print(f"Scaling from {img1_shape} to {img0_shape}")
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        gain_y = img1_shape[0] / img0_shape[0]  # height scale
        gain_x = img1_shape[1] / img0_shape[1]  # width scale
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] /= gain_x
    coords[:, 1] /= gain_y
    coords[:, 2] /= gain_x
    coords[:, 3] /= gain_y

    clip_coords(coords, img0_shape)
    return coords


def save_results_as_json(results, save_path):
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)


def predict(cfg=DEFAULT_CFG, use_python=True, save=True, save_dir="test_output", save_crop=True, save_txt=True, imgsz=(6368, 9560)):
    model_path = "model/train22/weights/last.pt"
    source = Path("temp_data/semifield-outputs/MD_2024-08-26")
    images = Path(source, "images")
    save_detections = Path(source, "plant-detections")
    save_detections.mkdir(exist_ok=True, parents=True)

    imgs = list(Path(images).glob("*.jpg"))
    imgs = [x for x in imgs if x.name == "MD_1724679187.jpg"]

    multiscale_inference_scales = [0.15, 0.25, 0.5, 1, 1.5]

    if use_python:
        model: YOLO = YOLO(model_path)
        for img in imgs:
            predictions = []
            original_img = cv2.imread(str(img))
            original_height, original_width = original_img.shape[:2]  # (H, W) from OpenCV

            for im_scale in multiscale_inference_scales:
                resized_width = int(original_width * im_scale)
                resized_height = int(original_height * im_scale)
                resized_img = cv2.resize(original_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                # Run model on resized image
                results = model(resized_img, save=False, save_txt=False, imgsz=(resized_width, resized_height))
                
                for result in results:
                    preds_xyxy_all = ops.xywh2xyxy(preds)
                    
                    preds = result.boxes.data  # Extract (x1, y1, x2, y2, conf, cls)

                    # 1️⃣ Rescale predictions for this scale to the original image
                    preds = preds.clone()
                    preds[:, :4] = scale_coords(
                        (resized_height, resized_width), 
                        preds[:, :4], 
                        (original_height, original_width)
                    )

                    predictions.append(preds)  # Save rescaled predictions for NMS later

            # 2️⃣ Concatenate all predictions from multiscale inference
            all_predictions = torch.cat(predictions, dim=0)  # (N, 6) where N is total number of boxes

            # 3️⃣ Apply Non-Maximum Suppression (NMS)
            nms_predictions = ops.non_max_suppression(all_predictions.unsqueeze(0), iou_thres=0.5)[0]  # (M, 6)

            # 4️⃣ Create Boxes object with final NMS predictions
            final_boxes = Boxes(nms_predictions[:, :6], original_img.shape)  # xyxy, conf, class

            for result in results:
                result.boxes = final_boxes

                path = Path(result.path)
                image_name = path.name
                save_path = Path(save_detections, image_name)
                result.save(save_path)

            json_path = Path(save_detections, img.stem + '_predictions.json')
            print(f"Saved predictions to {json_path}")
            save_results_as_json(final_boxes.data.tolist(), json_path)


if __name__ == "__main__":
    predict()
