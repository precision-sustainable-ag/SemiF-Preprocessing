# Ultralytics YOLO 🚀, GPL-3.0 license
from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.ops import batched_nms
from ultralytics import YOLO

from src.utils.utils import find_lts_dir, get_files

log = logging.getLogger(__name__)

def save_csv_predictions(preds, save_path: str, class_names: dict):
    """
    preds: tensor Nx6  -> [xmin, ymin, xmax, ymax, conf, cls]  (all normalized to [0,1])
    save_path: file path (.csv)
    class_names: model.names mapping {cls_id: name}
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bounding_box_id", "xmin", "ymin", "xmax", "ymax", "conf", "class", "classname"])

        for i, det in enumerate(preds):
            xmin, ymin, xmax, ymax, conf, cls_id = det.tolist()
            cls_id = int(cls_id)
            cls_name = class_names.get(cls_id, "unknown")
            writer.writerow([i, xmin, ymin, xmax, ymax, conf, cls_id, cls_name])


def export_predictions(results_raw_xyxy_abs: torch.Tensor, save_dir, filename, names, im0):
    """
    Save detections in YOLO txt format:
    one line per box:  cls x_center y_center width height conf
    (all coords normalized to [0,1] w.r.t image size).

    Also saves CSV: normalized xyxy + conf + cls.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    out_path = save_dir / f"{stem}.txt"
    csv_path = save_dir / f"{stem}.csv"

    if results_raw_xyxy_abs is None or results_raw_xyxy_abs.numel() == 0:
        out_path.touch()
        csv_path.touch()
        return

    boxes = results_raw_xyxy_abs.detach().cpu().float()
    xyxy = boxes[:, :4]
    conf = boxes[:, 4]
    cls = boxes[:, 5].to(torch.int64)

    h, w = im0.shape[:2]

    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    with open(out_path, "w") as f:
        for i in range(xc.shape[0]):
            f.write(
                f"{int(cls[i])} "
                f"{xc[i]:.6f} {yc[i]:.6f} "
                f"{bw[i]:.6f} {bh[i]:.6f} "
                f"{conf[i]:.6f}\n"
            )

    xyxy_norm = xyxy.clone()
    xyxy_norm[:, 0] /= w
    xyxy_norm[:, 2] /= w
    xyxy_norm[:, 1] /= h
    xyxy_norm[:, 3] /= h

    csv_preds = torch.cat([xyxy_norm, conf[:, None], cls[:, None].float()], dim=1)
    save_csv_predictions(preds=csv_preds, save_path=str(csv_path), class_names=names)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter + 1e-9
    return inter / union


def weighted_box_fusion_single_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
    score_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    boxes:  Nx4 normalized xyxy
    scores: N
    returns: fused_boxes (Mx4), fused_scores (M)
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    keep = scores >= float(score_thr)
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]

    fused_boxes = []
    fused_scores = []
    used = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if used[i]:
            continue

        cluster = [i]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if iou_xyxy(boxes[i], boxes[j]) >= float(iou_thr):
                cluster.append(j)
                used[j] = True

        cb = boxes[cluster]
        cs = scores[cluster]

        w = cs / (cs.sum() + 1e-9)
        fused = (cb * w[:, None]).sum(axis=0)
        fused_score = float((cs * w).sum())

        fused_boxes.append(fused)
        fused_scores.append(fused_score)

    return np.stack(fused_boxes, axis=0).astype(np.float32), np.asarray(fused_scores, dtype=np.float32)


def weighted_box_fusion_all_classes(
    boxes_xyxy_norm: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: float,
    score_thr: float,
) -> torch.Tensor:
    """
    returns: tensor Mx6 [x1,y1,x2,y2,conf,cls] in normalized coords
    """
    boxes_xyxy_norm = np.asarray(boxes_xyxy_norm, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)

    out = []
    for cls_id in np.unique(classes):
        m = classes == cls_id
        fb, fs = weighted_box_fusion_single_class(
            boxes_xyxy_norm[m],
            scores[m],
            iou_thr=iou_thr,
            score_thr=score_thr,
        )
        if len(fb):
            cls_col = np.full((len(fb), 1), cls_id, dtype=np.float32)
            conf_col = fs.reshape(-1, 1).astype(np.float32)
            out.append(np.concatenate([fb, conf_col, cls_col], axis=1))

    if not out:
        return torch.zeros((0, 6), dtype=torch.float32)

    out = np.concatenate(out, axis=0)
    out = out[np.argsort(-out[:, 4])]
    return torch.from_numpy(out).float()

def nms_xyxy_abs(dets_xyxy_conf_cls: torch.Tensor, iou_thr: float, max_det: int) -> torch.Tensor:
    """
    dets: Nx6 [x1,y1,x2,y2,conf,cls] absolute coords
    returns: Mx6 after class-aware NMS
    """
    if dets_xyxy_conf_cls is None or dets_xyxy_conf_cls.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    dets = dets_xyxy_conf_cls.float()
    boxes = dets[:, :4]
    scores = dets[:, 4]
    cls = dets[:, 5].to(torch.int64)

    keep = batched_nms(boxes, scores, cls, float(iou_thr))
    keep = keep[: int(max_det)]
    return dets[keep]

def run_multiscale(model: YOLO, im0_bgr, cfg_detect, device=None) -> torch.Tensor:
    """
    cfg_detect should contain:
      base_imgsz: int
      scales: list[float]
      per_scale_conf: float
      per_scale_iou: float
      per_scale_max_det: int
      conf: float
      iou: float
      final_max_det: int
      post_fusion_nms:
        enabled: bool
        iou: float

    Optional (recommended, but not required):
      wbf_iou: float
      wbf_score_thr: float
    """
    h, w = im0_bgr.shape[:2]

    base_imgsz = int(cfg_detect.base_imgsz)
    scale_factors = list(cfg_detect.scales)
    imgszs = [max(32, int(round(base_imgsz * float(s)))) for s in scale_factors]

    per_conf = float(cfg_detect.per_scale_conf)
    per_iou = float(cfg_detect.per_scale_iou)
    per_max_det = int(cfg_detect.per_scale_max_det)

    final_conf = float(cfg_detect.conf)
    final_iou = float(cfg_detect.iou)
    final_max_det = int(cfg_detect.final_max_det)

    wbf_iou = float(getattr(cfg_detect, "wbf_iou", 0.55))
    wbf_score_thr = float(getattr(cfg_detect, "wbf_score_thr", 0.001))

    all_boxes_norm, all_scores, all_classes = [], [], []

    im_rgb = cv2.cvtColor(im0_bgr, cv2.COLOR_BGR2RGB)

    for imgsz in imgszs:
        preds = model.predict(
            source=im_rgb,
            imgsz=int(imgsz),
            conf=per_conf,
            iou=per_iou,
            max_det=per_max_det,
            device=device,
            verbose=False,
            save=False,
        )

        r = preds[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy_abs = r.boxes.xyxy.detach().cpu().float().numpy()
        scores = r.boxes.conf.detach().cpu().float().numpy()
        classes = r.boxes.cls.detach().cpu().float().numpy().astype(np.int64)

        xyxy_norm = xyxy_abs.copy()
        xyxy_norm[:, 0] /= w
        xyxy_norm[:, 2] /= w
        xyxy_norm[:, 1] /= h
        xyxy_norm[:, 3] /= h

        all_boxes_norm.append(xyxy_norm)
        all_scores.append(scores)
        all_classes.append(classes)

    if not all_boxes_norm:
        return torch.zeros((0, 6), dtype=torch.float32)

    boxes_norm = np.concatenate(all_boxes_norm, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    raw_n = int(boxes_norm.shape[0])
    log.info(
        f"[multiscale] raw_concat={raw_n} "
        f"(scales={len(imgszs)}, per_conf={per_conf}, per_iou={per_iou}, per_max_det={per_max_det})"
    )

    fused_norm = weighted_box_fusion_all_classes(
        boxes_xyxy_norm=boxes_norm,
        scores=scores,
        classes=classes,
        iou_thr=wbf_iou,
        score_thr=wbf_score_thr,
    )

    wbf_n = int(fused_norm.shape[0])
    log.info(
        f"[wbf] fused={wbf_n} "
        f"(wbf_iou={wbf_iou}, wbf_score_thr={wbf_score_thr})"
    )

    if fused_norm.numel() == 0:
        return fused_norm

    fused_abs = fused_norm.clone()
    fused_abs[:, 0] *= w
    fused_abs[:, 2] *= w
    fused_abs[:, 1] *= h
    fused_abs[:, 3] *= h

    # optional post-fusion NMS
    if hasattr(cfg_detect, "post_fusion_nms") and bool(cfg_detect.post_fusion_nms.enabled):
        fused_abs = nms_xyxy_abs(
            fused_abs,
            iou_thr=float(cfg_detect.post_fusion_nms.iou),
            max_det=final_max_det,
        )

    if fused_abs.numel() == 0:
        return fused_abs

    fused_abs = fused_abs[fused_abs[:, 4] >= final_conf]
    fused_abs = nms_xyxy_abs(fused_abs, iou_thr=final_iou, max_det=final_max_det)

    return fused_abs

def predict(opt, cfg: DictConfig):
    model_path = opt["model_path"]
    batch_name = opt["batch_name"]
    source = opt["source"]
    save_dir = Path(opt["save_dir"])

    if isinstance(source, tuple):
        source_path = Path(source[0])
    else:
        source_path = Path(source)

    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    model = YOLO(model_path)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    out_det_dir = save_dir / "detections" / batch_name
    out_det_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.detect.device if "detect" in cfg and "device" in cfg.detect else None

    img_files = get_files(cfg, "detect_plants")
    if not img_files:
        log.warning(f"No images found under {source_path}")
        return

    for img_path in img_files:
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            log.warning(f"Failed to read {img_path}")
            continue

        det_abs = run_multiscale(
            model=model,
            im0_bgr=im0,
            cfg_detect=cfg.detect,
            device=device,
        )

        export_predictions(
            results_raw_xyxy_abs=det_abs,
            save_dir=out_det_dir,
            filename=str(img_path),
            names=names,
            im0=im0,
        )


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Starting detection for batch {cfg.batch_id}")

    batch_name = cfg.batch_id
    lts_dir = find_lts_dir(batch_name, cfg.paths.lts_locations, developed=True, jpgs=True)

    source = (Path(lts_dir) / "semifield-developed-images" / batch_name / "images", cfg)
    model_path = Path(cfg.paths.local_detection_model)

    if not model_path.exists():
        log.error(f"Model path {model_path} does not exist.")
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    save_dir = Path(cfg.paths.batch_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        opt = {
            "model_path": str(model_path),
            "source": source,
            "batch_name": batch_name,
            "save_dir": save_dir,
        }

        predict(opt, cfg)
        log.info("Detection completed.")
    except Exception as e:
        log.error(f"An error occurred during detection: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
