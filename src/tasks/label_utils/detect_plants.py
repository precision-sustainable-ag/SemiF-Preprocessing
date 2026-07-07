# Ultralytics YOLO 🚀, GPL-3.0 license
from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import hydra
import random
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.ops import batched_nms
from ultralytics import YOLO

from src.utils.utils import find_lts_dir, get_files, sanitize_time_for_path

log = logging.getLogger(__name__)

def save_csv_predictions(preds, save_path: str, class_names: dict):
    """
    preds: tensor Nx6  -> [xmin, ymin, xmax, ymax, conf, cls]  (all normalized to [0,1])
    save_path: file path (.csv)
    class_names: model.names mapping {cls_id: name}
    """
    save_path_p = Path(save_path)
    save_path_p.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bounding_box_id", "xmin", "ymin", "xmax", "ymax", "conf", "class", "classname"])

        if preds is None or len(preds) == 0:
            return

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

        empty_preds = torch.empty((0, 6), dtype=torch.float32)
        save_csv_predictions(
            preds=empty_preds,
            save_path=str(csv_path),
            class_names=names,
        )
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

def save_detection_overlay(
    image_bgr: np.ndarray,
    detections_xyxy_abs: torch.Tensor | None,
    save_path: str | Path,
    class_names: dict,
    cfg_detect: dict | None = None
) -> None:
    """
    Save a simple inspection image with detections overlaid.

    Args:
        image_bgr: Original image in BGR format.
        detections_xyxy_abs: Tensor Nx6 [x1, y1, x2, y2, conf, cls] in absolute pixel coords.
        save_path: Path to output visualization image.
        class_names: Mapping from class id -> class name.
        line_thickness: Rectangle thickness.
        font_scale: Label font scale.
        show_conf: Whether to include confidence in label text.
        draw_label_bg: Whether to draw filled background behind label text.
    """
    
    show_conf = cfg_detect.get("show_conf", True)
    draw_label_bg = cfg_detect.get("draw_label_bg", True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vis = image_bgr.copy()

    scale = float(cfg_detect.get("viz_scale", 1.0))

    if scale != 1.0:
        h, w = vis.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if detections_xyxy_abs is None or detections_xyxy_abs.numel() == 0:
        cv2.imwrite(str(save_path), vis)
        return

    dets = detections_xyxy_abs.detach().cpu().numpy()

    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
    
        if scale != 1.0:
            x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        x1, y1, x2, y2 = map(lambda v: int(round(v)), [x1, y1, x2, y2])
            
        cls_id = int(cls_id)

        cls_name = class_names.get(cls_id, str(cls_id))
        label = f"{cls_name} {conf:.2f}" if show_conf else cls_name

        color = _get_class_color(cls_id)

        # cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thickness)
        # --- dynamic thickness based on bbox size ---
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        box_scale = np.sqrt(bw * bh)  # geometric mean (more stable than width/height alone)

        min_thick = cfg_detect.get("min_line_thickness", 2)
        max_thick = cfg_detect.get("max_line_thickness", 20)
        scale_factor = cfg_detect.get("line_thickness_scale", 0.01)  # tune this
        
        scale_boost = cfg_detect.get("viz_thickness_boost", 1.0 / scale if scale < 1.0 else 1.0)
        
        dynamic_thickness = int(np.clip(box_scale * scale_factor * scale_boost, min_thick, max_thick))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, dynamic_thickness)

        base_font_scale = cfg_detect.get("font_scale", 0.6)
        font_scale_factor = cfg_detect.get("font_scale_factor", 0.0008)

        font_scale_dynamic = float(
            np.clip(box_scale * font_scale_factor, 0.4, 1.2)
        )
        font_scale = max(base_font_scale, font_scale_dynamic)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        text_x = x1
        text_y = max(y1 - 6, text_h + 4)

        if draw_label_bg:
            cv2.rectangle(
                vis,
                (text_x, text_y - text_h - 4),
                (text_x + text_w + 4, text_y + baseline - 2),
                color,
                thickness=-1,
            )
            text_color = (255, 255, 255)
        else:
            text_color = color

        cv2.putText(
            vis,
            label,
            (text_x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(str(save_path), vis)

def _get_class_color(class_id: int) -> tuple[int, int, int]:
    """
    Deterministic per-class BGR color.
    """
    palette = [
        (0, 0, 255),      # red
        (255, 255, 0),    # cyan
        (255, 0, 255),    # magenta
        (0, 255, 255),    # yellow
    ]
    return palette[class_id % len(palette)]
    
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

def edge_aware_filter(
    boxes_xyxy: np.ndarray,   # [N,4] absolute pixels
    scores: np.ndarray,       # [N]
    img_wh: tuple[int, int],  # (W, H)
    *,
    base_conf: float = 0.70,      # normal final conf
    edge_band_rel: float = 0.08,  # within 8% of the nearest edge = edge zone
    min_factor: float = 0.60,     # allow down to 60% of base_conf at the edge
    taper_rel: float = 0.20       # linearly ramp back to base_conf by 20% distance
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-box dynamic threshold:
        thr_i = base_conf * f(d_edge_rel)
      where d_edge_rel in [0, inf) is the center's normalized distance to the closest edge.
      If d_edge_rel <= edge_band_rel:
          thr_i = base_conf * min_factor
      If d_edge_rel >= taper_rel:
          thr_i = base_conf
      Else linearly interpolate between those.

    Returns:
      keep_mask: [N] bool
      dyn_thr:   [N] per-box thresholds used (float32)
    """
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)

    W, H = map(float, img_wh)
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5

    # distance (in pixels) from box center to the nearest frame edge
    d_left   = cx
    d_right  = W - cx
    d_top    = cy
    d_bottom = H - cy
    d_edge_px = np.minimum.reduce([d_left, d_right, d_top, d_bottom])

    # normalize by the smaller image dimension so it’s scale-invariant
    min_side = min(W, H)
    d_edge_rel = d_edge_px / (min_side + 1e-9)  # in [0, ~0.5]

    # piecewise-linear threshold factor
    #   close to edge → min_factor
    #   far from edge → 1.0
    #   between edge_band_rel and taper_rel → linear ramp
    f = np.ones_like(d_edge_rel, dtype=np.float32)
    near = d_edge_rel <= edge_band_rel
    far  = d_edge_rel >= taper_rel
    mid  = ~(near | far)

    f[near] = float(min_factor)
    if np.any(mid):
        # linear interpolation from (edge_band_rel -> min_factor) to (taper_rel -> 1.0)
        t = (d_edge_rel[mid] - edge_band_rel) / max(taper_rel - edge_band_rel, 1e-6)
        f[mid] = min_factor + t * (1.0 - min_factor)

    dyn_thr = (base_conf * f).astype(np.float32)
    keep_mask = scores >= dyn_thr
    return keep_mask, dyn_thr

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

    ea_cfg = getattr(cfg_detect, "edge_aware", None)
    if ea_cfg is not None and bool(getattr(ea_cfg, "enabled", False)):
        boxes_abs = boxes_norm.copy()
        boxes_abs[:, 0] *= w
        boxes_abs[:, 2] *= w
        boxes_abs[:, 1] *= h
        boxes_abs[:, 3] *= h

        keep_mask, dyn_thr = edge_aware_filter(
            boxes_xyxy=boxes_abs,
            scores=scores,
            img_wh=(int(w), int(h)),
            base_conf=float(cfg_detect.conf), 
            edge_band_rel=float(getattr(ea_cfg, "edge_band_rel", 0.08)),
            min_factor=float(getattr(ea_cfg, "min_factor", 0.60)),
            taper_rel=float(getattr(ea_cfg, "taper_rel", 0.20)),
        )

        before_ea = int(scores.shape[0])
        boxes_norm = boxes_norm[keep_mask]
        scores = scores[keep_mask]
        classes = classes[keep_mask]
        after_ea = int(scores.shape[0])

        log.info(
            f"[edge_aware] {before_ea} -> {after_ea} "
            f"(base_conf={float(cfg_detect.conf)}, edge_band_rel={float(getattr(ea_cfg,'edge_band_rel',0.08))}, "
            f"min_factor={float(getattr(ea_cfg,'min_factor',0.60))}, taper_rel={float(getattr(ea_cfg,'taper_rel',0.20))})"
        )
    else:
        log.info("[edge_aware] disabled")

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

    if bool(getattr(cfg_detect, "remove_enclosed_boxes", False)):
        before_enclosed = int(fused_abs.shape[0])

        fused_abs = remove_enclosed_boxes(
            fused_abs,
            containment_thr=float(getattr(cfg_detect, "enclosed_containment_thr", 0.98)),
            same_class_only=bool(getattr(cfg_detect, "enclosed_same_class_only", True)),
        )

        log.info(
            f"[remove_enclosed_boxes] {before_enclosed} -> {int(fused_abs.shape[0])} "
            f"(containment_thr={float(getattr(cfg_detect, 'enclosed_containment_thr', 0.98))}, "
            f"same_class_only={bool(getattr(cfg_detect, 'enclosed_same_class_only', True))})"
        )


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

    out_det_dir = Path(cfg.paths.plant_detection_dir)
    out_det_dir.mkdir(parents=True, exist_ok=True)

    save_vis = bool(getattr(cfg.detect, "save_visualizations", False))

    sanitized_time = sanitize_time_for_path(cfg.start_time) if cfg.start_time else ""
    local_inspection_dir = (
        Path(cfg.paths.batch_dir) / "inspection" / sanitized_time
        if sanitized_time else Path(cfg.paths.batch_dir) / "inspection"
    )
    out_vis_dir = local_inspection_dir / "detection_examples"
    
    viz_sample_rate = float(getattr(cfg.detect, "viz_sample_rate", 1.0))

    if save_vis:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

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

        if save_vis and random.random() < viz_sample_rate:
            vis_path = out_vis_dir / f"{Path(img_path).stem}.jpg"
            save_detection_overlay(
                image_bgr=im0,
                detections_xyxy_abs=det_abs,
                save_path=vis_path,
                class_names=names,
                cfg_detect=cfg.detect
            )

def remove_enclosed_boxes(
    dets_xyxy_conf_cls: torch.Tensor,
    containment_thr: float = 0.98,
    same_class_only: bool = True,
) -> torch.Tensor:
    """
    Remove boxes that are fully or almost fully contained inside another box.

    Args:
        dets_xyxy_conf_cls:
            Tensor Nx6 [x1, y1, x2, y2, conf, cls] in absolute pixel coords.
        containment_thr:
            Fraction of the smaller box area that must be inside the larger box
            to count as enclosed. Use 1.0 for strictly fully enclosed, or 0.98
            to tolerate tiny coordinate differences.
        same_class_only:
            If True, only remove enclosed boxes when both boxes have the same class.

    Returns:
        Filtered tensor with enclosed boxes removed.
    """
    if dets_xyxy_conf_cls is None or dets_xyxy_conf_cls.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    dets = dets_xyxy_conf_cls.float()

    boxes = dets[:, :4]
    scores = dets[:, 4]
    classes = dets[:, 5].to(torch.int64)

    n = boxes.shape[0]
    if n <= 1:
        return dets

    areas = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0)

    keep = torch.ones(n, dtype=torch.bool, device=dets.device)

    # Prefer keeping higher-confidence boxes.
    order = torch.argsort(scores, descending=True)

    for idx_i in range(n):
        i = order[idx_i]

        if not keep[i]:
            continue

        for idx_j in range(idx_i + 1, n):
            j = order[idx_j]

            if not keep[j]:
                continue

            if same_class_only and classes[i] != classes[j]:
                continue

            # Check whether lower-confidence box j is enclosed by higher-confidence box i.
            x1 = torch.maximum(boxes[i, 0], boxes[j, 0])
            y1 = torch.maximum(boxes[i, 1], boxes[j, 1])
            x2 = torch.minimum(boxes[i, 2], boxes[j, 2])
            y2 = torch.minimum(boxes[i, 3], boxes[j, 3])

            inter_w = torch.clamp(x2 - x1, min=0)
            inter_h = torch.clamp(y2 - y1, min=0)
            inter_area = inter_w * inter_h

            smaller_area = torch.minimum(areas[i], areas[j])
            containment = inter_area / torch.clamp(smaller_area, min=1e-9)

            if containment >= containment_thr:
                # Remove the lower-confidence box.
                keep[j] = False

    return dets[keep]

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
