# Ultralytics YOLO 🚀, GPL-3.0 license
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import csv
import hydra
import numpy as np
from omegaconf import DictConfig
import logging
from pathlib import Path
from src.utils.utils import find_lts_dir, get_files

log = logging.getLogger(__name__)


def save_csv_predictions(preds, save_path: str, class_names: dict):
    """
    preds: tensor Nx6  -> [xmin, ymin, xmax, ymax, conf, cls]
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

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img.copy()).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    #def postprocess(self, preds, img, orig_img, classes=None):
    #    preds = ops.non_max_suppression(preds,
    #                                    self.args.conf,
    #                                    self.args.iou,
    #                                    agnostic=self.args.agnostic_nms,
    #                                    max_det=self.args.max_det,
    #                                    classes=self.args.classes)
#
#        results = []
#        for i, pred in enumerate(preds):
#            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
#            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
#            results.append(Results(boxes=pred, orig_shape=shape[:2]))
#        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()
            c_int = int(c)
            if c_int in self.model.names:
                cls_name = self.model.names[c_int]
            else:
                cls_name = f"cls{c_int}"
            log_string += f"{n} {cls_name}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string
    
    def export_predictions(self, results_raw, save_dir, filename, names, im0):
        """
        Save detections in YOLO txt format:
        one line per box:  cls x_center y_center width height conf
        (all coords normalized to [0,1] w.r.t image size).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem
        out_path = save_dir / f"{stem}.txt"

        if results_raw is None or results_raw.numel() == 0:
            out_path.touch()
            return

        boxes = results_raw.detach().cpu() 
        xyxy = boxes[:, :4]
        conf = boxes[:, 4]
        cls = boxes[:, 5].to(torch.int64)

        h, w = im0.shape[:2]

        # xyxy → normalized xywh
        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]

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
        
        csv_path = save_dir / f"{stem}.csv"

        xyxy_norm = boxes[:, :4].clone()
        xyxy_norm[:, 0] /= w 
        xyxy_norm[:, 2] /= w    
        xyxy_norm[:, 1] /= h    
        xyxy_norm[:, 3] /= h   

        csv_preds = torch.cat([xyxy_norm, boxes[:, 4:6]], dim=1)

        save_csv_predictions(
            preds=csv_preds,
            save_path=csv_path,
            class_names=names
        )

def predict(opt, cfg=DEFAULT_CFG, use_python=True, save=True, save_dir="test_output", save_crop=True, save_txt=True, imgsz=3000):
    # model = "data/runs_yolov8/detect/train22/weights/last.pt"  #cfg.model or "yolov8n.pt"
    model = opt["model_path"] #cfg.model or "yolov8n.pt"
    batch_name = opt["batch_name"]
    source = opt["source"] #Path("test_images/"  #cfg.source if cfg.source is not None else ROOT / "assets" if (ROOT / "assets").exists() \
        #else "https://ultralytics.com/images/bus.jpg"
    save_dir = opt["save_dir"]
    
#/home/psa_images/temp_data/semifield-upload/
    args = dict(model=model, source=source)
    if use_python:
        print("using python option...")
        predictor = DetectionPredictor(overrides=args)
        predictor(source=source, model=model, batch_name=batch_name, save_dir=save_dir)
    else:
        print("using cli option...")
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Starting detection for batch {cfg.batch_id}")
    
    source = cfg.paths.down_photos
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
            "save_dir": save_dir}

        predict(opt)
        log.info("Detection completed.")
    except Exception as e:
        log.error(f"An error occurred during detection: {e}", exc_info=True)
        raise 
    
    return
    
    
if __name__ == "__main__":
    main()