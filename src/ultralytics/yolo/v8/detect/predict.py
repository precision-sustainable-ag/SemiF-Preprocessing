# Ultralytics YOLO 🚀, GPL-3.0 license
import sys

import torch
import os
import json
import pandas as pd

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img.copy()).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img, classes=None):
        preds = ops.non_max_suppression(preds,
                                        0.80,#self.args.conf,
                                        0.55,#self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=1000,#self.args.max_det,
                                        classes=self.args.classes)

        results = []
        results_raw = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            #pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred, orig_shape=shape[:2]))
            results_raw.append(pred)
        return results, results_raw

    def export_predictions(self, pred_raw, save_dir, filename, names, im0s):
        
        predlist = [torch.Tensor.cpu(x).numpy().tolist() for x in pred_raw]
        #print(predlist)
        #print(type(predlist))
        #print(predlist.size)
        #print(predlist[0].shape)
        for imix, imagepred in enumerate(predlist):
            #print("imix, ", str(imix))
            for boxix, box in enumerate(imagepred):
                #print("boxix, ", str(boxix))
                predlist[imix][boxix] = {'xmin': box[0], 'ymin': box[1],
                                         'xmax': box[2], 'ymax': box[3], 
                                         'conf': box[4], 'class': box[5],
                                         'classname': names[int(box[5])]}
        
        save_path_export = str(save_dir / filename)  # im.jpg
        #os.makedirs(str(save_dir / "plant-detections"), exist_ok=True)
        #print(str(save_dir / "plant-detections"))
        # json_pred = json.dumps(predlist)
        # json_out_path = save_path_export[0:-3] + "json"
        csv_out_path = save_path_export[0:-3] + "csv"
        # yolo_out_path = save_path_export[0:-3] + "txt"
        # with open(json_out_path, 'w') as outfile:
        #     json.dump(json_pred, outfile)
            
        appended_data = pd.concat([pd.DataFrame(d) for d in predlist])
        if len(appended_data)>0:
            appended_data['class'] = appended_data['class'].astype(int)
            appended_data.index.name = 'bounding_box_id'
            #appended_data = appended_data.replace('tokim', 'dicot', regex=True)
            #appended_data = appended_data.replace('enkim', 'monocot', regex=True)
            appended_data['xmin'] = appended_data['xmin'].div(im0s.shape[1]).round(7)
            appended_data['xmax'] = appended_data['xmax'].div(im0s.shape[1]).round(7)
            appended_data['ymin'] = appended_data['ymin'].div(im0s.shape[0]).round(7)
            appended_data['ymax'] = appended_data['ymax'].div(im0s.shape[0]).round(7)
            # make sure xmin, ymin, xmax, and ymax are in the range of 0 to 1
            appended_data['xmin'] = appended_data['xmin'].clip(lower=0, upper=1)
            appended_data['xmax'] = appended_data['xmax'].clip(lower=0, upper=1)
            appended_data['ymin'] = appended_data['ymin'].clip(lower=0, upper=1)
            appended_data['ymax'] = appended_data['ymax'].clip(lower=0, upper=1)
        else:
            # Account for empty predictions
            appended_data = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class', 'classname'])
            appended_data.index.name = 'bounding_box_id'

        appended_data.to_csv(csv_out_path)
        
        # appended_data_yolo = pd.concat([pd.DataFrame(d) for d in predlist])
        # if len(appended_data_yolo)>0:
        #     appended_data.index.name = 'bounding_box_id'
        #     appended_data_yolo['class'] = appended_data_yolo['class'].astype(int)
        #     appended_data_yolo['x_center'] = (appended_data_yolo['xmax'] + appended_data_yolo['xmin']).div(2).div(im0s.shape[1]).round(7)
        #     appended_data_yolo['y_center'] = (appended_data_yolo['ymax'] + appended_data_yolo['ymin']).div(2).div(im0s.shape[0]).round(7)
        #     appended_data_yolo['x_width'] = (appended_data_yolo['xmax'] - appended_data_yolo['xmin']).div(im0s.shape[1]).round(7)
        #     appended_data_yolo['y_width'] = (appended_data_yolo['ymax'] - appended_data_yolo['ymin']).div(im0s.shape[0]).round(7)
        #     del appended_data_yolo['xmin']
        #     del appended_data_yolo['ymin']
        #     del appended_data_yolo['xmax']
        #     del appended_data_yolo['ymax']
        #     del appended_data_yolo['conf']
        #     del appended_data_yolo['classname']
        # appended_data_yolo.to_csv(yolo_out_path, index=False, header=False, sep="\t")

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
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

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



def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or "yolov8n.pt"
    source = cfg.source if cfg.source is not None else ROOT / "assets" if (ROOT / "assets").exists() \
        else "https://ultralytics.com/images/bus.jpg"

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == "__main__":
    predict()
