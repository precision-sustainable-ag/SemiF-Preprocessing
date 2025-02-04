# Ultralytics YOLO 🚀, GPL-3.0 license
import sys
import argparse
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))


assert (len(sys.argv) > 2)
    
batch_name = sys.argv[1]
image_dir = sys.argv[2]


# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
#         img /= 255  # 0 - 255 to 0.0 - 1.0
#         return img

#     def write_results(self, idx, results, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  # expand for batch dim
#         self.seen += 1
#         imc = im0.copy() if self.args.save_crop else im0
#         if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  # print string
#         self.annotator = self.get_annotator(im0)

#         det = results[idx].boxes  # TODO: make boxes inherit from tensors
#         if len(det) == 0:
#             return log_string
#         for c in det.cls.unique():
#             n = (det.cls == c).sum()  # detections per class
#             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

#         # write
#         for d in reversed(det):
#             cls, conf = d.cls.squeeze(), d.conf.squeeze()
#             if self.args.save_txt:  # Write to file
#                 line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
#                     if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
#                 with open(f'{self.txt_path}.txt', 'a') as f:
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')
#             if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
#                 c = int(cls)  # integer class
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                 self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
#             if self.args.save_crop:
#                 save_one_box(d.xyxy,
#                              imc,
#                              file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
#                              BGR=True)

#         return log_string


def predict(opt, cfg=DEFAULT_CFG, use_python=True, save=True, save_dir="test_output", save_crop=True, save_txt=True, imgsz=3000):
    model = "model/train22/weights/last.pt"
    batch_name = opt.batch_name
    source = opt.source
    
    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        print("using python option...")
        YOLO(batch_name, model)(**args)
    # else:
    #     print("using cli option...")
    #     predictor = DetectionPredictor(overrides=args)
    #     predictor.predict_cli()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / '', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--batch_name', type=str, default=ROOT / 'test_batch', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--save_crop', type=bool, default=False, help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    return opt
    

if __name__ == "__main__":

    opt = parse_opt()
    predict(opt)
