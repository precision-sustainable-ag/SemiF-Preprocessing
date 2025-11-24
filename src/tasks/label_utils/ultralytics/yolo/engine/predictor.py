# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ yolo task=... mode=predict  model=s.pt --source 0                         # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ yolo task=... mode=predict --weights yolov8n.pt          # PyTorch
                                    yolov8n.torchscript        # TorchScript
                                    yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                    yolov8n_openvino_model     # OpenVINO
                                    yolov8n.engine             # TensorRT
                                    yolov8n.mlmodel            # CoreML (macOS-only)
                                    yolov8n_saved_model        # TensorFlow SavedModel
                                    yolov8n.pb                 # TensorFlow GraphDef
                                    yolov8n.tflite             # TensorFlow Lite
                                    yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                    yolov8n_paddle_model       # PaddlePaddle
    """
import platform
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import numpy as np
import os

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.data.augment import LetterBox
import math

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
        

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        print("image shape when converting.. ", str(img0_shape), str(img1_shape))
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        gain_y = (img1_shape[0] / img0_shape[0])  # gain  = old / new
        gain_x = (img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    #coords[:, [0, 2]] -= pad[0]  # x padding
    #coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, 0] /= gain_x
    coords[:, 1] /= gain_y
    coords[:, 2] /= gain_x
    coords[:, 3] /= gain_y
    #coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
    


class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, ):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f"{self.args.mode}"
        #self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        #self.batch_name = batch_name
        #self.save_dir = Path("/home/psa_images/temp_data/semifield-outputs")
        self.model = None
        self.data = self.args.data  # data_dict
        self.bs = None
        self.imgsz = 5000
        self.args.imgsz = 5000
        self.multiscale_inference_scales = [0.15, 0.25, 0.5, 1, 1.5]
        self.device = None
        self.classes = self.args.classes
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    def preprocess(self, img):
        pass

    def get_annotator(self, img):
        raise NotImplementedError("get_annotator function needs to be implemented")

    def write_results(self, results, batch, print_string):
        raise NotImplementedError("print_results function needs to be implemented")

    def postprocess(self, preds, img, orig_img, classes=None):
        return preds

    @smart_inference_mode()
    def __call__(self, source=None, model=None, stream=False, batch_name=None, save_dir=None, dev_mode_write_path=None):

        self.batch_name = batch_name
        self.save_dir = save_dir
        
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        # Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.dataset = load_inference_source(source=source,
                                             transforms=getattr(self.model.model, 'transforms', None),
                                             imgsz=self.imgsz,
                                             vid_stride=self.args.vid_stride,
                                             stride=self.model.stride,
                                             auto=self.model.pt)
        self.source_type = self.dataset.source_type
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    def stream_inference(self, source=None, model=None):
        self.run_callbacks("on_predict_start")
        if self.args.verbose:
            LOGGER.info("")

        # setup model
        if not self.model:
            self.setup_model(model)
        # setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        for batch in self.dataset:
            self.run_callbacks("on_predict_batch_start")
            self.batch = batch
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

            img0 = im0s[0] if isinstance(im0s, list) else im0s
            print(img0.shape)

            all_det = [] 

            for idx, im_scale in enumerate(self.multiscale_inference_scales):
                with self.dt[0]:
                    fitted_im_width = check_imgsz(int(self.args.imgsz * im_scale))
                    fitted_im_height = check_imgsz(
                        int(self.args.imgsz * im_scale / img0.shape[1] * img0.shape[0])
                    )

                    resized = cv2.resize(
                        img0,
                        (fitted_im_width, fitted_im_height),
                        interpolation=cv2.INTER_CUBIC,
                    )

                    im_multi_res_scaled = resized.transpose((2, 0, 1))[::-1].copy()
                    im_multi_res_scaled = self.preprocess(im_multi_res_scaled)
                    if im_multi_res_scaled.ndim == 3:
                        im_multi_res_scaled = im_multi_res_scaled[None]  # [1,3,H,W]

                with self.dt[1]:
                    raw = self.model(
                        im_multi_res_scaled,
                        augment=self.args.augment,
                        visualize=visualize
                    )[0]  # raw predictions for this scale

                    # Run standard Ultralytics NMS at this scale
                    nms_out = ops.non_max_suppression(
                        raw,
                        self.args.conf,
                        self.args.iou,
                        classes=self.args.classes,
                        agnostic=self.args.agnostic_nms,
                        max_det=self.args.max_det,
                    )

                    det = nms_out[0]
                    if det is None or len(det) == 0:
                        continue

                    # det is [N,6] in scale-space coords (xyxy, conf, cls)
                    # scale to original image coords
                    det[:, :4] = scale_coords(
                        img1_shape=(im_multi_res_scaled.shape[2], im_multi_res_scaled.shape[3]),
                        coords=det[:, :4],
                        img0_shape=img0.shape,
                    ).round()

                    all_det.append(det)

            with self.dt[2]:
                if len(all_det):
                    merged = torch.cat(all_det, dim=0)  # [K,6]
                else:
                    merged = torch.zeros((0, 6), device=self.device)

                self.results, self.results_raw = self.postprocess(
                    merged, img0, im0s, self.classes
                )

            self.args.save = True
            self.save_txt = False
            self.args.export_predictions = True
            if ("dev" in self.batch_name):
                self.args.save_crop = True

            for i in range(1):
                p, im0 = (path[i], im0s[i]) if self.source_type.webcam or self.source_type.from_img else (path, im0s)
                p = Path(p)

                if self.args.verbose:
                    num_det = 0
                    if hasattr(self, "results_raw") and self.results_raw is not None:
                        try:
                            num_det = int(self.results_raw.shape[0])
                        except Exception:
                            num_det = 0

                    det_str = "" if num_det > 0 else "(no detections), "
                    LOGGER.info(f"{s}{det_str}{self.dt[1].dt * 1E3:.1f}ms")

                if self.args.show:
                    self.show(p)

                if self.args.save:
                    print("Saving predictions.. " + str(self.save_dir / "inspection" / "prediction_images" / p.name))
                    os.makedirs(str(self.save_dir / "inspection" / "prediction_images"), exist_ok=True)
                    self.save_preds(vid_cap, i, str(self.save_dir / "inspection" / "prediction_images" / p.name))

                if self.args.export_predictions:
                    print("Exporting predictions.. " + str(self.save_dir / "plant-detections" / p.name))
                    os.makedirs(str(self.save_dir / "plant-detections"), exist_ok=True)
                    self.export_predictions(self.results_raw, self.save_dir / "plant-detections", p.name, self.model.names, im0)

            self.run_callbacks("on_predict_batch_end")
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f"{s}{self.dt[1].dt * 1E3:.1f}ms")  

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save_txt or self.args.save:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")

    def setup_model(self, model):
        device = select_device(self.args.device)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model, device=device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
        self.device = device
        self.model.eval()

    def show(self, p):
        im0 = self.annotator.result()
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        im0 = self.annotator.result()
        # save imgs
        if True:#self.dataset.mode == 'image':
            print("saving image...")
            h,w = im0.shape[0:2]
            resized_h, resized_w = h // 8, w // 8
            im0 = cv2.resize(im0, (resized_w, resized_h), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)
