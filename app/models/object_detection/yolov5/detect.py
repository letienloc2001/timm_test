import sys 
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, LoadBase64Images, LoadImages
from utils.general import (check_file, check_img_size,
                           non_max_suppression, scale_coords, strip_optimizer)
from utils.torch_utils import select_device, time_sync


class Inference:
    def __init__(self,
                 checkpoint_path,
                 device='',
                 update=False,
                 half=False,
                 dnn=False) -> None:
        self.device = select_device(device)
        
        # Load model
        self.model = DetectMultiBackend(checkpoint_path, device=self.device, dnn=dnn, data=None, fp16=half)
        self.update = update
        
    @torch.no_grad()
    def run(
            self,
            source,
            imgsz: Tuple[int, int] = (640, 640),
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            max_det: int = 1000,
            classes: Optional[str] = None, # filter by class: --class 0, or --class 0 2 3
            agnostic_nms: bool = True,  # class-agnostic NMS
            augment: bool = False,  # augmented inference
        ):
        
        if isinstance(source, str):
            source = str(source)
            is_file = Path(source).suffix[1:] in (IMG_FORMATS)
            if is_file:
                source = check_file(source)  # download

        stride, pt = self.model.stride, self.model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        opt = {
            "img_size": imgsz,
            "stride": stride,
            "auto": pt,
        }
        
        dataset = LoadImages(source, **opt) if isinstance(source, str) \
                else LoadBase64Images(source, **opt)
        bs = 1  # batch_size

        # Run inference
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        result = []
        for _, im, im0s, _, _ in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = self.model(im, augment=augment)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for _, det in enumerate(pred):  # per image
                seen += 1
                im0= im0s.copy()

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Format det: *xyxy, conf, cls in reversed(det)
                    result.append(det)

        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)
        return result


