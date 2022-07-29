from app.models.object_detection.yolov5.detect import Inference
from app.models.object_detection.yolov5.train import Training

class InferenceDetection:
    """Inference Detection model"""
    
    def __init__(self,
                 checkpoint_path: str,
                 conf_thres: float = 0.45, 
                 iou_thres: float = 0.45, 
                 img_size: int = 640,
                ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.training_model = Training()
        self.inference_model = Inference(checkpoint_path)

    def train(self,
              batch_size: int = 20,
              epochs: int = 150,
              pretrained_weights: str = "yolov5l.pt",
              config: str = "yolov5l.yaml"
              ) -> None:
        """Training process for object_detection

        Args:
            batch_size (int, optional): batch size training. Defaults to 20.
            epochs (int, optional): num of epochs. Defaults to 150.
            pretrained_weights (str, optional): init pre-trained weighted. Defaults to "yolov5l.pt".
            config (str, optional): model configuration. Defaults to "yolov5l.yaml".
        """
        #TODO: generate training dataset yml
        return self.training_model.train(
            data = "coco128.yaml",
            epochs = epochs,
            hyp = "hyp.yaml",
            batch = batch_size,
            cfg = config,
            weights = pretrained_weights,
            imgsz = self.img_size,
        )
         
    def infer(self, source):
        """Infer process

        Args:
            source: image base 64 or array image

        Returns:
            list: result of inference. Reversing of *xyxy, conf, cls
        """
        return self.inference_model.run(
            source=source,
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres,
        )