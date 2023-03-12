from detectron2.engine import DefaultPredictor 
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np 

class Detector: 
    def __init__(self, model_type="OD"): 
        self.cfg = get_cfg()
        self.model_type = model_type 

        if model_type == "OD": 
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "PS": 
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu" 

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath): 
        image = cv2.imread(imagePath) 
        if self.model_type != "PS": 
            predictions = self.predictor(image) 

            viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
                            instance_mode = ColorMode.IMAGE)
            
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        else: 
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)