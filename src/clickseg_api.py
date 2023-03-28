import os
import sys
from typing import List
sys.path.append('ClickSEG')
from ClickSEG.isegm.inference import utils
from ClickSEG.isegm.inference.predictors import get_predictor, FocalPredictor
from ClickSEG.isegm.inference import clicker
from ClickSEG.isegm.inference.evaluation import Progressive_Merge

import numpy as np
import cv2
import torch
import gdown

from src.model_zoo import model_zoo


class UserClicker:
    def __init__(self):
        self.clicks_list = []

    def get_clicks(self, clicks_limit=None) -> List[clicker.Click]:
        return self.clicks_list[:clicks_limit]
    
    def add_click(self, x, y, is_positive):
        self.clicks_list.append(clicker.Click(is_positive, [y, x], indx=len(self.clicks_list)))

    def add_clicks(self, positives, negatives):
        print("Warn: add_clicks method ignores the order of clicks!")
        for x,y in positives:
            self.clicks_list.append(clicker.Click(True, [y, x], indx=len(self.clicks_list)))
        for x,y in negatives:
            self.clicks_list.append(clicker.Click(False, [y, x], indx=len(self.clicks_list)))

    def reset(self):
        self.clicks_list = []


def get_models():
    return model_zoo

def load_model(weights_path, device):
    state_dict = torch.load(weights_path, map_location='cpu')
    model: torch.Module = utils.load_single_is_model(state_dict, device)
    predictor_params, zoom_in_params = get_predictor_and_zoomin_params()
    mode = "FocalClick"  # choices=['CDNet', 'Baseline', 'FocalClick', 'NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
    infer_size = 384
    thresh = 0.49
    focus_crop_r = 1.4
    predictor = get_predictor(model, mode, device,
                                infer_size=infer_size,
                                prob_thresh=thresh,
                                predictor_params=predictor_params,
                                focus_crop_r=focus_crop_r,
                                zoom_in_params=zoom_in_params)
    return predictor

def download_weights(url, output_path):
    if not os.path.exists(output_path):
        res = gdown.download(url, output=output_path)
        return res
    return output_path

def get_predictor_and_zoomin_params(net_clicks_limit=20, target_size=(600,600), target_crop_r=1.4, skip_clicks=-1):
    predictor_params = {'net_clicks_limit': net_clicks_limit}
    zoom_in_params = {
        'target_size': target_size,
        'expansion_ratio': target_crop_r,
        'skip_clicks' : skip_clicks
    }
    return predictor_params, zoom_in_params

def load_image(image_path):    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def reset_input_image(image: np.ndarray, predictor: FocalPredictor, clicker: UserClicker):
    predictor.set_input_image(image)
    clicker.reset()

def set_prev_mask(mask: np.ndarray, predictor: FocalPredictor):
    # mask: [H,W]
    predictor.set_prev_mask(mask)

@torch.no_grad()
def inference_step(image: np.ndarray, predictor: FocalPredictor, clicker: UserClicker, pred_thr=0.49, progressive_mode=True):
    # image: [H,W,C]
    h,w,c = image.shape
    prev_mask = predictor.prev_prediction.clone().squeeze().numpy()
    
    # Inference
    pred_probs = predictor.get_prediction(clicker)  # np.array: [H,W]
    pred_mask = pred_probs > pred_thr

    # Merge with prev_mask
    if progressive_mode and len(clicker.get_clicks()) > 0:
        last_click = clicker.get_clicks()[-1]
        last_y, last_x = last_click.coords[0], last_click.coords[1]
        pred_mask = Progressive_Merge(pred_mask, prev_mask, last_y, last_x)
        predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask,0),0)

    return pred_mask
