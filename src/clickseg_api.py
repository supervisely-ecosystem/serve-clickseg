import os
import sys
from typing import List

sys.path.append("ClickSEG")
from ClickSEG.isegm.inference.predictors import get_predictor, FocalPredictor
from ClickSEG.isegm.inference.evaluation import Progressive_Merge
from ClickSEG.isegm.inference.transforms import ZoomIn
from ClickSEG.isegm.inference import utils, clicker

import numpy as np
import cv2
import torch
import gdown

from src.model_zoo import get_model_zoo
from supervisely.nn.inference import InteractiveSegmentation


class UserClicker:
    def __init__(self):
        self.clicks_list = []

    def get_clicks(self, clicks_limit=None) -> List[clicker.Click]:
        return self.clicks_list[:clicks_limit]

    def add_click(self, x, y, is_positive):
        self.clicks_list.append(clicker.Click(is_positive, [y, x], indx=len(self.clicks_list)))

    def add_clicks(self, clicks: List[InteractiveSegmentation.Click]):
        for click in clicks:
            self.add_click(click.x, click.y, click.is_positive)

    def reset(self):
        self.clicks_list = []


def download_weights(url, output_path):
    if not os.path.exists(output_path):
        res = gdown.download(url, output=output_path, fuzzy=True)
        return res
    return output_path


def load_model(model_info, weights_path, device):
    # DEFAULT PARAMS:
    mode = "FocalClick"  # ['CDNet', 'Baseline', 'FocalClick']
    infer_size = 384
    thresh = 0.55
    focus_crop_r = 1.4
    target_size = 600
    target_crop_r = 1.4
    skip_clicks = -1

    mode = model_info["mode"]

    state_dict = torch.load(weights_path, map_location=device)
    model: torch.Module = utils.load_single_is_model(state_dict, device)
    predictor_params, zoom_in_params = get_predictor_and_zoomin_params(
        target_size=target_size, target_crop_r=target_crop_r, skip_clicks=skip_clicks
    )

    predictor = get_predictor(
        model,
        mode,
        device,
        infer_size=infer_size,
        prob_thresh=thresh,
        predictor_params=predictor_params,
        focus_crop_r=focus_crop_r,
        zoom_in_params=zoom_in_params,
    )
    return predictor


def get_predictor_and_zoomin_params(
    net_clicks_limit=20, target_size=(600, 600), target_crop_r=1.4, skip_clicks=-1
):
    predictor_params = {"net_clicks_limit": net_clicks_limit}
    zoom_in_params = {
        "target_size": target_size,
        "expansion_ratio": target_crop_r,
        "skip_clicks": skip_clicks,
    }
    return predictor_params, zoom_in_params


def set_prob_thres(prob_thresh, predictor):
    if hasattr(predictor, "opt_functor"):
        # only for BRS modes
        predictor.opt_functor.prob_thresh = prob_thresh

    if hasattr(predictor, "transforms"):
        for t in predictor.transforms:
            if isinstance(t, ZoomIn):
                print("set_ZoomIn")
                t.prob_thresh = prob_thresh


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
def inference_step(
    image: np.ndarray,
    predictor: FocalPredictor,
    clicker: UserClicker,
    pred_thr=0.49,
    progressive_mode=False,
) -> np.ndarray:
    # image: [H,W,C]
    h, w, c = image.shape

    # Inference
    pred_probs = predictor.get_prediction(clicker)  # np.array: [H,W]
    pred_mask = pred_probs > pred_thr

    # Merge with prev_mask
    if progressive_mode and len(clicker.get_clicks()) > 0:
        prev_mask = predictor.prev_prediction.clone().squeeze().cpu().numpy()
        last_click = clicker.get_clicks()[-1]
        last_y, last_x = last_click.coords[0], last_click.coords[1]
        pred_mask = Progressive_Merge(pred_mask, prev_mask, last_y, last_x)
        predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask, 0), 0)

    return pred_mask, pred_probs
