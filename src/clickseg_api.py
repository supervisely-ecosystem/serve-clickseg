import os
import sys

sys.path.append("ClickSEG")
from ClickSEG.isegm.inference.predictors import get_predictor, FocalPredictor

# from ClickSEG.isegm.inference.transforms import ZoomIn
from ClickSEG.isegm.inference import utils, clicker

import numpy as np
import cv2
import torch
import gdown

from src.clicker import UserClicker, IterativeUserClicker


def download_weights(url, output_path):
    if not os.path.exists(output_path):
        res = gdown.download(url, output=output_path, fuzzy=True)
        return res
    return output_path


def load_model(model_info, weights_path, device):
    # DEFAULT PARAMS:
    mode = "FocalClick"  # ['CDNet', 'Baseline', 'FocalClick']
    net_clicks_limit = 100
    infer_size = 384
    thresh = 0.5
    focus_crop_r = 1.4
    target_size = 600
    target_crop_r = 1.4
    skip_clicks = -1

    mode = model_info["mode"]

    state_dict = torch.load(weights_path, map_location=device)
    model: torch.Module = utils.load_single_is_model(state_dict, device)
    predictor_params, zoom_in_params = get_predictor_and_zoomin_params(
        net_clicks_limit=net_clicks_limit,
        target_size=target_size,
        target_crop_r=target_crop_r,
        skip_clicks=skip_clicks,
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

    from isegm.inference.transforms import ZoomIn

    if hasattr(predictor, "transforms"):
        for t in predictor.transforms:
            if isinstance(t, ZoomIn):
                t.prob_thresh = prob_thresh


def set_inference_parameters(predictor: FocalPredictor, params: dict):
    conf_thres = params["conf_thres"]
    inference_resolution = params["inference_resolution"]
    focus_crop_r = params["focus_crop_r"]
    target_crop_r = params["target_crop_r"]
    # target_size = params["target_size"]
    predictor.crop_l = inference_resolution
    predictor.focus_crop_r = focus_crop_r
    predictor.refine_mode = params["refine_mode"]

    from isegm.inference.transforms import ZoomIn, ResizeTrans

    for transform in predictor.transforms:
        if isinstance(transform, ZoomIn):
            zoom_in = transform
            # zoom_in.target_size = target_size
            # zoom_in.prob_thresh = conf_thres
            zoom_in.expansion_ratio = target_crop_r
        elif isinstance(transform, ResizeTrans):
            transform.crop_height = inference_resolution
            transform.crop_width = inference_resolution


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


def Progressive_Merge_v2(pred_mask, previous_mask, y, x, is_positive):
    corr_mask = np.zeros_like(previous_mask)
    if is_positive:
        num, labels = cv2.connectedComponents(pred_mask.astype(np.uint8))
        label = labels[y, x]
        if label != 0:
            corr_mask = labels == label
            progressive_mask = np.logical_or(previous_mask, corr_mask)
        else:
            progressive_mask = previous_mask
    else:
        diff = np.logical_and(previous_mask, np.logical_not(pred_mask))
        num, labels = cv2.connectedComponents(diff.astype(np.uint8))
        label = labels[y, x]
        if label != 0:
            corr_mask = labels == label
            progressive_mask = np.logical_and(previous_mask, np.logical_not(corr_mask))
        else:
            progressive_mask = previous_mask

    # Debug
    # if True:
    #     import supervisely as sly

    #     sly.image.write("previous_mask.jpg", previous_mask * 255)
    #     sly.image.write("pred_mask.jpg", pred_mask * 255)
    #     # sly.image.write("diff_regions.jpg", diff_regions*255)
    #     sly.image.write("corr_mask.jpg", corr_mask * 255)
    #     sly.image.write("progressive_mask.jpg", progressive_mask * 255)

    #     grid = np.concatenate([previous_mask, pred_mask, corr_mask, progressive_mask], 1)
    #     sly.image.write("grid.png", grid * 255)

    return progressive_mask


@torch.no_grad()
def inference_step(
    image: np.ndarray,
    predictor: FocalPredictor,
    clicker: UserClicker,
    pred_thr=0.49,
    progressive_merge=False,
    init_mask=None,
) -> np.ndarray:
    # Set up inputs
    prev_mask = np.zeros_like(image[..., 0])
    predictor.set_input_image(image)
    if init_mask is not None:
        predictor.set_prev_mask(init_mask)
        pred_mask = init_mask
        prev_mask = init_mask
        predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask, 0), 0)

    # Inference
    pred_probs = predictor.get_prediction(clicker)  # np.array: [H,W]
    pred_mask = pred_probs > pred_thr

    # Post-processing
    # if progressive_merge:
    #     last_click = clicker.get_clicks()[-1]
    #     last_y, last_x = last_click.coords[0], last_click.coords[1]
    #     pred_mask = Progressive_Merge_v2(
    #         pred_mask, prev_mask, last_y, last_x, is_positive=last_click.is_positive
    #     )
    #     predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask, 0), 0)

    return pred_mask, pred_probs


@torch.no_grad()
def iterative_inference(
    image: np.ndarray,
    predictor: FocalPredictor,
    clicker: IterativeUserClicker,
    pred_thr=0.49,
    progressive_merge=False,
    init_mask=None,
) -> np.ndarray:
    # Set up inputs
    prev_mask = np.zeros_like(image[..., 0])
    predictor.set_input_image(image)
    if init_mask is not None:
        predictor.set_prev_mask(init_mask)
        pred_mask = init_mask
        prev_mask = init_mask
        predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask, 0), 0)

    # Inference
    for click_indx in range(len(clicker.all_clicks)):
        clicker.make_next_click()
        pred_probs = predictor.get_prediction(clicker)
        pred_mask = pred_probs > pred_thr

        # Post-processing
        if progressive_merge:
            last_click = clicker.get_clicks()[-1]
            last_y, last_x = last_click.coords[0], last_click.coords[1]
            pred_mask = Progressive_Merge_v2(
                pred_mask, prev_mask, last_y, last_x, is_positive=last_click.is_positive
            )
            predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask, 0), 0)

        prev_mask = pred_mask

        # debug
        # if True:
        #     import supervisely as sly

        #     vis_pred = np.repeat((pred_probs * 255).astype(np.uint8)[..., None], 3, axis=2)
        #     vis_pred = draw_rois(vis_pred, predictor.focus_roi, predictor.global_roi)
        #     sly.image.write(f"test_{click_indx}.png", vis_pred)

    return pred_mask, pred_probs


def draw_rois(pred_probs, focus_roi, global_roi):
    focus_roi_x = focus_roi[::2][::-1]
    focus_roi_y = focus_roi[1::2][::-1]
    global_roi_x = global_roi[::2][::-1]
    global_roi_y = global_roi[1::2][::-1]
    pred_probs = cv2.rectangle(pred_probs, focus_roi_x, focus_roi_y, [0, 255, 0], 5)
    pred_probs = cv2.rectangle(pred_probs, global_roi_x, global_roi_y, [255, 0, 0], 4)
    return pred_probs
