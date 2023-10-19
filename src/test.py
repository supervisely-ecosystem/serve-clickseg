import os
import time
import torch
from pathlib import Path

import supervisely as sly
from dotenv import load_dotenv

from supervisely.nn.inference import InteractiveSegmentation
from supervisely.nn.prediction_dto import PredictionSegmentation

from src.model_zoo import get_model_zoo
from src import clickseg_api


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])
model_dir = "app_data/models"
conf_thres = 0.55


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

image_path = "demo_data/aniket-solankar.jpg"
clicks_json = sly.json.load_json_file("demo_data/clicks.json")
clicks = [InteractiveSegmentation.Click(**p) for p in clicks_json]

for i, model_info in enumerate(get_model_zoo()):
    model_name = model_info["model_id"]
    print(f"Evaluating {model_name}...")

    weights_path = os.path.join(model_dir, f"{model_name}.pth")
    res = clickseg_api.download_weights(model_info["weights_url"], weights_path)
    assert res is not None, f"Can't download model weights {model_info['weights_url']}"

    predictor = clickseg_api.load_model(model_info, weights_path, device)
    clicker = clickseg_api.UserClicker()

    img = clickseg_api.load_image(image_path)
    clickseg_api.reset_input_image(img, predictor, clicker)
    clicker.add_clicks(clicks)

    clickseg_api.set_prob_thres(conf_thres, predictor)
    t0 = time.time()
    pred_mask, pred_probs = clickseg_api.inference_step(
        img, predictor, clicker, pred_thr=conf_thres, progressive_merge=False
    )
    print(f"in {time.time()-t0:.2f} sec")

    res = PredictionSegmentation(mask=pred_mask)

    vis_path = f"test/hard_{i:02}_{model_name}.jpg"
    sly.image.write(vis_path, pred_mask * 255)

    vis_path = f"test/soft_{i:02}_{model_name}.jpg"
    sly.image.write(vis_path, pred_probs * 255)

    vis_path = f"test/img_{i:02}_{model_name}.jpg"
    obj_class = sly.ObjClass("test", sly.Bitmap, color=[255, 0, 0])
    geometry = sly.Bitmap(pred_mask)
    label = sly.Label(geometry, obj_class)
    ann = sly.Annotation.from_img_path(image_path).add_label(label)
    ann.draw_pretty(bitmap=img, thickness=2, output_path=vis_path)
