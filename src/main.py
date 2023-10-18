import os
from typing_extensions import Literal
import torch
from pathlib import Path
from cachetools import LRUCache

import supervisely as sly
from dotenv import load_dotenv

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict, Literal, Optional, Union

from supervisely.nn.inference import InteractiveSegmentation
from supervisely.nn.prediction_dto import PredictionSegmentation

from src.model_zoo import get_model_zoo
from src import clickseg_api

from src.gui import ClickSegGUI
from src.clicker import IterativeUserClicker, UserClicker


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])


class ClickSegModel(InteractiveSegmentation):
    DEFAULT_ROW_IDX = 5

    def load_on_device(
        self,
        model_dir: str = None,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        if self.gui and sly.is_production():
            model_index = self.gui._models_table.get_selected_row_index()
            model_info = get_model_zoo()[model_index]
        else:
            model_info = get_model_zoo()[self.DEFAULT_ROW_IDX]
            sly.logger.warn(f"GUI can't be used, default model is {model_info['model_id']}.")

        self.model_name = model_info["model_id"]
        self.model_info = model_info
        self.device = device

        if os.path.exists(f"/checkpoints/{self.model_name}.pth"):
            weights_path = f"/checkpoints/{self.model_name}.pth"
        else:
            sly.logger.info(f"Downloading model {self.model_name}...")
            weights_path = os.path.join(model_dir, f"{self.model_name}.pth")
            pbar = self._gui.download_progress(
                message=f"Downloading model {self.model_name}...", total=1
            )
            res = clickseg_api.download_weights(model_info["weights_url"], weights_path)
            pbar.update(1)
            assert res is not None, f"Can't download model weights {model_info['weights_url']}"

        sly.logger.info(f"Building model {self.model_name}...")
        self.predictor = clickseg_api.load_model(model_info, weights_path, self.device)
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def predict(
        self,
        image_path: str,
        clicks: List[InteractiveSegmentation.Click],
        settings: Dict[str, Any],
    ) -> PredictionSegmentation:
        # Load image
        img = clickseg_api.load_image(image_path)

        # Set inference parameters
        init_mask = settings["init_mask"]
        params = self._gui.get_inference_parameters()
        if init_mask is not None:
            params["iterative_mode"] = True
        iterative_mode = params["iterative_mode"]
        progressive_merge = params["progressive_merge"]
        conf_thres = params["conf_thres"]
        clickseg_api.set_inference_parameters(self.predictor, params)

        # Inference
        if iterative_mode:
            # Init clicker
            clicker = IterativeUserClicker()
            clicker.add_clicks(clicks)

            pred_mask, pred_probs = clickseg_api.iterative_inference(
                img,
                self.predictor,
                clicker,
                pred_thr=conf_thres,
                progressive_merge=progressive_merge,
                init_mask=init_mask,
            )
        else:
            # Init clicker
            clicker = UserClicker()
            clicker.add_clicks(clicks)

            pred_mask, pred_probs = clickseg_api.inference_step(
                img,
                self.predictor,
                clicker,
                pred_thr=conf_thres,
                progressive_merge=progressive_merge,
                init_mask=init_mask,
            )

        res = PredictionSegmentation(mask=pred_mask)

        # debug
        if sly.is_debug_with_sly_net():
            sly.image.write("crop.jpg", img)
            sly.image.write("pred.jpg", pred_mask * 255)
            sly.image.write("pred_probs.jpg", pred_probs * 255)
            sly.fs.silent_remove("init_mask.png")
            if init_mask is not None:
                sly.image.write("init_mask.png", init_mask)
        return res

    def get_models(self):
        models = []
        for info in get_model_zoo():
            info.pop("weights_url")
            info.pop("model_id")
            models.append(info)
        return models

    def get_info(self):
        info = super().get_info()
        info["model_name"] = self.model_name
        return info

    def support_custom_models(self):
        return False

    def initialize_gui(self) -> None:
        models = self.get_models()
        support_pretrained_models = True
        self._gui = ClickSegGUI(
            models,
            self.api,
            support_pretrained_models=support_pretrained_models,
            support_custom_models=self.support_custom_models(),
            add_content_to_pretrained_tab=self.add_content_to_pretrained_tab,
            add_content_to_custom_tab=self.add_content_to_custom_tab,
            custom_model_link_type=self.get_custom_model_link_type(),
        )


# inference_settings_path = os.path.join(root_source_path, "custom_settings.yaml")
inference_settings_path = None

if sly.is_production() and not os.environ.get("DEBUG_WITH_SLY_NET"):
    # production
    m = ClickSegModel(use_gui=True, custom_inference_settings=inference_settings_path)
    m.gui._models_table.select_row(ClickSegModel.DEFAULT_ROW_IDX)
    m.serve()
else:
    # debug
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m = ClickSegModel(use_gui=True, custom_inference_settings=inference_settings_path)
    m.gui._models_table.select_row(ClickSegModel.DEFAULT_ROW_IDX)
    # m.load_on_device(m.model_dir, device)
    if os.environ.get("DEBUG_WITH_SLY_NET"):
        print("mode=DEBUG_WITH_SLY_NET")
        m.serve()
    else:
        print("mode=LOCAL_DEBUG")
        image_path = "demo_data/aniket-solankar.jpg"
        clicks_json = sly.json.load_json_file("demo_data/clicks.json")
        clicks = [InteractiveSegmentation.Click(**p) for p in clicks_json]
        pred = m.predict(image_path, clicks, settings={})
        vis_path = f"demo_data/prediction.jpg"
        m.visualize([pred], image_path, vis_path, thickness=0)
        print(f"predictions and visualization have been saved: {vis_path}")
