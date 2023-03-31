import os
import torch
from pathlib import Path

import supervisely as sly
from dotenv import load_dotenv

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict

from supervisely.nn.inference import InteractiveInstanceSegmentation
from supervisely.nn.prediction_dto import PredictionMask

from src.model_zoo import get_model_zoo
from src import clickseg_api


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])


class ClickSegModel(InteractiveInstanceSegmentation):
    def load_on_device(
        self,
        model_dir: str = None,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        if self.gui:
            self.model_name = self.gui.get_checkpoint_info()["Model"]
        else:
            self.model_name = "SegFormerB3-S2 (Comb)"
            sly.logger.warn(f"GUI can't be used, default model is {self.model_name}.")

        model_info = get_model_zoo()[self.model_name]
        self.device = device

        sly.logger.info(f"Downloading the model {self.model_name}...")
        weights_path = os.path.join(model_dir, f"{self.model_name}.pth")
        res = clickseg_api.download_weights(model_info["weights_url"], weights_path)
        assert res is not None, "Can't download model weights"

        sly.logger.info(f"Building the model {self.model_name}...")
        self.predictor = clickseg_api.load_model(self.model_name, weights_path, self.device)
        self.clicker = clickseg_api.UserClicker()

        self.class_names = ["object_mask"]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def predict(
        self,
        image_path: str,
        clicks: List[InteractiveInstanceSegmentation.Click],
        settings: Dict[str, Any],
    ) -> PredictionMask:
        conf_thres = settings.get("conf_thres", 0.55)
        clickseg_api.set_prob_thres(conf_thres, self.predictor)

        img = clickseg_api.load_image(image_path)
        clickseg_api.reset_input_image(img, self.predictor, self.clicker)
        self.clicker.add_clicks(clicks)

        pred_mask, pred_probs = clickseg_api.inference_step(
            img, self.predictor, self.clicker, pred_thr=conf_thres, progressive_mode=False
        )

        res = sly.nn.PredictionMask(class_name=self.class_names[0], mask=pred_mask)

        sly.image.write("pred_soft.png", pred_probs * 255)
        sly.image.write("pred.png", pred_mask * 255)
        c = [x.__dict__ for x in clicks]
        sly.json.dump_json_file(c, "demo_data/clicks.json", indent=2)

        return res

    def get_models(self):
        models = []
        for name, info in get_model_zoo().items():
            info.pop("weights_url")
            models.append({"Model": name, **info})
        return models

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta(
                [sly.ObjClass(self.class_names[0], sly.Bitmap, [255, 0, 0])]
            )
            self._get_confidence_tag_meta()
        return self._model_meta

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        info["model_name"] = self.model_name
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    def support_custom_models(self):
        return False


m = ClickSegModel(
    use_gui=True,
    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "demo_data/aniket-solankar.jpg"
    clicks_json = sly.json.load_json_file("demo_data/clicks.json")
    clicks = [InteractiveInstanceSegmentation.Click(**p) for p in clicks_json]
    pred = m.predict(image_path, clicks, settings={})
    vis_path = f"demo_data/prediction.jpg"
    m.visualize([pred], image_path, vis_path, thickness=0)
    print(f"predictions and visualization have been saved: {vis_path}")
