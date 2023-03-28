from src import clickseg_api

import os
import torch
from pathlib import Path
from PIL import Image, ImageOps

import supervisely as sly
from dotenv import load_dotenv

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict

from supervisely.app.v1.app_service import AppService
from src.model_zoo import model_zoo
from src import clickseg_api


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])

class ClickSegModel(sly.nn.inference.InstanceSegmentation):
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
        
        model_info = model_zoo[self.model_name]
        self.device = device

        sly.logger.info(f"Downloading the model {self.model_name}...")
        weights_path = os.path.join(model_dir, f"{self.model_name}.pth")
        res = clickseg_api.download_weights(model_info["weights_url"], weights_path)
        assert res is not None, "Can't download model weights"

        sly.logger.info(f"Building the model {self.model_name}...")
        self.predictor = clickseg_api.load_model(weights_path, self.device)
        self.clicker = clickseg_api.UserClicker()

        self.class_names = ["object_mask"]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionMask]:
        thres = 0.49
        crop_rect = settings['crop']

        img = clickseg_api.load_image(image_path)

        if settings['figureId'] is None:
            print("reset_input_image")
            clickseg_api.reset_input_image(img, self.predictor, self.clicker)

        # to local coords
        from itertools import chain
        for coords_xy in chain(settings['positive'], settings['negative']):
            coords_xy[0] -= crop_rect[0][0]
            coords_xy[1] -= crop_rect[0][1]

        self.clicker.reset()
        self.clicker.add_clicks(settings['positive'], settings['negative'])

        mask = clickseg_api.inference_step(img, self.predictor, self.clicker, pred_thr=thres)
        res = [sly.nn.PredictionMask(class_name=self.class_names[0], mask=mask)]
        return res
    
    def get_models(self):
        models = []
        for name, info in model_zoo.items():
            info = info.copy()
            models.append({"Model": name, **info})
        return models

    def binarize_mask(self, mask, threshold):
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        return mask

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
    # custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()

    # my_app = AppService()
    # @my_app.callback("smart_segmentation")
    # def smart_segmentation(api: sly.Api, task_id, context, state, app_logger):
    #     print("smart")
    #     request_id = context["request_id"]
    #     my_app.send_response(
    #         request_id,
    #         data={
    #             # "origin": bitmap_origin,
    #             # "bitmap": bitmap_data,
    #             "success": True,
    #             "error": None,
    #         },
    #     )
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_03.jpg"
    settings = {"figureId":None,"crop":[[305,72],[1425,1091]],"positive":[[865,582]],"negative":[],"taskId":29920}
    # results = m.predict(image_path, settings=settings)
    results = m._inference_image_id(sly.Api(), {"image_id":19524016, "settings":settings})
    vis_path = "./demo_data/image_03_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=0)
    print(f"predictions and visualization have been saved: {vis_path}")
