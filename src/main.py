import os
from typing_extensions import Literal
import time
import torch
from pathlib import Path
from cachetools import LRUCache

import supervisely as sly
from dotenv import load_dotenv
from fastapi import Response, Request, status

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict, Literal, Optional, Union

from supervisely.nn.inference import InteractiveSegmentation
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.imaging import image as sly_image
from supervisely.io.fs import silent_remove
from supervisely._utils import rand_str
from supervisely.app.content import get_data_dir

from src.model_zoo import get_model_zoo
from src import clickseg_api

from src.gui import ClickSegGUI
from src.clicker import IterativeUserClicker, UserClicker


if sly.is_development() or sly.is_debug_with_sly_net():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


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

    def serve(self):
        sly.nn.inference.Inference.serve(self)
        server = self._app.get_server()
        self.cache.add_cache_endpoint(server)

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            sly.logger.debug(
                f"smart_segmentation inference: context=",
                extra={**request.state.context, "api_token": "***"},
            )

            # Parse request
            try:
                state = request.state.state
                settings = self._get_inference_settings(state)
                smtool_state = request.state.context
                api = request.state.api
                crop = smtool_state["crop"]
                positive_clicks, negative_clicks = (
                    smtool_state["positive"],
                    smtool_state["negative"],
                )
                if len(positive_clicks) + len(negative_clicks) == 0:
                    sly.logger.warn("No clicks received.")
                    response = {
                        "origin": None,
                        "bitmap": None,
                        "success": True,
                        "error": None,
                    }
                    return response
            except Exception as exc:
                sly.logger.warn("Error parsing request:" + str(exc), exc_info=True)
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "400: Bad request.", "success": False}

            # Pre-process clicks
            clicks = [{**click, "is_positive": True} for click in positive_clicks]
            clicks += [{**click, "is_positive": False} for click in negative_clicks]
            clicks = functional.transform_clicks_to_crop(crop, clicks)
            is_in_bbox = functional.validate_click_bounds(crop, clicks)
            if not is_in_bbox:
                sly.logger.warn(f"Invalid value: click is out of bbox bounds.")
                return {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }

            # Download the image if is not in Cache
            app_dir = get_data_dir()
            hash_str = functional.get_hash_from_context(smtool_state)
            if hash_str not in self._inference_image_cache:
                sly.logger.debug(f"downloading image: {hash_str}")
                image_np = functional.download_image_from_context(
                    smtool_state,
                    api,
                    app_dir,
                    self.cache.download_image,
                    self.cache.download_frame,
                    self.cache.download_image_by_hash,
                )
                self._inference_image_cache.set(hash_str, image_np)
            else:
                sly.logger.debug(f"image found in cache: {hash_str}")
                image_np = self._inference_image_cache.get(hash_str)

            # Crop the image
            image_path = os.path.join(app_dir, f"{time.time()}_{rand_str(10)}.jpg")
            if isinstance(image_np, list) and len(image_np) > 0:
                image_np = image_np[0]
            image_np = functional.crop_image(crop, image_np)
            sly_image.write(image_path, image_np)

            # Prepare init_mask (only for images)
            figure_id = smtool_state.get("figure_id")
            image_id = smtool_state.get("image_id")
            if smtool_state.get("init_figure") is True and image_id is not None:
                # Download and save in Cache
                init_mask = functional.download_init_mask(api, figure_id, image_id)
                self._init_mask_cache[figure_id] = init_mask
            elif self._init_mask_cache.get(figure_id) is not None:
                # Load from Cache
                init_mask = self._init_mask_cache[figure_id]
            else:
                init_mask = None
            if init_mask is not None:
                image_info = api.image.get_info_by_id(image_id)
                init_mask = functional.bitmap_to_mask(init_mask, image_info.height, image_info.width)
                init_mask = functional.crop_image(crop, init_mask)
                assert init_mask.shape[:2] == image_np.shape[:2]
            settings["init_mask"] = init_mask

            # Predict
            self._inference_image_lock.acquire()
            try:
                # sly.logger.debug(f"predict: {smtool_state['request_uid']}")
                clicks_to_predict = [self.Click(c["x"], c["y"], c["is_positive"]) for c in clicks]
                pred_mask = self.predict(image_path, clicks_to_predict, settings).mask
            finally:
                # sly.logger.debug(f"predict done: {smtool_state['request_uid']}")
                self._inference_image_lock.release()
                silent_remove(image_path)

            if pred_mask.any():
                bitmap = sly.Bitmap(pred_mask)
                bitmap_origin, bitmap_data = functional.format_bitmap(bitmap, crop)
                sly.logger.debug(f"smart_segmentation inference done!")
                response = {
                    "origin": bitmap_origin,
                    "bitmap": bitmap_data,
                    "success": True,
                    "error": None,
                }
            else:
                sly.logger.debug(f"Predicted mask is empty.")
                response = {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }
            return response


        @server.post("/is_online")
        def is_online(response: Response, request: Request):
            response = {"is_online": True}
            return response


        @server.post("/smart_segmentation_batched")
        def smart_segmentation_batched(response: Response, request: Request):
            response_batch = {}
            data = request.state.context["data_to_process"]
            app_session_id = sly.io.env.task_id()
            for image_idx, image_data in data.items():
                image_prediction = self.api.task.send_request(
                    app_session_id,
                    "smart_segmentation",
                    data={},
                    context=image_data,
                )
                response_batch[image_idx] = image_prediction
            return response_batch


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
