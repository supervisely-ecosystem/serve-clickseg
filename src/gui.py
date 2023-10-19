from typing import Optional, Union, List, Dict, Callable

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.nn.inference.gui import InferenceGUI
from supervisely.app import widgets
from supervisely import Api
from supervisely import logger


class ClickSegGUI(InferenceGUI):
    def __init__(
        self,
        models: Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]],
        api: Api,
        support_pretrained_models: Optional[bool],
        support_custom_models: Optional[bool],
        add_content_to_pretrained_tab: Optional[Callable] = None,
        add_content_to_custom_tab: Optional[Callable] = None,
        custom_model_link_type: Optional[Literal["file", "folder"]] = "file",
    ):
        super().__init__(
            models=models,
            api=api,
            support_pretrained_models=support_pretrained_models,
            support_custom_models=support_custom_models,
            add_content_to_pretrained_tab=add_content_to_pretrained_tab,
            add_content_to_custom_tab=add_content_to_custom_tab,
            custom_model_link_type=custom_model_link_type,
        )
        self._inference_parameters_card = self._create_inference_parameters_card()
        self._content = widgets.Container(
            [
                self._tabs,
                self._device_field,
                self._download_progress,
                self._success_label,
                self._serve_button,
                self._change_model_button,
                self._inference_parameters_card,
            ],
            gap=5,
        )

    def get_ui(self):
        return self._content

    def _create_inference_parameters_card(self):
        self.iterative_mode_switch = widgets.Switch()
        iterative_mode_switch_f = widgets.Field(
            self.iterative_mode_switch,
            "Iterative Mode",
            "Iterative mode enhances prediction accuracy, but requires more time to compute. The calculation time increases linearly depending on the number of clicks.",
        )
        self.progressive_merge_switch = widgets.Switch()
        progressive_merge_switch_f = widgets.Field(
            self.progressive_merge_switch,
            "Progressive Merge",
            "Progressive merge is a post-processing feature that ensures that each subsequent click will only modify the prediction mask in the area directly associated with that click.",
        )
        self.conf_thres_input = widgets.InputNumber(0.5, 0.0, 1.0, step=0.05)
        conf_thres_input_f = widgets.Field(
            self.conf_thres_input,
            "Confidence Threshold",
            "",
        )
        self.inference_resolution_input = widgets.InputNumber(384, 0.0, step=64)
        inference_resolution_input_f = widgets.Field(
            self.inference_resolution_input,
            "Inference Resolution",
            "An image will be resized to this resolution before running through the model.",
        )
        self.focus_crop_r_input = widgets.InputNumber(1.4, 0.0, 5.0, step=0.2)
        focus_crop_r_input_f = widgets.Field(
            self.focus_crop_r_input,
            "Focus Crop Ratio",
            "",
        )
        self.target_crop_r_input = widgets.InputNumber(1.4, 0.0, 5.0, step=0.2)
        target_crop_r_input_f = widgets.Field(
            self.target_crop_r_input,
            "Target Crop Ratio",
            "",
        )
        info = widgets.NotificationBox(
            'Please, click the "Apply" button to update the model\'s settings.'
        )
        self.refine_mode = widgets.Switch(True)
        refine_mode_f = widgets.Field(
            self.refine_mode,
            "Refine Mode",
            "",
        )

        self.apply_btn = widgets.Button("Apply")

        @self.apply_btn.click
        def on_click():
            # Does nothing. It's a workaround to save the state in UI.
            logger.info(
                "Inference settings have been updated:", extra=self.get_inference_parameters()
            )

        content = widgets.Container(
            [
                iterative_mode_switch_f,
                progressive_merge_switch_f,
                conf_thres_input_f,
                inference_resolution_input_f,
                focus_crop_r_input_f,
                target_crop_r_input_f,
                refine_mode_f,
                info,
                self.apply_btn,
            ]
        )
        return widgets.Card("Inference Parameters", content=content)

    def get_inference_parameters(self):
        params = dict(
            iterative_mode=self.iterative_mode_switch.is_switched(),
            progressive_merge=self.progressive_merge_switch.is_switched(),
            conf_thres=self.conf_thres_input.get_value(),
            inference_resolution=self.inference_resolution_input.get_value(),
            focus_crop_r=self.focus_crop_r_input.get_value(),
            target_crop_r=self.target_crop_r_input.get_value(),
            refine_mode=self.refine_mode.is_switched()
        )
        return params
