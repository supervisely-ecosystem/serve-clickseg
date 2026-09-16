"""Offline harness for the production Serve ClickSEG Smart Tool routes.

It imports ``src/main.py`` (or a verbatim copy of it, used by the baseline
reproducer) exactly as production does - ``ENV=production`` so the module level
code builds the model object and calls ``ClickSegModel.serve()`` - while
replacing only the seams that need weights, a GPU, an agent or a Supervisely
instance:

* ``ClickSEG``-dependent modules (``src.clickseg_api``, ``src.clicker``) are
  stubbed, so no model code, checkpoint or CUDA context is touched;
* ``Inference.__init__`` / ``Inference.serve`` are stubbed, so no GUI, task
  session, uvicorn server or API connection is created. The FastAPI server the
  app registers its routes on is real, and so are the route handlers;
* the image cache and ``sly.Api`` are stubbed with recording doubles;
* ``ClickSegModel.predict`` is replaced by a recorder that captures the
  ``settings["init_mask"]`` array handed to the predictor.

Everything between the HTTP request and the predictor call - request parsing,
click transforms, image cropping, init-mask decoding/placement/caching and the
bitmap response encoding - is the real production code.
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import List, Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]

_TMP_DIR = tempfile.mkdtemp(prefix="clickseg-offline-")


def _prepare_environment() -> None:
    """Production env flags plus writable HOME/app-data dirs (pods run as uid 0-less users)."""
    os.environ["ENV"] = "production"
    os.environ.pop("DEBUG_WITH_SLY_NET", None)
    os.environ.pop("TASK_ID", None)
    home = os.environ.get("HOME")
    if not home or not os.access(home, os.W_OK):
        os.environ["HOME"] = _TMP_DIR
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_TMP_DIR, "cache"))
    os.environ["SLY_APP_DATA_DIR"] = os.path.join(_TMP_DIR, "app_data")
    os.makedirs(os.environ["SLY_APP_DATA_DIR"], exist_ok=True)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


_prepare_environment()

import numpy as np  # noqa: E402  (import order matters: env flags first)
import supervisely as sly  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from supervisely.nn.inference.inference import Inference  # noqa: E402
from supervisely.nn.prediction_dto import PredictionSegmentation  # noqa: E402


class AnnotationDownloadForbidden(RuntimeError):
    """Sentinel: the direct-mask contract must not download annotations/figures."""


def _stub_model_modules() -> None:
    """Replace the ClickSEG/torch model modules; tests never run inference."""
    import src  # namespace package of the app

    clickseg_api = types.ModuleType("src.clickseg_api")
    clicker = types.ModuleType("src.clicker")

    class _Clicker:
        def __init__(self, *args, **kwargs):
            raise AssertionError("offline tests must not run ClickSEG inference")

    clicker.UserClicker = _Clicker
    clicker.IterativeUserClicker = _Clicker
    for name, module in (("clickseg_api", clickseg_api), ("clicker", clicker)):
        sys.modules["src." + name] = module
        setattr(src, name, module)


class _StubGui:
    """Only the attributes the production module touches at startup."""

    class _ModelsTable:
        def __init__(self):
            self.selected_row = None

        def select_row(self, index):
            self.selected_row = index

        def get_selected_row_index(self):
            return self.selected_row

    def __init__(self):
        self._models_table = self._ModelsTable()

    def download_progress(self, *args, **kwargs):
        raise AssertionError("offline tests must not download weights")

    def get_inference_parameters(self):
        raise AssertionError("offline tests must not run ClickSEG inference")


class StubImageCache:
    """Stands in for ``InferenceImageCache``; serves prepared numpy images."""

    def __init__(self):
        self.images = {}
        self.frames = {}
        self.images_by_hash = {}
        self.calls = []

    def add_cache_endpoint(self, server):  # production calls this in serve()
        self.calls.append(("add_cache_endpoint",))

    def download_image(self, api, image_id, related=False):
        self.calls.append(("download_image", image_id))
        return self.images[image_id]

    def download_frame(self, api, video_id, frame_index):
        self.calls.append(("download_frame", video_id, frame_index))
        return self.frames[(video_id, frame_index)]

    def download_image_by_hash(self, api, image_hash):
        self.calls.append(("download_image_by_hash", image_hash))
        return self.images_by_hash[image_hash]


class _StubApp:
    def __init__(self):
        from fastapi import FastAPI

        self._server = FastAPI()

    def get_server(self):
        return self._server


def _stub_inference_init(
    self,
    model_dir=None,
    custom_inference_settings=None,
    sliding_window_mode="basic",
    use_gui=False,
    **kwargs,
):
    self._model_dir = model_dir
    self._model_meta = None
    self._custom_inference_settings = custom_inference_settings or {}
    self._sliding_window_mode = sliding_window_mode
    # keep the GUI branch of InteractiveSegmentation.__init__ so that it does
    # not try to load the model on a device
    self._use_gui = True
    self._gui = _StubGui()
    self._app = _StubApp()
    self._api = None
    self._task_id = None
    self._model_served = False
    self.autorestart = None
    self.device = None
    self.model_name = None
    self.cache = StubImageCache()


class _AnnotationApi:
    def __init__(self, root):
        self._root = root

    def download_json(self, image_id, *args, **kwargs):
        self._root.annotation_calls.append(image_id)
        result = self._root.annotation_json
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise AnnotationDownloadForbidden(
                "no annotation prepared for image {}".format(image_id)
            )
        return result


class _VideoFigureApi:
    def __init__(self, root):
        self._root = root

    def get_info_by_id(self, figure_id, *args, **kwargs):
        self._root.video_figure_calls.append(figure_id)
        result = self._root.video_figure
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise AnnotationDownloadForbidden(
                "no video figure prepared for figure {}".format(figure_id)
            )
        return result


class _VideoApi:
    def __init__(self, root):
        self.figure = _VideoFigureApi(root)


class _ImageApi:
    def __init__(self, root):
        self._root = root

    def get_info_by_id(self, image_id, *args, **kwargs):
        self._root.image_info_calls.append(image_id)
        raise AnnotationDownloadForbidden(
            "image info must not be requested for the direct-mask contract"
        )


class StubApi:
    """Recording ``sly.Api`` double: every geometry download is observable."""

    def __init__(self, annotation_json=None, video_figure=None):
        self.annotation_json = annotation_json
        self.video_figure = video_figure
        self.annotation_calls = []
        self.video_figure_calls = []
        self.image_info_calls = []
        self.annotation = _AnnotationApi(self)
        self.video = _VideoApi(self)
        self.image = _ImageApi(self)

    @property
    def download_calls(self):
        return {
            "annotation.download_json": list(self.annotation_calls),
            "video.figure.get_info_by_id": list(self.video_figure_calls),
            "image.get_info_by_id": list(self.image_info_calls),
        }


class VideoFigure:
    """Minimal ``FigureInfo`` stand-in: only ``geometry`` is used by the app."""

    def __init__(self, geometry):
        self.geometry = geometry


class PredictCall:
    def __init__(self, image_path, clicks, settings, crop_shape):
        self.image_path = image_path
        self.clicks = [(c.x, c.y, c.is_positive) for c in clicks]
        self.init_mask = settings.get("init_mask")
        self.settings = {k: v for k, v in settings.items() if k != "init_mask"}
        self.crop_shape = crop_shape


def load_app_module(source_path: Optional[Path] = None, name: str = "clickseg_app"):
    """Import the app module with production wiring, without weights or startup."""
    _stub_model_modules()
    source_path = Path(source_path) if source_path else REPO_ROOT / "src" / "main.py"
    sys.modules.pop(name, None)
    with mock.patch.object(Inference, "__init__", _stub_inference_init), mock.patch.object(
        Inference, "serve", lambda self: None
    ):
        spec = importlib.util.spec_from_file_location(name, str(source_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


class OfflineSmartToolApp:
    """The production app served by ``TestClient``, with recording doubles."""

    def __init__(self, source_path: Optional[Path] = None, name: str = "clickseg_app"):
        self.module = load_app_module(source_path, name)
        self.model = self.module.m
        self.server = self.model._app.get_server()
        self.cache = self.model.cache
        self.api = StubApi()
        self.predict_calls: List[PredictCall] = []
        self.pred_mask = None
        self.model.predict = self._predict
        self._install_state_middleware()
        self.client = TestClient(self.server)

    # request.state is populated by the Supervisely middleware in production;
    # the offline harness fills it from the request body.
    def _install_state_middleware(self):
        harness = self

        @self.server.middleware("http")
        async def _inject_request_state(request, call_next):
            body = await request.json()
            request.state.state = body.get("state", {})
            request.state.context = body.get("context", {})
            request.state.api = harness.api
            return await call_next(request)

    def _predict(self, image_path, clicks, settings):
        crop_np = sly.image.read(image_path)
        self.predict_calls.append(PredictCall(image_path, clicks, settings, crop_np.shape[:2]))
        mask = self.pred_mask
        if mask is None:
            mask = np.zeros(crop_np.shape[:2], bool)
        return PredictionSegmentation(mask=mask)

    # -- helpers ---------------------------------------------------------
    def put_image(self, image_id, image_np):
        self.cache.images[image_id] = image_np

    def put_frame(self, video_id, frame_index, image_np):
        self.cache.frames[(video_id, frame_index)] = image_np

    def post(self, path, context, state=None):
        return self.client.post(path, json={"context": context, "state": state or {}})

    @property
    def last_init_mask(self):
        assert self.predict_calls, "predictor was not called"
        return self.predict_calls[-1].init_mask


def encode_mask(data: np.ndarray) -> str:
    """Encode a boolean mask exactly like Supervisely Bitmap JSON does."""
    return sly.Bitmap.data_2_base64(np.asarray(data, dtype=bool))


def solid_image(height: int, width: int) -> np.ndarray:
    """Deterministic RGB image; JPEG-encoded by the app, so keep it flat."""
    return np.full((height, width, 3), 128, np.uint8)


def crop(x1: int, y1: int, x2: int, y2: int) -> list:
    return [{"x": x1, "y": y1}, {"x": x2, "y": y2}]
