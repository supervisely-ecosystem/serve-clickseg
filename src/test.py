import os
import io
import numpy as np
import torch

import supervisely as sly
from dotenv import load_dotenv

from src.clickseg_api import *

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

device = "cuda" if torch.cuda.is_available() else "cpu"
