from ClickSEG.isegm.inference import utils, clicker
from supervisely.nn.inference import InteractiveSegmentation
from typing import List


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


class IterativeUserClicker:
    def __init__(self):
        self.all_clicks = []
        self.clicks_list = []
        self.cur_idx = 0

    def get_clicks(self, clicks_limit=None) -> List[clicker.Click]:
        return self.clicks_list[:clicks_limit]

    def add_click(self, x, y, is_positive):
        self.all_clicks.append(clicker.Click(is_positive, [y, x], indx=len(self.clicks_list)))

    def add_clicks(self, clicks: List[InteractiveSegmentation.Click]):
        for click in clicks:
            self.add_click(click.x, click.y, click.is_positive)

    def reset(self):
        self.all_clicks = []
        self.clicks_list = []
        self.cur_idx = 0

    def make_next_click(self):
        next_click = self.all_clicks[self.cur_idx]
        self.clicks_list.append(next_click)
        self.cur_idx += 1

