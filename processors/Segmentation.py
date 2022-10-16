import math
from utils.general_util import get_template

class Segmentation():
    def __init__(self, img, template=None):
        if template is not None:
            self.template = template
        else:
            self.template = get_template()

        self.cheque_height = self.template["cheque_height"]
        self.cheque_width = self.template["cheque_width"]
        self.img = img

    def get_scaled_coordinates(self, point_x, point_y):
        ih, iw = self.img.shape[:2]
        x = iw * point_x / self.cheque_width
        y = ih * point_y / self.cheque_height

        return (math.floor(x), math.floor(y))

    def get_scaled_xywh(self, x, y, w, h):
        ih, iw = self.img.shape[:2]
        x = iw * x / self.cheque_width
        w = iw * w / self.cheque_width
        y = ih * y / self.cheque_height
        h = ih * h / self.cheque_height
        return math.floor(x), math.floor(y), math.floor(w), math.floor(h)

    def get_cropped_segments(self, bbox):
        # print(bbox)
        x, y, w, h = self.get_scaled_xywh(*bbox)
        segment = self.img[y:y+h, x:x+w]
        return segment

    def auto_segmentation(self):
        result_builder = {}
        for key, bbox in self.template["bounding_boxes"].items():
            segments = self.get_cropped_segments(bbox)
            result_builder[key] = segments

        return result_builder

