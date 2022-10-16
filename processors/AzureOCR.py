import io 
import os
import cv2
import time
import json 
import numpy as np 
from shapely.geometry import Polygon
from utils.general_util import get_template
from utils.img_utils import get_scaled_xywh

from dotenv import load_dotenv
load_dotenv()

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = os.getenv('AZURE_CV_SUBSCRIPTION_KEY')
endpoint = os.getenv('AZURE_CV_ENDPOINT')

class AzureOCR():
    def __init__(self):
        self.computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    def recognize_text(self, read_image):
        """RecognizeTextUsingRecognizeAPI.
        This will recognize text of the given image using the recognizeText API.
        """
        is_success, buffer = cv2.imencode(".jpg", read_image)
        io_buf = io.BytesIO(buffer) 
        # Call API with image and raw response (allows you to get the operation location)
        read_response = self.computervision_client.read_in_stream(io_buf, raw=True)
        # Get the operation location (URL with ID as last appendage)
        read_operation_location = read_response.headers["Operation-Location"]
        # Take the ID off and use to get results
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for the retrieval of the results
        timeout = 60
        start_time = time.time()
        while True:
            read_result = self.computervision_client.get_read_result(operation_id)
            if read_result.status.lower() not in ['notstarted', 'running']:
                break
            print ('Waiting for result...')
            if time.time() - start_time > timeout:
                break
            time.sleep(5)

        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                # Debugging
                # for line in text_result.lines:
                #     print(line.text)
                #     print(line.bounding_box)
                return text_result.lines

    def cvt_tempbb_polygonbb(self, img, template, bbox):
        x, y, w, h = get_scaled_xywh(img, template["cheque_width"], template["cheque_height"], *bbox)
        return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]] 
    
    def cvt_azurebb_polygonbb(self, bb):
        return [[bb[0], bb[1]], [bb[2], bb[3]], [bb[4], bb[5]], [bb[6], bb[7]]]

    def get_roi_key(self, azure_segment_box, cheque_areas):
        max_iou = 0
        best_area = None

        for key, cheque_segment_box in cheque_areas.items():
            poly_1 = Polygon(azure_segment_box)
            poly_2 = Polygon(cheque_segment_box)

            iou = poly_1.intersection(poly_2).area / min(poly_1.area, poly_2.area)
            if iou > 0.5 and iou > max_iou:
                max_iou = iou
                best_area = key

        return best_area

    # def get_dummy_data(self):
    #     file1 = open('azure_result.txt', 'r', encoding="utf8")
    #     recognized_lines = []

    #     class Dummy():
    #         def __init__(self, text, bounding_box):
    #             self.text = text
    #             self.bounding_box = bounding_box

    #     while True:
    #         text = file1.readline()
    #         bbox = file1.readline()
    #         if text:
    #             recognized_lines.append(Dummy(text.rstrip(), json.loads(bbox)))
    #         else:
    #             break

    #     return recognized_lines
            
    def process_cheque_image(self, img, segmentation_template=None):
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image is not an numpy.array")
        if segmentation_template is None:
            segmentation_template = get_template()
        recognized_lines = self.recognize_text(img)
        # recognized_lines = self.get_dummy_data()

        bounding_boxes = segmentation_template["bounding_boxes"]
        cheque_areas = {}
        recognized_areas = {}
        for key, segment in bounding_boxes.items():
            recognized_areas[key] = []
            cheque_areas[key] = self.cvt_tempbb_polygonbb(img, segmentation_template, segment)

        for line in recognized_lines:
            pts = self.cvt_azurebb_polygonbb(line.bounding_box)
            key_area = self.get_roi_key(pts, cheque_areas)
            if key_area is not None:
                recognized_areas[key_area].append(line.text)

        return recognized_areas

if __name__ == "__main__":
    img = cv2.imread('./images/doc5.jpg') 
    is_success, buffer = cv2.imencode(".jpg", img)
    io_buf = io.BytesIO(buffer) 

    instance = AzureOCR()
    print(instance.process_cheque_image(img))
    # instance.recognize_text(io_buf)