import os, sys
import json
import cv2
import time
import pickle
import pydaisi as pyd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from dotenv import load_dotenv
load_dotenv()

automatic_cheque_clearing_system = pyd.Daisi("soul0101/Automatic Cheque Clearing System")

raw_image = np.array(Image.open('./images/icici1_tilted.png'))
segmentation_template = json.loads(open('./templates/icici.json', "r").read())

# Running Cheque Verification Pipeline
st = time.time()
print("Running Cheque Verification Pipeline...")
raw_extraction_result_pipeline, sanitized_result_pipeline, segments_pipeline = automatic_cheque_clearing_system.cheque_verification_pipeline(raw_image, segmentation_template=segmentation_template).value
print(sanitized_result_pipeline)
with open('./results/raw_extraction_result_pipeline.pickle', 'rb') as handle:
    raw_extraction_result_pipeline_gold = pickle.load(handle)
if raw_extraction_result_pipeline != raw_extraction_result_pipeline_gold:
    sys.exit(1)
with open('./results/sanitized_result_pipeline.pickle', 'rb') as handle:
    sanitized_result_pipeline_gold = pickle.load(handle)
if sanitized_result_pipeline != sanitized_result_pipeline_gold:
    sys.exit(1)
with open('./results/segments_pipeline.pickle', 'rb') as handle:
    segments_pipeline_gold = pickle.load(handle)
for key, value in segments_pipeline_gold.items():
    if(np.any(segments_pipeline[key] != segments_pipeline_gold[key])):
        sys.exit(1)
print("Time taken for running pipeline: ", time.time() - st)

# Cheque Cropping
st = time.time()
print("Performing Cheque Cropping...")
cropped_img = automatic_cheque_clearing_system.crop_to_cheque(raw_image).value
with open('./results/cropped_img.pickle', 'rb') as handle:
    cropped_img_gold = pickle.load(handle)
if np.any(cropped_img != cropped_img_gold):
    sys.exit(1)
cleaned_img = automatic_cheque_clearing_system.clean_cheque(cropped_img).value
with open('./results/cleaned_img.pickle', 'rb') as handle:
    cleaned_img_gold = pickle.load(handle)
if np.any(cleaned_img != cleaned_img_gold):
    sys.exit(1)
print("Time taken for cheque cropping: ", time.time() - st)

# Segmentation
st = time.time()
print("Running cheque segmentation...")
segmentation_result = automatic_cheque_clearing_system.segment_cheque(cleaned_img, segmentation_template=segmentation_template).value
with open('./results/segmentation_result.pickle', 'rb') as handle:
    segmentation_result_gold = pickle.load(handle)
for key, value in segmentation_result_gold.items():
    if(np.any(segmentation_result[key] != segmentation_result_gold[key])):
        sys.exit(1)
print("Time taken for running cheque segmentation: ", time.time() - st)

# Raw Information Extraction
st = time.time()
print("Extracting raw information")
raw_extraction_result = automatic_cheque_clearing_system.azure_cheque_ocr(cleaned_img, segmentation_template=segmentation_template).value
with open('./results/raw_extraction_result.pickle', 'rb') as handle:
    raw_extraction_result_gold = pickle.load(handle)
if raw_extraction_result != raw_extraction_result_gold:
    sys.exit(1)
print("Time taken to extract raw information: ", time.time() - st)

# Result Sanitization
st = time.time()
print("Running result sanitization...")
sanitized_result = automatic_cheque_clearing_system.extraction_result_sanitizer(raw_extraction_result).value
with open('./results/sanitized_result.pickle', 'rb') as handle:
    sanitized_result_gold = pickle.load(handle)
if sanitized_result != sanitized_result_gold:
    sys.exit(1)
print("Time taken to run result sanitiziation: ", time.time() - st)

# Signature Cleaning and Verification
st = time.time()
print("Running signature cleaning and verification...")
orig_sign_np = cv2.imread('./images/original_sign.png')
check_sign_np = cv2.imread('./images/original_sign.png')

cleaned_orig_sign, cleaned_check_sign = automatic_cheque_clearing_system.signature_cleaner([orig_sign_np, check_sign_np]).value
with open('./results/cleaned_signatures.pickle', 'rb') as handle:
    cleaned_orig_sign_gold, cleaned_check_sign_gold = pickle.load(handle)
if np.any(cleaned_orig_sign != cleaned_orig_sign_gold) or np.any(cleaned_check_sign != cleaned_check_sign_gold):
    sys.exit(1)

verification_result = automatic_cheque_clearing_system.verify_signatures(orig_sign_np, check_sign_np).value
with open('./results/verification_result.pickle', 'rb') as handle:
    verification_result_gold = pickle.load(handle)
if verification_result != verification_result_gold:
    sys.exit(1)
print("Time taken to run signature verification: ", time.time() - st)