from functools import lru_cache
from re import template
import cv2
import sys, os
import numpy as np 
import pandas as pd
from PIL import Image
import pydaisi as pyd
import streamlit as st
from constants import * 

from utils.general_util import get_template
from utils.validation_utils import is_empty, check_valid_micr, get_digits, get_alnum

import processors.autoCrop as autoCrop
from processors.AzureOCR import AzureOCR
from processors.Preprocess import Preprocess
from processors.Segmentation import Segmentation
from processors.FuzzyCorrection import FuzzyCorrection

from dotenv import load_dotenv
load_dotenv()

DIR = os.path.dirname(__file__)

signature_verification_toolkit = pyd.Daisi("soul0101/Signature Verification Toolkit")

def crop_to_cheque(img, send_steps=False):
    """
    Crops the image to the cheque.
    
    Parameters
    ----------
    img : numpy.ndarray
        The image to be cropped.
    send_steps : bool, optional
        Whether to return the intermediate steps of the cropping process.
        The default is False.
    
    Returns
    -------
    numpy.ndarray
        The cropped image.
    list
        The intermediate steps of the cropping process. [cropped_img, blurred_img, canny_img, final_contour_img]
        Only returned if send_steps is True.
    """
    return autoCrop.autocrop(img, send_steps=send_steps)

@st.experimental_memo
def clean_cheque(img):
    """
    This function takes an image containing a cheque and performs binarization
    and cleaning operations for further processing.
    
    Parameters
    ----------
    img : numpy.ndarray
        An image containing the cheque
        
    Returns
    -------
    numpy.ndarray
        The processed image of the cheque
    """
    preprocessor = Preprocess(img)
    processed_img = preprocessor.auto_preprocess()
    return processed_img

@st.experimental_memo
def segment_cheque(img, segmentation_template=None):
    """
    Segments the given image using the segmentation template
    
    Parameters
    ----------
    img: numpy.ndarray
        Image to be segmented
    segmentation_template: 
        The template used for segmentation
        
    Returns
    -------
    numpy.ndarray
        The processed image of the cheque
    """
    segmentor = Segmentation(img, template=segmentation_template)
    segmentation_result = segmentor.auto_segmentation()
    return segmentation_result

@st.experimental_memo
def azure_cheque_ocr(img, segmentation_template=None):
    """
    This function takes an image of a cheque and returns a dictionary of the
    extracted data.

    Parameters
    ----------
    img : numpy.ndarray
        The image of the cheque to be processed.
    segmentation_template : numpy.ndarray, optional
        The template to be used for segmentation. If not provided, the default
        template will be used.

    Returns
    -------
    dict
        A dictionary containing the extracted data.
    """
    azure_ocr = AzureOCR()
    return azure_ocr.process_cheque_image(img, segmentation_template=segmentation_template)

@st.experimental_memo
def extraction_result_sanitizer(cheque_extraction_result):
    """
    This function takes the result of the cheque extraction as input and returns a sanitized version of the same.
    
    Parameters
    ----------
    cheque_extraction_result : dict
        The dictionary containing the results of cheque extraction to be sanitized.

    Returns
    -------
    sanitized_result : dict
        A dictionary containing the relevant fields from the cheque extraction 
        result if the fields are valid or None for the fields that are invalid.
    """
    sanitized_result = {}

    if not any("₹" in item for item in cheque_extraction_result["courtesy_amount"]):
        print("Ruppee symbol not found in Courtesy Amount")

    if not is_empty(cheque_extraction_result["micr_strip"]):
        for field in cheque_extraction_result["micr_strip"]:
            field = field.replace(" ", "")
            if check_valid_micr(field):
                sanitized_result["micr_strip"] = field  
                break
        if "micr_strip" not in sanitized_result:
            sanitized_result["micr_strip"] = None         
    else:
        sanitized_result["micr_strip"] = None

    if not is_empty(cheque_extraction_result["acc_number"]):
        for field in cheque_extraction_result["acc_number"]:
            stripped_field = ''.join(c for c in field if c.isdigit())
            if(len(stripped_field) >= 9):
                sanitized_result["acc_number"] = stripped_field
                break
        if "acc_number" not in sanitized_result:
            sanitized_result["acc_number"] = None
    else:
        sanitized_result["acc_number"] = None

    if not is_empty(cheque_extraction_result["payee"]):
        sanitized_result["payee"] = ' '.join(cheque_extraction_result["payee"])
    else:
        sanitized_result["payee"] = None

    if not is_empty(cheque_extraction_result["courtesy_amount"]):
        amount_string = ""
        for field in cheque_extraction_result["courtesy_amount"]:
            stripped_field = get_digits(field)
            if(len(stripped_field) != 0):
                amount_string += stripped_field
        if len(amount_string) != 0:
            sanitized_result["courtesy_amount"] = int(amount_string)
        else:
            sanitized_result["courtesy_amount"] = None
    else:
        sanitized_result["courtesy_amount"] = None

    legal_amount_string = ""
    legal_amount_corrector = FuzzyCorrection()
    if not is_empty(cheque_extraction_result["legal_amount_line_1"]):
        for field in cheque_extraction_result["legal_amount_line_1"]:
            if("rupees" in field.lower().replace(" ", "")):
                continue
            closest_match = legal_amount_corrector.get_closest_match(field)
            if not is_empty(closest_match):
                legal_amount_string += " " + closest_match

    if not is_empty(cheque_extraction_result["legal_amount_line_2"]):
         for field in cheque_extraction_result["legal_amount_line_2"]:
            closest_match = legal_amount_corrector.get_closest_match(field)
            if not is_empty(closest_match):
                legal_amount_string += " " + closest_match 

    if not is_empty(legal_amount_string):
        sanitized_result["legal_amount"] = legal_amount_string[1:].capitalize()
    else:
        sanitized_result["legal_amount"] = None

    if not is_empty(cheque_extraction_result["bank_details"]):
        for detail in cheque_extraction_result["bank_details"]:
            t_detail = get_alnum(detail)
            if "ifsc" in t_detail:
                sanitized_result["IFSC_code"] = t_detail.replace("ifsc", "").upper()
    else:
        sanitized_result["IFSC_code"] = None
    

    if not is_empty(cheque_extraction_result["date_box"]):
        date_string = ''.join([get_digits(date_field) for date_field in cheque_extraction_result["date_box"]])
        if len(date_string) == 8:
            sanitized_result["date_box"] = "%s-%s-%s" % (date_string[:2], date_string[2:4],date_string[4:])
        else:
            print("Invalid Date String")
            sanitized_result["date_box"] = None
    else:
        sanitized_result["date_box"] = None
    return sanitized_result

@st.experimental_memo
def get_df_from_result(result):
    """
    This function takes a dictionary as input and returns a pandas dataframe.
    The keys of the input dictionary are used as the index of the dataframe.
    The values of the input dictionary are used as the data of the dataframe.
    
    Parameters
    ----------
    result : dict
        A dictionary
        
    Returns
    -------
    df : pandas.DataFrame
        A pandas dataframe.
    """
    keys = []
    data = []
    for key, fields in result.items():
        keys.append(key)
        data.append(fields)
    df = pd.DataFrame({"Keys": keys, "Data": data})
    return df

@st.experimental_memo
def verify_signatures(sig1_np, sig2_np):
    """
    Verify two signatures for match
    
    Parameters
    ----------
    sig1_np : numpy.ndarray
        The first signature.
    sig2_np : numpy.ndarray
        The second signature.
    
    Returns
    -------
    dict 
        A dictionary with two keys:
        - cosine_distance: the cosine distance between the two signatures
        - is_match: a boolean indicating whether the two signatures are a match or not
    """
    return signature_verification_toolkit.verify_signatures(sig1_np, sig2_np).value  

@st.experimental_memo
def cheque_verification_pipeline(cheque_image, bank_name=None, segmentation_template=None, required_segments=None):
    """
    Verify two signatures for match
    
    Parameters
    ----------
    cheque_image: numpy.ndarray
        Image of the cheque
        
    bank_name: str
        Name of the bank whose cheque this image is of. Currently supported
        banks are "axis", "icici", "syndicate", "CTS-2010" (Default template)

    segmentation_template: dict
        A list of rectangles which represents the location of different segments
        on the cheque. If this argument is not provided, a default template will be
        used. The default template is used based on the bank name provided.

    required_segments: list
        A list of the names of the segments to be extracted from the segmentation_result
        dictionary. If None, all the segments in the segmentation_result dictionary
        will be extracted in a new dictionary
    
    Returns
    -------
    raw_extraction_result: dict
        A dictionary of the raw text extracted from the different segments

    sanitized_result: dict
        A dictionary of the sanitized text extracted from the different segments

    segments: dict
        A dictionary of the segmented images from the original check image
    """

    if segmentation_template is None:
        segmentation_template = get_template(bank_name)

    cropped_img = crop_to_cheque(cheque_image)
    cleaned_img = clean_cheque(cropped_img)
    segmentation_result = segment_cheque(cleaned_img, segmentation_template=segmentation_template)
    raw_extraction_result = azure_cheque_ocr(cleaned_img, segmentation_template=segmentation_template)
    sanitized_result = extraction_result_sanitizer(raw_extraction_result)
    
    if required_segments is None:
        segments = segmentation_result
    else:
        segments = segmentation_result[required_segments]

    return raw_extraction_result, sanitized_result, segments

def st_ui():
    st.title('Welcome to the Automatic Cheque Clearing System 🤖')
    st.write("""
        In spite of the overall rapid emergence of electronic payments, huge volumes of handwritten bank cheques are issued and processed manually every day.
        
        
        The Automatic Cheque Clearing System leverages **AI fuelled** technology along with rich **rule-based** automations to organize and streamline cheque clearance.
        
        ***
        """)

    demo_type = st.sidebar.radio("Select the Demonstration ✨", ["Sample Document", "File Upload"])
    cheque_templates_dict = {
        "CTS-2010 (Default)" : DEFAULT_TEMPLATE_PATH,
        "Axis Bank" : AXIS_TEMPLATE_PATH,
        "ICICI Bank": ICICI_TEMPLATE_PATH,
        "Syndicate Bank": SYNDICATE_TEMPLATE_PATH
    }

    if demo_type == "File Upload":
        cheque_image = st.sidebar.file_uploader("Upload the Cheque Image", type=['png', 'jpg', 'jpeg'])
        cheque_template_name = st.sidebar.selectbox("Choose the cheque template", cheque_templates_dict.keys())
        cheque_template_path = cheque_templates_dict[cheque_template_name]
    else:
        selector_dict = {
            "Sample 1": {
                "image_path": "./images/icici1_tilted.png",
                "template_path": ICICI_TEMPLATE_PATH
            },
            "Sample 2": {
                "image_path": "./images/ab1.jpeg",
                "template_path": AXIS_TEMPLATE_PATH
            },
            "Sample 3": {
                "image_path": "./images/sb1.jpeg",
                "template_path": SYNDICATE_TEMPLATE_PATH
            }
        } 

        document_selector = st.sidebar.selectbox("Select the Sample Document 📄", selector_dict.keys())
        sample_cheque = selector_dict[document_selector]    
        cheque_image = sample_cheque["image_path"]
        cheque_template_path = sample_cheque["template_path"]
        
    if cheque_image is None:
        return
    
    raw_image = Image.open(cheque_image)
    cheque_template = get_template(cheque_template_path)
    raw_image = np.array(raw_image)
    st.markdown("#### Input Cheque")
    st.image(raw_image, "Input Cheque")
    
    if st.sidebar.button("Verify"):
        
        st.markdown("#### Crop and Skew Removal")
        cropped_img, image_blurred, edges, image_cnt = crop_to_cheque(raw_image, send_steps=True)
        st.image(cropped_img, "Crop and Skew Removal")
        with st.expander("Note", expanded=False):
            st.write("Take a look at the steps behind this process...")
            crop_note_col1, crop_note_col2 = st.columns(2)
            crop_note_col1.image(image_blurred, "Median Blur")
            crop_note_col2.image(edges, "Canny Edge Detection")
            crop_note_col1.image(image_cnt, "Contour Detection")
            crop_note_col2.image(cropped_img, "Four Point Transform")

        st.markdown("### Cleaning and Binarization")
        cleaned_img = clean_cheque(cropped_img)
        st.image(cleaned_img, "Cleaning and Binarization")

        st.markdown("#### Segmentation")
        with st.expander("Segments", expanded=True):
            segmentation_result = segment_cheque(cleaned_img, segmentation_template=cheque_template)
            for key, segment in segmentation_result.items():
                col1, col2 = st.columns(2)
                col1.markdown("**%s**" % (key))
                with col2:
                    st.image(segment)

        st.markdown("#### Raw Extracted Information")
        with st.spinner("Extracting Text..."):
            raw_extraction_result = azure_cheque_ocr(cleaned_img, segmentation_template=cheque_template)
        sanitized_result = extraction_result_sanitizer(raw_extraction_result)
        with st.expander("Result", expanded=False):
            st.json(raw_extraction_result)
        
        st.markdown("#### Sanitized Extracted Information")
        df = get_df_from_result(sanitized_result)
        st.dataframe(df, use_container_width=True)
        with st.expander("Note", expanded=False):
            st.write("""
                To increase the robustness of text recognition for the legal amount, a sentence 
                segmentation and fuzzy search correction algorithm has been implemented.
            """)

        st.markdown("#### Signature Verification")
        orig_sign_np = cv2.imread(os.path.join(DIR, r'./images/original_sign.png'))
        check_sign_np = segmentation_result['sign_area']


        with st.expander("Verification", expanded=True):    
            col1, col2 = st.columns(2)
            col1.image(orig_sign_np, "Original Signature")
            col2.image(check_sign_np, "Sign to be verified")   

        with st.spinner("Cleaning Signatures..."):
            cleaned_orig_sign, cleaned_check_sign = signature_verification_toolkit.signature_cleaner([orig_sign_np, check_sign_np]).value

        with st.expander("Cleaned Signatures", expanded=True):
            col1, col2 = st.columns(2)
            col1.image(cleaned_orig_sign, "Original Signature")
            col2.image(cleaned_check_sign, "Sign to be verified")

        with st.spinner("Verifying Authenticity..."):
            verification_result = verify_signatures(orig_sign_np, check_sign_np)
        
        st.markdown("##### Verification Result")
        st.json(verification_result, expanded=False)

if __name__ == "__main__":
    st_ui()
