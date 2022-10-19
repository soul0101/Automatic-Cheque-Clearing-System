# Automatic-Cheque-Clearing-System

In spite of the overall rapid emergence of electronic payments, huge volumes of handwritten bank cheques are issued and processed manually every day.

The Automatic Cheque Clearing System leverages AI fuelled technology along with rich rule-based automations to optimize, organize and streamline cheque clearance.

### Flow Diagram
![flow](https://user-images.githubusercontent.com/53980340/196203080-9c9c552a-dd04-4d8d-87d3-d36440b02430.png)

### Crop and Skew Removal 

In scenarios where the scanned cheque is misaligned and not to scale, the daisi runs robust computer vision algorithms to detect and crop to the cheque boundary. 

![image](https://user-images.githubusercontent.com/53980340/196203985-9debd44b-2589-4dc2-b57b-de72136b9549.png)

### Cleaning and Binarization
After cropping the cheque is preprocessed to be cleaned and binarized for uniform text recognition. 

![image](https://user-images.githubusercontent.com/53980340/196204386-01d0dbd5-1291-4b86-b669-097522755dfd.png)

### Segmentation
All Indian banks have to follow the CTS-2010 format of the cheque which lays down dimensions for a valid cheque. We use these dimensions as a template for segmenting our cheques.
The daisi is flexible in terms of input templates and can be modified as per the needs. 

![image](https://user-images.githubusercontent.com/53980340/196204727-1f7af578-f600-4dc4-9f0c-114772e0e7c2.png)

### Extracted Information
After segmentation, OCR (Optical Character Recognition) and ICR (Intelligent Character Recognition) algorithms are run to detect printed and handwritten text. 
The raw extracted text is further sanitized to account for errors in the Text Recognition. <br>
***
For example, a **fuzzy segmentation and correction algorithm** is in place for detecting the legal amount. <br>
**"One LackTwenthosand" -> "One Lack Ten Thousand"**
***

![image](https://user-images.githubusercontent.com/53980340/196206028-6cd1b0ce-0910-4a76-8cdd-6d29a076772d.png)

### Signature Verification
The signature is paramount to verify the authenticity of the cheque, a deep learning signature extraction and verification algorithm is used to extract, clean and 
verify the signature. 

![image](https://user-images.githubusercontent.com/53980340/196206804-d6bd5348-0644-4b1c-ae67-3d94ea6eba56.png)

### Example API
```python
import os, sys
import json
import numpy as np
from PIL import Image

import pydaisi as pyd
automatic_cheque_clearing_system = pyd.Daisi("soul0101/Automatic Cheque Clearing System")

raw_image = np.array(Image.open('./images/icici1_tilted.png'))
segmentation_template = json.loads(open('./templates/icici.json', "r").read())

raw_extraction_result_pipeline, sanitized_result_pipeline, segments_pipeline = automatic_cheque_clearing_system.cheque_verification_pipeline(raw_image, segmentation_template=segmentation_template).value
```

### Template Format
```
All dimensions are in millimeter
{
  "cheque_height": <cheque height>,
  "cheque_width": <cheque width>,
  "bounding_boxes": {
    "acc_number": [x, y, w, h],
    "bank_details": [x, y, w, h],
    "payee": [x, y, w, h],
    "courtesy_amount": [x, y, w, h],
    "legal_amount_line_1": [x, y, w, h],
    "legal_amount_line_2": [x, y, w, h],
    "date_box": [x, y, w, h],
    "micr_strip": [x, y, w, h],
    "sign_area": [x, y, w, h]
  }
}
```

### CTS-2010 Template

```json
{
  "cheque_height": 92,
  "cheque_width": 202,
  "bounding_boxes": {
    "acc_number": [24, 42, 90, 15],
    "bank_details": [0, 0, 144, 15],
    "payee": [25, 18, 145, 9],
    "courtesy_amount": [143, 31, 54, 13],
    "legal_amount_line_1": [29, 25, 152, 9.5],
    "legal_amount_line_2": [17, 35, 115, 9.5],
    "date_box": [144, 3, 56, 15],
    "micr_strip": [25, 78, 152, 13],
    "sign_area": [148, 55, 48, 25]
  }
}
```

### Credits
- https://github.com/victordibia/signver
- <a href="https://www.freepik.com/free-photo/hand-using-laptop-computer-with-virtual-screen-document-online-approve-paperless-quality-assurance-erp-management-concept_24755711.htm#query=document%20scanner&position=37&from_view=search&track=sph#position=37&query=document%20scanner">Image by DilokaStudio</a> on Freepik
