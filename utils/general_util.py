import json 
from constants import *

def get_template(template_path=None, bank_name=None):
    name2path = {
        "axis": AXIS_TEMPLATE_PATH,
        "icici": ICICI_TEMPLATE_PATH,
        "syndicate": SYNDICATE_TEMPLATE_PATH
    }

    if template_path is not None:
        return json.loads(open(template_path, "r").read())
    elif bank_name is not None and bank_name in name2path.keys():
        return json.loads(open(name2path[bank_name], "r").read())
    else:
        return json.loads(open(DEFAULT_TEMPLATE_PATH, "r").read())
        
    