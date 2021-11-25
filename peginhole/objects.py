import numpy as np
from pathlib import Path
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements

asset_path = lambda obj_id, obj_type: "peginhole/assets/"+str(obj_id)+"/"+obj_type+".xml"

class Hole(MujocoXMLObject):
    def __init__(self, name, peg_class):
        super().__init__(asset_path(peg_class, "hole"), name=name, joints=None)
        
class Peg(MujocoXMLObject):
    def __init__(self, name, peg_class, joints=None):
        super().__init__(asset_path(peg_class, "peg"), name=name, joints=joints)