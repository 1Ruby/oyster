import numpy as np
from pathlib import Path
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements

asset_path = lambda obj_id, obj_type: "./assets/"+str(obj_id)+"/"+obj_type+".xml"

class Hole0(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(asset_path(0, "hole"), name=name, joints=None)
        
class Peg0(MujocoXMLObject):
    def __init__(self, name, joints=None):
        super().__init__(asset_path(0, "peg"), name=name, joints=joints)


class Hole1(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(asset_path(1, "hole"), name=name, joints=None)


class Peg1(MujocoXMLObject):
    def __init__(self, name, joints=None):
        super().__init__(asset_path(1, "peg"), name=name, joints=joints)
        
class Hole2(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(asset_path(2, "hole"), name=name, joints=None)


class Peg2(MujocoXMLObject):
    def __init__(self, name, joints=None):
        super().__init__(asset_path(2, "peg"), name=name, joints=joints)