from . import constants

def set_config(**kwargs):
    constants._set_config(**kwargs)

def __getattr__(name):
    return getattr(constants, name)