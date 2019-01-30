import os
import glob as glob

def mkdir(path):
    """
    params:
        path:the path need to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(path_list):
    """
    params:
        path_list:the path list need to create
    """
    for path in path_list:
        mkdir(path)
        