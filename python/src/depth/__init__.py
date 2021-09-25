from .stereo_depth import *

def get_stereo_depth_algo(algo_type):
    if algo_type == 'bm':
        return BMDisparity()
    else:
        return SGBMDisparity()