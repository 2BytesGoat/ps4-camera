from .stereo_depth import *

def get_stereo_depth_algo(algo_type, smoothen):
    if algo_type == 'bm':
        return BMDisparity(smoothen=smoothen)
    else:
        return SGBMDisparity(smoothen=smoothen)