import os
import cv2
import json
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from stereovision.calibration import StereoCalibration

# Frame parameters
frame_height = 800
frame_width = 1264
frame_path = './data/calibration/pairs'

calibration_params = './src/calibration_params'
checkpoint_file = './src/calibration_params/3dmap_set.txt'

# Depth map function
PFC = 29
MDS = -25
NOD = 128
TTH = 100
UR = 10
SR = 15
SPWS = 100
SIGMA = 1.5
LMBDA = 8000.0
loading_settings = 0

def stereo_depth_map(frame_r, frame_l):
    # slightly blur the image and downsample it to half
    stereo_r = cv2.pyrDown(cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY))
    stereo_l = cv2.pyrDown(cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY))

    # create disparities
    left_matcher = cv2.StereoBM_create()

    # left_matcher.setPreFilterType(1)
    left_matcher.setPreFilterCap(PFC)
    left_matcher.setMinDisparity(MDS)
    left_matcher.setNumDisparities(NOD)
    left_matcher.setTextureThreshold(TTH)
    left_matcher.setUniquenessRatio(UR)
    left_matcher.setSpeckleRange(SR)
    left_matcher.setSpeckleWindowSize(SPWS)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # create disparities
    left_disp = left_matcher.compute(stereo_l, stereo_r)
    right_disp = right_matcher.compute(stereo_r, stereo_l)

    # now create DisparityWLSFilter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(LMBDA)
    wls_filter.setSigmaColor(SIGMA)
    disparity = wls_filter.filter(left_disp, stereo_l, disparity_map_right=right_disp)

    local_max = disparity.max()
    local_min = disparity.min()

    disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
    return disparity_visual

def update(val):
    global PFC, MDS, NOD, TTH, UR, SR, SPWS, SIGMA, LMBDA, loading_settings
    PFC   = int(sPFC.val/2)*2+1 #convert to ODD
    MDS   = int(sMDS.val)    
    NOD   = int(sNOD.val/16)*16  
    TTH   = int(sTTH.val)
    UR    = int(sUR.val)
    SR    = int(sSR.val)
    SPWS  = int(sSPWS.val)
    SIGMA = sSIGMA.val
    LMBDA = sLMBDA.val
    if (loading_settings==0):
        disparity = stereo_depth_map(frame_r, frame_l)
        dmObject.set_data(disparity)
        plt.draw()

def save_map_settings(event):
    savebtn.label.set_text ("Saving...")
    result = json.dumps({'lambda': LMBDA, 'sigma': SIGMA, 'preFilterCap':PFC, \
             'minDisparity':MDS, 'numberOfDisparities':NOD, 'textureThreshold':TTH, \
             'uniquenessRatio':UR, 'speckleRange':SR, 'speckleWindowSize':SPWS},\
             sort_keys=True, indent=4, separators=(',',':'))
    with open(checkpoint_file, 'w') as f:
        f.write(result)
    savebtn.label.set_text ("Save to file")

def load_map_settings(event):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    if not os.path.isfile(checkpoint_file):
        return

    loading_settings = 1
    loadbtn.label.set_text ("Loading...")
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    sPFC.set_val(data['preFilterCap'])
    sMDS.set_val(data['minDisparity'])
    sNOD.set_val(data['numberOfDisparities'])
    sTTH.set_val(data['textureThreshold'])
    sUR.set_val(data['uniquenessRatio'])
    sSR.set_val(data['speckleRange'])
    sSPWS.set_val(data['speckleWindowSize'])
    sSIGMA.set_val(data['sigma'])
    sLMBDA.set_val(data['lambda'])

    loadbtn.label.set_text ("Load settings")
    loading_settings = 0
    update(0)

if __name__ == '__main__':
    frame_r_path = str(next(Path(frame_path).glob('right_*.png')))
    frame_l_path = frame_r_path.replace('right', 'left')

    frame_r = cv2.imread(frame_r_path)
    frame_l = cv2.imread(frame_l_path)

    calibration = StereoCalibration(input_folder=calibration_params)
    frame_r, frame_l = calibration.rectify((frame_r, frame_l))

    disparity = stereo_depth_map(frame_r, frame_l)

    fig = plt.subplots(1,2)
    plt.subplots_adjust(left=0.15, bottom=0.5)

    plt.subplot(1,2,1)
    dmObject = plt.imshow(np.concatenate([frame_r, frame_l], axis=0), 'gray')

    plt.subplot(1,2,2)
    dmObject = plt.imshow(disparity, aspect='equal', cmap='jet')

    axcolor = 'lightgoldenrodyellow'

    SIGMAaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor)
    LMBDAaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor)
    PFCaxe   = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor)
    MDSaxe   = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor)
    NODaxe   = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor)
    TTHaxe   = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor)
    URaxe    = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor)
    SRaxe    = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor)
    SPWSaxe  = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor)

    sSIGMA = Slider(SIGMAaxe, 'SIGMA',         0,      5, valinit=SIGMA)
    sLMBDA = Slider(LMBDAaxe, 'LMBDA',      5000,  20000, valinit=LMBDA)
    sPFC   = Slider(PFCaxe,   'PreFiltCap',  5.0,   63.0, valinit=PFC)
    sMDS   = Slider(MDSaxe,   'MinDISP',  -400.0,  100.0, valinit=MDS)
    sNOD   = Slider(NODaxe,   'NumOfDisp',  16.0,  256.0, valinit=NOD)
    sTTH   = Slider(TTHaxe,   'TxtrThrshld', 0.0, 1000.0, valinit=TTH)
    sUR    = Slider(URaxe,    'UnicRatio',   1.0,   20.0, valinit=UR)
    sSR    = Slider(SRaxe,    'SpcklRng',    0.0,   40.0, valinit=SR)
    sSPWS  = Slider(SPWSaxe,  'SpklWinSze',  0.0,  300.0, valinit=SPWS)

    sSIGMA.on_changed(update)
    sLMBDA.on_changed(update)
    sPFC.on_changed(update)
    sMDS.on_changed(update)
    sNOD.on_changed(update)
    sTTH.on_changed(update)
    sUR.on_changed(update)
    sSR.on_changed(update)
    sSPWS.on_changed(update)

    saveax  = plt.axes([0.3,  0.38, 0.15, 0.04]) #stepX stepY width height
    savebtn = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')
    savebtn.on_clicked(save_map_settings)

    loadax  = plt.axes([0.55, 0.38, 0.15, 0.04]) #stepX stepY width height
    loadbtn = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')
    loadbtn.on_clicked(load_map_settings)

    load_map_settings(None)
    plt.show()