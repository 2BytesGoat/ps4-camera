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

# Depth map function
SWS = 5
PFS = 5
PFC = 29
MDS = -25
NOD = 128
TTH = 100
UR = 10
SR = 15
SPWS = 100
loading_settings = 0

def stereo_depth_map(rectified_pair):
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    #sbm.SADWindowSize = SWS
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)

    disparity = sbm.compute(*rectified_pair)

    local_max = disparity.max()
    local_min = disparity.min()

    disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
    return disparity_visual

def update(val):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    SWS = int(sSWS.val/2)*2+1 #convert to ODD
    PFS = int(sPFS.val/2)*2+1
    PFC = int(sPFC.val/2)*2+1    
    MDS = int(sMDS.val)    
    NOD = int(sNOD.val/16)*16  
    TTH = int(sTTH.val)
    UR = int(sUR.val)
    SR = int(sSR.val)
    SPWS= int(sSPWS.val)
    if (loading_settings==0):
        disparity = stereo_depth_map(rectified_pair)
        dmObject.set_data(disparity)
        plt.draw()

if __name__ == '__main__':
    frame_r_path = str(next(Path(frame_path).glob('right_*.png')))
    frame_l_path = frame_r_path.replace('right', 'left')

    frame_r = cv2.imread(frame_r_path, 0)
    frame_l = cv2.imread(frame_l_path, 0)

    calibration = StereoCalibration(input_folder='calibration_params')
    rectified_pair = calibration.rectify((frame_r, frame_l))

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo_depth_map(rectified_pair)

    fig = plt.subplots(1,2)
    plt.subplots_adjust(left=0.15, bottom=0.5)

    plt.subplot(1,2,1)
    dmObject = plt.imshow(np.concatenate(rectified_pair,axis=0), 'gray')

    plt.subplot(1,2,2)
    dmObject = plt.imshow(disparity, aspect='equal', cmap='jet')

    axcolor = 'lightgoldenrodyellow'

    SWSaxe  = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor)
    PFSaxe  = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor)
    PFCaxe  = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor)
    MDSaxe  = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor)
    NODaxe  = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor)
    TTHaxe  = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor)
    URaxe   = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor)
    SRaxe   = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor)
    SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor)

    sSWS  = Slider(SWSaxe,  'SWS',         5.0,  255.0, valinit=5)
    sPFS  = Slider(PFSaxe,  'PFS',         5.0,  255.0, valinit=5)
    sPFC  = Slider(PFCaxe,  'PreFiltCap',  5.0,   63.0, valinit=29)
    sMDS  = Slider(MDSaxe,  'MinDISP',  -400.0,  100.0, valinit=-25)
    sNOD  = Slider(NODaxe,  'NumOfDisp',  16.0,  256.0, valinit=128)
    sTTH  = Slider(TTHaxe,  'TxtrThrshld', 0.0, 1000.0, valinit=100)
    sUR   = Slider(URaxe,   'UnicRatio',   1.0,   20.0, valinit=10)
    sSR   = Slider(SRaxe,   'SpcklRng',    0.0,   40.0, valinit=15)
    sSPWS = Slider(SPWSaxe, 'SpklWinSze',  0.0,  300.0, valinit=100)

    sSWS.on_changed(update)
    sPFS.on_changed(update)
    sPFC.on_changed(update)
    sMDS.on_changed(update)
    sNOD.on_changed(update)
    sTTH.on_changed(update)
    sUR.on_changed(update)
    sSR.on_changed(update)
    sSPWS.on_changed(update)

    plt.show()