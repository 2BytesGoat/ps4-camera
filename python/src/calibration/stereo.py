import cv2
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path
from stereovision.calibration import StereoCalibration

frame_path = './data/calibration/pairs'
calibration_params = './src/data_source/calibration_params'

def stereo_depth_map(frame_r, frame_l):
    block_size = 11
    min_disp = -128
    max_disp = 128
    uniquenessRatio = 5
    speckleWindowSize = 200
    speckleRange = 2
    disp12MaxDiff = 0

    # slightly blur the image and downsample it to half
    stereo_r = cv2.pyrDown(cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY))
    stereo_l = cv2.pyrDown(cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY))

    num_disp = max_disp - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )

    disparity_SGBM = stereo.compute(stereo_r, stereo_l)

    # Normalize the values to a range from 0..255 for a grayscale image
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)
    return disparity_SGBM

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
    dmObject = plt.imshow(disparity, cmap='plasma')
    plt.colorbar()
    plt.show()
