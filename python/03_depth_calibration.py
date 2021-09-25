import cv2

from pathlib import Path
from stereovision.calibration import StereoCalibration

from src.calibration.depth_calibration_ui import DepthCalibrationUI

if __name__ == '__main__':
    frame_path = './data/calibration/pairs'
    calibration_params = './src/data_source/calibration_params'

    frame_r_path = str(next(Path(frame_path).glob('right_*.png')))
    frame_l_path = frame_r_path.replace('right', 'left')

    frame_r = cv2.imread(frame_r_path)
    frame_l = cv2.imread(frame_l_path)

    calibration = StereoCalibration(input_folder=calibration_params)
    frame_r, frame_l = calibration.rectify((frame_r, frame_l))

    frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)

    # slightly blur the image and downsample it to half
    frame_r = cv2.pyrDown(frame_r)
    frame_l = cv2.pyrDown(frame_l)

    depth_calibrator = DepthCalibrationUI(frame_r, frame_l, 'sgbm')