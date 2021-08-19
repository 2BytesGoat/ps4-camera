import cv2
import numpy as np

from pathlib import Path
from numpy.core.fromnumeric import resize

from stereovision.calibration import StereoCalibrator, StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

# Chessboard parameters
rows = 6
columns = 9
square_size = 2.1 # may vary based on your print

# Frame parameters
frame_height = 800
frame_width = 1264
rect_rame_height = 600
rect_frame_width = 1600
frame_path = './data/calibration/pairs'

def draw_line(img, start, end, color=(0, 0, 255)):
    thickness = 2
    line_type = 8
    cv2.line(img, start, end,
             color, thickness, line_type)

if __name__ == '__main__':
    calibrator = StereoCalibrator(rows, columns, square_size, (frame_width, frame_height))
    for frame_l_path in Path(frame_path).glob('left_*.png'):
        frame_idx = frame_l_path.stem.split('_')[-1]
        
        frame_l_path = str(frame_l_path)
        frame_r_path = frame_l_path.replace('left', 'right')

        frame_l = cv2.imread(frame_l_path,1)
        frame_r = cv2.imread(frame_r_path,1)

        try:
            calibrator._get_corners(frame_l)
            calibrator._get_corners(frame_r)
        except ChessboardNotFoundError as error:
            print (error)
            print ("Pair No "+ str(frame_idx) + " ignored")
        else:
            calibrator.add_corners((frame_r, frame_l), True)

    print ('End cycle')

    print ('Starting calibration... It can take several minutes!')
    calibration = calibrator.calibrate_cameras()
    calibration.export('calibration_params')
    print ('Calibration complete!')

    # Lets rectify and show last pair after  calibration
    calibration = StereoCalibration(input_folder='calibration_params')
    rectified_pair = calibration.rectify((frame_r, frame_l))

    result = np.concatenate(rectified_pair, axis=1)
    result = cv2.resize(result, (rect_frame_width, rect_rame_height))

    # Draw lines to visualize calibration
    for y in range(50, rect_rame_height, 50):
        draw_line(result, start=[0, y], end=[rect_frame_width, y], color=(0, 0, 255))

    cv2.imshow('Rectified Images', result)
    cv2.imwrite(f'./data/calibration/rectified_images.jpg', result)
    cv2.waitKey(0)