import cv2
import numpy as np
import os
import time

from pathlib import Path
from stereovision.calibration import StereoCalibrator, StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

from src.data_source.ps4_data_source import PS4DataSource

""" ! IMPORTANT !

The checkerboard pattern I used is located in ./data/calibration/pattern.png
"""

class PS4Calibratior():
    def __init__(self, camera_index=0, total_photos=30, cnt_interval=5):
        self.display_font = cv2.FONT_HERSHEY_SIMPLEX # Cowntdown timer font

        self.camera_index = camera_index # Which camera to use for calibration
        self.total_photos = total_photos # Number of images to take
        self.cnt_interval = cnt_interval # Interval for count-down timer, seconds

        self.counter = 0 # How many frames were captured so far

        self.data_source = PS4DataSource(self.camera_index) # Create capture source

    def capture_data(self, dst_folder='./data/calibration/pairs'):
        os.makedirs(dst_folder, exist_ok=True)

        # Record start time
        start = time.time()
        for frame_r, frame_l in self.data_source.stream():
            if frame_r is None or frame_l is None:
                break

            # How much time has passed from start
            cntdwn_timer = int(time.time() - start)

            # If cowntdown is zero - let's record next image
            if cntdwn_timer >= self.cnt_interval:            
                self.counter += 1

                # Save the frames
                frame_name = f'{dst_folder}/right_{self.counter}.png'
                cv2.imwrite(frame_name, frame_r)
                
                frame_name = f'{dst_folder}/left_{self.counter}.png'
                cv2.imwrite(frame_name, frame_l)

                # Display text and wait for person to asimilate
                print (f'Frame captured [{self.counter} of {self.total_photos}] ')
                time.sleep(1)

                # Record new start time
                start = time.time()

            # Stop recording if all images were recorded
            if self.counter >= self.total_photos:
                break

            # Resize images to fit on the screen
            frame_r = cv2.resize(frame_r, (640, 480))
            frame_l = cv2.resize(frame_l, (640, 480))

            # Draw cowntdown counter, seconds
            cv2.putText(frame_r, str(cntdwn_timer), (50,50), self.display_font, 2.0, (0,0,255),4, cv2.LINE_AA)
            cv2.imshow("Frame Right", np.concatenate([frame_r, frame_l], axis=1))
            key = cv2.waitKey(1) & 0xFF

            # Press 'Q' key to quit, or wait till all photos are taken
            if cv2.waitKey(1) == ord('q'):
                    break

        self.data_source.close_stream()

    def _draw_line(self, img, start, end, color=(0, 0, 255)):
        cv2.line(img, start, end, color, 2, 8)

    def _display_calibration_lines(self, frame_pair, color=(0, 0, 255)):
        result = np.concatenate(frame_pair, axis=1)
        rect_height, rect_width, _ = result.shape

        # Draw lines to visualize calibration
        for y in range(50, rect_height, 50):
            self._draw_line(result, start=[0, y], end=[rect_width, y], color=color)

        return result

    def calculate_calibration_params(self, frame_path='./data/calibration/pairs', 
            calib_rows=6, calib_columns=9, calib_square_size=2.5):

        frame_width, frame_height = self.data_source.get_frame_shape()
        calibrator = StereoCalibrator(calib_rows, calib_columns, calib_square_size, (frame_width, frame_height))

        for frame_r_path in Path(frame_path).glob('right_*.png'):
            frame_idx = frame_r_path.stem.split('_')[-1]

            frame_r_path = str(frame_r_path)
            frame_l_path = frame_r_path.replace('right', 'left')

            frame_r = cv2.imread(frame_r_path,1)
            frame_l = cv2.imread(frame_l_path,1)

            try:
                calibrator._get_corners(frame_r)
                calibrator._get_corners(frame_l)
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

        result = self._display_calibration_lines([frame_r, frame_l])
        cv2.imshow('Un-rectified Images', result)
        cv2.waitKey(0)

        result = self._display_calibration_lines(rectified_pair, color=(0, 255, 0))
        cv2.imshow('Rectified Images', result)
        cv2.imwrite(f'./data/calibration/rectified_images.jpg', result)
        cv2.waitKey(0)
    
