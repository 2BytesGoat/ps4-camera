import cv2
import json
import numpy as np
import os
import time
import subprocess

from stereovision.calibration import StereoCalibration

FRAME_INFO = { # move these to config file
    cv2.CAP_PROP_FRAME_WIDTH: 3448,
    cv2.CAP_PROP_FRAME_HEIGHT: 808
}

class PS4DataSource():
    def __init__(self, camera_idx=0, frame_width=1264, frame_height=800,
            calibrate_camera=True, calibration_params='./src/data_source/calibration_params'):
        self.camera_idx   = camera_idx
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.calibrate_camera = calibrate_camera
        self.calibration_params = calibration_params
        self._skip_brightness_calibration = False
        self._load_camera_firmware()
        self._open_capture_source()
        self._adapt_brightness()
        self._load_calibration_params()

    def _load_camera_firmware(self):
        _cwd = os.getcwd()
        os.chdir('.\dependencies\PS4-CAMERA-DRIVERS-master')
        proc = subprocess.Popen('OrbisEyeCameraFirmwareLoader.exe', stdout=subprocess.PIPE)
        status = str(proc.stdout.read()).strip() # check if camera firmware was previously loaded
        # self._skip_brightness_calibration = 'Usb Boot device not found...' in status
        os.chdir(_cwd)

    def _open_capture_source(self):
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        for key, value in FRAME_INFO.items():
            self.cap.set(key,value)

    def _adapt_brightness(self):
        if self._skip_brightness_calibration:
            return
        self._adapt_brightness_using_windows()  

    def _adapt_brightness_using_windows(self):
        # Using windows camera predefined camera init functionality
        # Warining! There will be exposure difference between times of day
        subprocess.run('start microsoft.windows.camera:', shell=True)
        time.sleep(4) # wait for camera brightness to calibrate
        subprocess.run('Taskkill /IM WindowsCamera.exe /F', shell=True)
        time.sleep(1)

    def _adapt_brightness_using_config(self):
        # TODO: find magic numbers for brightness configuration
        # Step1: load ps4_config.yaml
        # Step2: self.cap.set(key, value)
        pass

    def _load_depth_calibration_params(self):
        param_path = os.path.join(self.calibration_params, '3dmap_set.txt')
        if not os.path.isfile(param_path):
            return None, None, None

        with open(param_path, 'r') as f:
            param_dict = json.load(f)
        
        # define algorithm for calculating disparities
        left_matcher = cv2.StereoBM_create()

        # load params for computing disparity
        left_matcher.setPreFilterCap(param_dict['preFilterCap'])
        left_matcher.setMinDisparity(param_dict['minDisparity'])
        left_matcher.setNumDisparities(param_dict['numberOfDisparities'])
        left_matcher.setTextureThreshold(param_dict['textureThreshold'])
        left_matcher.setUniquenessRatio(param_dict['uniquenessRatio'])
        left_matcher.setSpeckleRange(param_dict['speckleRange'])
        left_matcher.setSpeckleWindowSize(param_dict['speckleWindowSize'])

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # load params for smoothing disparity
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(param_dict['lambda'])
        wls_filter.setSigmaColor(param_dict['sigma'])
        return left_matcher, right_matcher, wls_filter

    def _load_calibration_params(self):
        if os.path.isdir(self.calibration_params) and self.calibrate_camera:
            self.frame_calibration = StereoCalibration(input_folder=self.calibration_params)
            self.left_matcher, self.right_matcher, self.wls_filter = self._load_depth_calibration_params()
            self.use_disparity = True
        else:
            print('Could not load calibration params')
            self.calibrate_camera  = False
            self.frame_calibration = None
            self.left_matcher  = None
            self.right_matcher = None
            self.wls_filter    = None
        if self.left_matcher is None or self.right_matcher is None or self.wls_filter is None:
            self.use_disparity = False

    def _extract_stereo(self, frame, x_shift=64, y_shift=0, frame_shape=None):
        frame_r = frame[y_shift : y_shift+self.frame_height,
                        x_shift : x_shift+self.frame_width]
        frame_l = frame[y_shift : y_shift+self.frame_height, 
                        x_shift+self.frame_width : x_shift+self.frame_width*2]
        if frame_shape:
            frame_r = cv2.resize(frame_r, frame_shape)
            frame_l = cv2.resize(frame_l, frame_shape)
        return frame_r, frame_l

    def get_frame_shape(self):
        return (self.frame_width, self.frame_height)
        
    def calculate_disparity(self, frame_r, frame_l):
        if not self.use_disparity:
            return None

        # slightly blur the image and downsample it to half
        stereo_r = cv2.pyrDown(cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY))
        stereo_l = cv2.pyrDown(cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY))

        # create disparities
        left_disp  = self.left_matcher.compute(stereo_l, stereo_r)
        right_disp = self.right_matcher.compute(stereo_r, stereo_l)

        # filter disparities
        disparity = self.wls_filter.filter(left_disp, stereo_l, disparity_map_right=right_disp)

        local_max = disparity.max()
        local_min = disparity.min()

        disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
        return disparity_visual

    def stream(self):
        while True:
            # capture frame-by-frame
            ret, frame = self.cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_r, frame_l = self._extract_stereo(frame)

            if self.calibrate_camera:
                frame_r, frame_l = self.frame_calibration.rectify((frame_r, frame_l))

            yield frame_r, frame_l
        return None, None

    def close_stream(self):
        self.cap.release()