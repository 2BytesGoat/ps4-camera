import subprocess
import time
import os
import cv2
import numpy as np

class PS4DataSource():
    def __init__(self, camera_idx=0):
        self.camera_idx = camera_idx
        self._skip_brightness_calibration = False
        self._load_camera_firmware()
        self._adapt_camera_brightess()

    def _load_camera_firmware(self):
        _cwd = os.getcwd()
        os.chdir('.\dependencies\PS4-CAMERA-DRIVERS-master')
        proc = subprocess.Popen('OrbisEyeCameraFirmwareLoader.exe', stdout=subprocess.PIPE)
        status = str(proc.stdout.read()).strip() # check if camera firmware was previously loaded
        self._skip_brightness_calibration = 'Usb Boot device not found...' in status
        os.chdir(_cwd)

    def _adapt_camera_brightess(self):
        # Using windows camera predefined camera init functionality
        # TODO: try to replace these with OpenCV functionalities
        if self._skip_brightness_calibration:
            return # if firmware was previously loaded we skip the calibration
        subprocess.run('start microsoft.windows.camera:', shell=True)
        time.sleep(4) # wait for camera brightness to calibrate
        subprocess.run('Taskkill /IM WindowsCamera.exe /F', shell=True)

    def _extract_stereo(self, frame, x_shift=48, y_shift=0, width=320, height=192):
        frame_l = frame[y_shift:y_shift+height,
                        x_shift:x_shift + width]
        frame_r = frame[y_shift:y_shift+height, 
                        x_shift + width:x_shift + width*2]
        return frame_l, frame_r
        
    def calculate_disparity(self, frame_l, frame_r):
        # https://docs.opencv.org/4.5.2/dd/d53/tutorial_py_depthmap.html
        # https://docs.opencv.org/4.5.2/d3/d14/tutorial_ximgproc_disparity_filtering.html <- improve disparity
        cv2.resize(frame_l, (600, 800))
        cv2.resize(frame_r, (600, 800))
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
        disparity = stereo.compute(frame_r, frame_l)
        return disparity

    def start(self):
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_l, frame_r = self._extract_stereo(gray)
            disparity = self.calculate_disparity(frame_l, frame_r)

            # Display the resulting frame
            cv2.imshow('stereo', np.concatenate([frame_l, frame_r]))
            cv2.imshow('disparity', disparity)
            if cv2.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = PS4DataSource()
    camera.start()