import subprocess
import time
import os
import cv2
import numpy as np

FRAME_INFO = {
    cv2.CAP_PROP_FRAME_WIDTH: 3448,
    cv2.CAP_PROP_FRAME_HEIGHT: 808
}

class PS4DataSource():
    def __init__(self, camera_idx=0):
        self.camera_idx = camera_idx
        self._skip_brightness_calibration = False
        self._load_camera_firmware()
        self._open_capture_source()
        self._adapt_brightness()
        self._adjust_frames()

    def _load_camera_firmware(self):
        _cwd = os.getcwd()
        os.chdir('.\dependencies\PS4-CAMERA-DRIVERS-master')
        proc = subprocess.Popen('OrbisEyeCameraFirmwareLoader.exe', stdout=subprocess.PIPE)
        status = str(proc.stdout.read()).strip() # check if camera firmware was previously loaded
        self._skip_brightness_calibration = 'Usb Boot device not found...' in status
        os.chdir(_cwd)

    def _open_capture_source(self):
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def _adapt_brightness(self):
        if self._skip_brightness_calibration:
            return
        self._adapt_brightness_using_windows()  

    def _adapt_brightness_using_windows(self):
        # Using windows camera predefined camera init functionality
        subprocess.run('start microsoft.windows.camera:', shell=True)
        time.sleep(4) # wait for camera brightness to calibrate
        subprocess.run('Taskkill /IM WindowsCamera.exe /F', shell=True)
        time.sleep(1)

    def _adapt_brightness_using_config(self):
        # TODO: find magic numbers for brightness configuration
        # for key, value in BRIGHTNESS_INFO.items():
        #     self.cap.set(key, value)
        pass

    def _adjust_frames(self):
        for key, value in FRAME_INFO.items():
            self.cap.set(key,value)

    def _extract_stereo(self, frame, x_shift=64, y_shift=0, width=1264, height=800, frame_shape=None):
        frame_l = frame[y_shift:y_shift+height,
                        x_shift:x_shift + width]
        frame_r = frame[y_shift:y_shift+height, 
                        x_shift + width:x_shift + width*2]
        if frame_shape:
            frame_l = cv2.resize(frame_l, frame_shape)
            frame_r = cv2.resize(frame_r, frame_shape)
        return frame_l, frame_r
        
    def calculate_disparity(self, frame_l, frame_r, minDisparity=10, maxDisparity=98, winSize=5):
        numDisparities = maxDisparity - minDisparity # Needs to be divisible by 16
        # stereo = cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities,
        #                                blockSize=5, P1=8*3*winSize**2, P2=32*3*winSize**2,
        #                                disp12MaxDiff=50, uniquenessRatio=10,
        #                                speckleWindowSize=128, speckleRange=10
        #                                )
        stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
        disparity = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0

        disparity_scaled = (disparity - minDisparity) / numDisparities
        disparity_scaled += abs(np.amin(disparity_scaled))
        disparity_scaled /= np.amax(disparity_scaled)
        disparity_scaled[disparity_scaled < 0] = 0
        return np.array(255 * disparity_scaled, np.uint8) 

    def start(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_l, frame_r = self._extract_stereo(gray)
            # disparity = self.calculate_disparity(frame_l, frame_r)

            # Display the resulting frame
            cv2.imshow('stereo', np.concatenate([frame_l, frame_r], axis=1))
            # cv2.imshow('disparity', disparity)
            if cv2.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = PS4DataSource()
    camera.start()