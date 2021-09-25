import cv2
import numpy as np
import os
import yaml

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from stereovision.calibration import StereoCalibration

class CalibrationUI():
    def __init__(self, frame_r, frame_l, disp_algo='bm', config_path='src/calibration/configs'):
        self.frame_r = frame_r
        self.frame_l = frame_l
        self.disp_algo = disp_algo
        self.config_path = config_path

        self.loading_settings = False

        self.__init_display()
        self.__init_slider_info()
        self.__init_sliders()
        self.__slider_update()
        plt.show()

    def __init_display(self, cmap='plasma'):
        placeholder = np.random.uniform(0, 1, (600, 800))

        plt.subplots(1,2)
        plt.subplots_adjust(left=0.15, bottom=0.5)

        plt.subplot(1,2,1)
        self.imgs_plot = plt.imshow(np.concatenate([self.frame_r, self.frame_l], axis=0), 'gray')

        plt.subplot(1,2,2)
        self.disp_plot = plt.imshow(placeholder, cmap=cmap)

        plt.colorbar()

    def __init_slider_info(self, _event=None):
        if self.disp_algo == 'bm':
            self.slider_info_yaml = os.path.join(self.config_path, 'stereoBM_sliders.yaml')
            self.calculate_disparity = self.BM_disparity
        elif self.disp_algo == 'sgbm':
            self.slider_info_yaml = os.path.join(self.config_path, 'stereoSGBM_sliders.yaml')
            self.calculate_disparity = self.SGBM_disparity

        self.loading_settings = True

        with open(self.slider_info_yaml, 'r') as f:
            self.slider_info = yaml.safe_load(f)
        self.slider_info_keys = sorted(list(self.slider_info.keys()))

        self.loading_settings = False

    def __init_sliders(self, axcolor='lightgoldenrodyellow'):
        self.sliders = {}
        self.buttons = {}
        self.curr_slider_val = {}
        
        # we got thorugh the ordered list of slider keys just in case
        # information in the yaml file is not sorted
        for i, key in enumerate(self.slider_info_keys):
            shift = (i * 4 + 1) / 100
            slider_axe = (plt.axes([0.15, shift, 0.7, 0.025], facecolor=axcolor))
            slider = Slider(slider_axe, 
                            key, 
                            self.slider_info[key]['min'], 
                            self.slider_info[key]['max'],
                            valstep=self.slider_info[key]['valstep'],
                            valinit=self.slider_info[key]['default'])
            slider.on_changed(self.__slider_update)
            self.sliders[key] = slider
            self.curr_slider_val[key] = self.slider_info[key]['default']

        saveax  = plt.axes([0.3,  0.38, 0.15, 0.04]) #stepX stepY width height
        savebtn = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')
        savebtn.on_clicked(self.__save_slider_info)
        self.buttons['save'] = savebtn # create a ref such that it won't go in GC

        loadax  = plt.axes([0.55, 0.38, 0.15, 0.04]) #stepX stepY width height
        loadbtn = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')
        loadbtn.on_clicked(self.__load_slider_info)
        self.buttons['load'] = loadbtn # create a ref such that it won't go in GC

    def __slider_update(self, _val=None):
        for key, curr_slider in self.sliders.items():
            self.curr_slider_val[key] = int(curr_slider.val)
            
        if not self.loading_settings:
            disparity = self.calculate_disparity()
            self.disp_plot.set_data(disparity)
            plt.draw()

    def __save_slider_info(self, _event=None):
        self.buttons['save'].label.set_text("Saving...")

        for key in self.slider_info:
            self.slider_info[key]['default'] = self.curr_slider_val[key]

        with open(self.slider_info_yaml, 'w') as f:
            yaml.dump(self.slider_info, f)

        self.buttons['save'].label.set_text("Save to file")

    def __load_slider_info(self, _event=None):
        self.loading_settings = True
        self.buttons['load'].label.set_text("Loading...")

        with open(self.slider_info_yaml, 'r') as f:
            self.slider_info = yaml.safe_load(f)
        
        for key, items in self.slider_info.items():
            self.sliders[key].set_val(items['default'])

        self.loading_settings = False
        self.buttons['load'].label.set_text("Load settings")

        self.__slider_update()

    def __create_bm_matcher(self):
        stereo_matcher = cv2.StereoBM_create()

        stereo_matcher.setMinDisparity(self.curr_slider_val['MinDISP'])
        stereo_matcher.setNumDisparities(self.curr_slider_val['NumOfDisp'])
        stereo_matcher.setPreFilterCap(self.curr_slider_val['PreFiltCap'])
        stereo_matcher.setSpeckleRange(self.curr_slider_val['SpcklRng'])
        stereo_matcher.setSpeckleWindowSize(self.curr_slider_val['SpklWinSze'])
        stereo_matcher.setTextureThreshold(self.curr_slider_val['TxtrThrshld'])
        stereo_matcher.setUniquenessRatio(self.curr_slider_val['UnicRatio'])
        
        return stereo_matcher

    def __create_sgbm_matcher(self):
        stereo_matcher = cv2.StereoSGBM_create()

        # NumOfDisp = self.curr_slider_val['MaxDISP'] - self.curr_slider_val['MinDISP']

        stereo_matcher.setMinDisparity(self.curr_slider_val['MinDISP'])
        stereo_matcher.setNumDisparities(self.curr_slider_val['NumOfDisp'])
        stereo_matcher.setSpeckleRange(self.curr_slider_val['SpcklRng'])
        stereo_matcher.setSpeckleWindowSize(self.curr_slider_val['SpklWinSze'])
        stereo_matcher.setUniquenessRatio(self.curr_slider_val['UnicRatio'])

        stereo_matcher.setBlockSize(self.curr_slider_val['BlockSize'])
        stereo_matcher.setP1(8 * (self.curr_slider_val['BlockSize'] ** 2))
        stereo_matcher.setP2(32 * (self.curr_slider_val['BlockSize'] ** 2))
        stereo_matcher.setDisp12MaxDiff(self.curr_slider_val['Disp12MaxDiff'])

        return stereo_matcher

    def __compute_smooth_disparity(self, left_matcher):
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # create disparities
        left_disp = left_matcher.compute(self.frame_l, self.frame_r)
        right_disp = right_matcher.compute(self.frame_r, self.frame_l)

        # now create DisparityWLSFilter
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(self.curr_slider_val['LMBDA'])
        wls_filter.setSigmaColor(self.curr_slider_val['SIGMA'])
        disparity = wls_filter.filter(left_disp, self.frame_l, disparity_map_right=right_disp)
        return disparity

    def __normalize_disparity(self, disparity):
        # Normalize the values to a range from 0..255 for a grayscale image
        local_max = disparity.max()
        local_min = disparity.min()

        disparity = (disparity-local_min)*(1.0/(local_max-local_min))
        return disparity
    
    def BM_disparity(self, smooth=True):
        matcher = self.__create_bm_matcher()

        if smooth:
            disparity_BM = self.__compute_smooth_disparity(matcher)
        else:
            disparity_BM = matcher.compute(self.frame_l, self.frame_r)

        disparity_BM = self.__normalize_disparity(disparity_BM)
        return disparity_BM

    def SGBM_disparity(self, smooth=True):
        matcher = self.__create_sgbm_matcher()

        if smooth:
            disparity_SGBM = self.__compute_smooth_disparity(matcher)
        else:
            disparity_SGBM = matcher.compute(self.frame_l, self.frame_r)

        disparity_SGBM = self.__normalize_disparity(disparity_SGBM)
        return disparity_SGBM


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

    ui = CalibrationUI(frame_r, frame_l, 'sgbm')
    # ui = CalibrationUI(frame_r, frame_l, 'bm')

