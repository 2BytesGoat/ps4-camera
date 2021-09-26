import numpy as np
import os
import yaml

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

from src.depth import get_stereo_depth_algo

class DepthCalibrationUI():
    def __init__(self, frame_r, frame_l, depth_algo_type='bm', smoothen_depth=True, config_path='src/calibration/configs'):
        self.frame_r = frame_r
        self.frame_l = frame_l
        self.depth_algo_type = depth_algo_type
        self.config_path = config_path

        self.depth_algo = get_stereo_depth_algo(depth_algo_type, smoothen=smoothen_depth)
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
        if self.depth_algo_type == 'bm':
            self.slider_info_yaml = os.path.join(self.config_path, 'stereoBM_sliders.yaml')
        elif self.depth_algo_type == 'sgbm':
            self.slider_info_yaml = os.path.join(self.config_path, 'stereoSGBM_sliders.yaml')

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

        self.depth_algo.load_params(self.curr_slider_val)
            
        if not self.loading_settings:
            disparity = self.depth_algo.compute_disparity(self.frame_l, self.frame_r)
            self.disp_plot.set_data(disparity)
            plt.draw()

    def __save_slider_info(self, _event=None):
        self.buttons['save'].label.set_text("Saving...")

        for key in self.slider_info:
            self.slider_info[key]['default'] = self.curr_slider_val[key]

        with open(self.slider_info_yaml, 'w') as f:
            yaml.dump(self.slider_info, f)

        self.depth_algo.save_params()

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
