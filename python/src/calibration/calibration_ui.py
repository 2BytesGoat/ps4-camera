import numpy as np
import os
import yaml

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

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
        plt.show()

    def __init_display(self, cmap='plasma'):
        placeholder = np.zeros((600, 800))

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

if __name__ == '__main__':
    placeholder = np.zeros((600, 800))
    ui = CalibrationUI(placeholder, placeholder)
