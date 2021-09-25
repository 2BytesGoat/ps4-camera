import cv2
import yaml

DEFAULT_BM_CONFIG = 'src/depth/configs/stereoBM.yaml'
DEFAULT_SGBM_CONFIG = 'src/depth/configs/stereoSGBM.yaml'

class AbstractDisparity():
    def __init__(self, config_path=None, smoothen=True):
        self.config_path = config_path
        self.smoothen = smoothen

        self.left_matcher = None
        self.right_matcher = None
        self.matcher_params = {}

    def _init_matcher_params(self):
        with open(self.config_path, 'r') as f:
            self.matcher_params = yaml.safe_load(f)

    def _init_matchers(self):
        raise NotImplementedError

    def _init_wls_filter(self):
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)

    def _compute_coarse_disparity(self, frame_l, frame_r):
        return self.left_matcher.compute(frame_l, frame_r)

    def _compute_smooth_disparity(self, frame_l, frame_r):
        left_disp = self.left_matcher.compute(frame_l, frame_r)
        right_disp = self.right_matcher.compute(frame_r, frame_l)
        disparity = self.wls_filter.filter(left_disp, frame_l, disparity_map_right=right_disp)
        return disparity

    def _normalize_disparity(self, disparity):
        # Normalize the values to a range from 0..255 for a grayscale image
        local_max = disparity.max()
        local_min = disparity.min()

        disparity = (disparity-local_min)*(1.0/(local_max-local_min))
        return disparity

    def _update_params(self):
        self.wls_filter.setLambda(self.matcher_params['LMBDA'])
        self.wls_filter.setSigmaColor(self.matcher_params['SIGMA'])
    
    def compute_disparity(self, frame_l, frame_r):
        if not self.smoothen:
            disparity = self._compute_coarse_disparity(frame_l, frame_r)
        else:
            disparity = self._compute_smooth_disparity(frame_l, frame_r)
        return self._normalize_disparity(disparity)

    def load_params(self, matcher_params):
        self.matcher_params = matcher_params
        self._update_params()

    def save_params(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.matcher_params, f)

    def get_params(self):
        return self.matcher_params


class BMDisparity(AbstractDisparity):
    def __init__(self, config_path=DEFAULT_BM_CONFIG, smoothen=True):
        super().__init__(config_path, smoothen)
        self._init_matcher_params()
        self._init_matchers()
        self._init_wls_filter()
        self._update_params()

    def _init_matchers(self):
        self.left_matcher = cv2.StereoBM_create()
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

    def _update_params(self):
        super()._update_params()
        self.left_matcher.setMinDisparity(self.matcher_params['MinDISP'])
        self.left_matcher.setNumDisparities(self.matcher_params['NumOfDisp'])
        self.left_matcher.setPreFilterCap(self.matcher_params['PreFiltCap'])
        self.left_matcher.setSpeckleRange(self.matcher_params['SpcklRng'])
        self.left_matcher.setSpeckleWindowSize(self.matcher_params['SpklWinSze'])
        self.left_matcher.setTextureThreshold(self.matcher_params['TxtrThrshld'])
        self.left_matcher.setUniquenessRatio(self.matcher_params['UnicRatio'])

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)


class SGBMDisparity(AbstractDisparity):
    def __init__(self, config_path=DEFAULT_SGBM_CONFIG, smoothen=True):
        super().__init__(config_path, smoothen)
        self._init_matcher_params()
        self._init_matchers()
        self._init_wls_filter()
        self._update_params()

    def _init_matchers(self):
        self.left_matcher = cv2.StereoSGBM_create()
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

    def _update_params(self):
        super()._update_params()
        self.left_matcher.setMinDisparity(self.matcher_params['MinDISP'])
        self.left_matcher.setNumDisparities(self.matcher_params['NumOfDisp'])
        self.left_matcher.setSpeckleRange(self.matcher_params['SpcklRng'])
        self.left_matcher.setSpeckleWindowSize(self.matcher_params['SpklWinSze'])
        self.left_matcher.setUniquenessRatio(self.matcher_params['UnicRatio'])

        self.left_matcher.setBlockSize(self.matcher_params['BlockSize'])
        self.left_matcher.setP1(8 * (self.matcher_params['BlockSize'] ** 2))
        self.left_matcher.setP2(32 * (self.matcher_params['BlockSize'] ** 2))
        self.left_matcher.setDisp12MaxDiff(self.matcher_params['Disp12MaxDiff'])

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
