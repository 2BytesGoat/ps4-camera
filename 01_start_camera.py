import cv2
import numpy as np

from src.data_source.ps4_data_source import PS4DataSource
from src.depth import get_stereo_depth_algo

if __name__ == '__main__':
    data_source = PS4DataSource()
    depth_algo = get_stereo_depth_algo('bm', smoothen=True)
    # depth_algo = get_stereo_depth_algo('sgbm', smoothen=True)
    for frame_r, frame_l in data_source.stream(grayscale=True):
        if frame_r is None or frame_l is None:
            break

        disparity = depth_algo.compute_disparity(frame_l, frame_r)

        cv2.imshow('stereo', np.concatenate([frame_r, frame_l], axis=1))

        if not disparity is None:
            cv2.imshow('disparity', disparity)

        if cv2.waitKey(1) == ord('q'):
                break

    data_source.close_stream()