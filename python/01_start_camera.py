import cv2
import numpy as np

from src.data_source.ps4_data_source import PS4DataSource

if __name__ == '__main__':
    data_source = PS4DataSource()
    for frame_r, frame_l in data_source.stream():
        if frame_r is None or frame_l is None:
            break

        disparity = data_source.calculate_disparity(frame_r, frame_l)

        cv2.imshow('stereo', np.concatenate([frame_r, frame_l], axis=1))

        if not disparity is None:
            cv2.imshow('disparity', disparity)

        if cv2.waitKey(1) == ord('q'):
                break

    data_source.close_stream()