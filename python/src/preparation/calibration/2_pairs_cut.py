import os
import cv2
from pathlib import Path

src_folder = './data/calibration/scenes'
dst_folder = './data/calibration/pairs'

if __name__ == '__main__':
    # Create desitnation path
    os.makedirs(dst_folder, exist_ok=True)

    for image_path in Path(src_folder).glob('*.png'):
        image_number = image_path.stem.split('_')[-1]
        image = cv2.imread(str(image_path))
        frame_height, frame_width, _ = image.shape
        frame_width = frame_width // 2

        frame_r = image[:, :frame_width]
        frame_l = image[:, frame_width:]

        frame_r_name = f'{dst_folder}/right_{image_number}.png'
        frame_l_name = f'{dst_folder}/left_{image_number}.png'

        cv2.imwrite(frame_r_name, frame_r)
        cv2.imwrite(frame_l_name, frame_l)

        print(f'Pair No {image_number} saved.')

    print ('End cycle')
