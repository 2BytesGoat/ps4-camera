import os
import cv2
import time
from datetime import datetime

from src.preparation.data_source.ps4_data_source import PS4DataSource

# Photo session settings
total_photos = 30               # Number of images to take
countdown = 5                   # Interval for count-down timer, seconds
font = cv2.FONT_HERSHEY_SIMPLEX # Cowntdown timer font
camera_index = 0
grayscale = False
dst_folder = './data/calibration'

if __name__ == '__main__':
    # Initialize the camera
    data_source = PS4DataSource(camera_index, grayscale)

    # Create desitnation path
    os.makedirs(dst_folder, exist_ok=True)

    # Lets start taking photos! 
    counter = 0
    t2 = datetime.now()
    print ("Starting photo sequence")
    for frame_l, frame_r in data_source.stream():
        if frame_l is None or frame_r is None:
            break
        t1 = datetime.now()
        cntdwn_timer = countdown - int ((t1-t2).total_seconds())

        # If cowntdown is zero - let's record next image
        if cntdwn_timer == -1:
            counter += 1
            filename = f'{dst_folder}/scene_PS4cam_{counter}.png'
            cv2.imwrite(filename, frame_r)
            print (' ['+str(counter)+' of '+str(total_photos)+'] '+filename)
            t2 = datetime.now()
            time.sleep(1)
            cntdwn_timer = 0  # To avoid "-1" timer display next

        # Draw cowntdown counter, seconds
        cv2.putText(frame_r, str(cntdwn_timer), (50,50), font, 2.0, (0,0,255),4, cv2.LINE_AA)
        cv2.imshow("Frame Right", frame_r)
        key = cv2.waitKey(1) & 0xFF

        # Press 'Q' key to quit, or wait till all photos are taken
        if cv2.waitKey(1) == ord('q'):
                break

    data_source.close_stream()
