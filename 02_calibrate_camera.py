from src.calibration.ps4_calibrator import PS4Calibratior

if __name__ == '__main__':
    frame_calibrator = PS4Calibratior()
    frame_calibrator.capture_data()
    frame_calibrator.calculate_calibration_params()
