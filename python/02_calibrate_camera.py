from src.calibration.ps4_calibrator import PS4Calibratior

if __name__ == '__main__':
    calibrator = PS4Calibratior()
    calibrator.capture_data()
    calibrator.calculate_calibration_params()