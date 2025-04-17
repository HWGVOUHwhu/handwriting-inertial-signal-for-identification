import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


target_length = 10000


def process(data_path, preprocessed_data_path):
    data = np.loadtxt(data_path)

    # 滤波
    filtered_data = savgol_filter(data, 51, 3, axis=1)

    # 样条插值对齐
    processed_signals = []
    for signal in filtered_data:
        x_origin = np.linspace(0, 1, len(signal))
        x_target = np.linspace(0, 1, target_length)

        spline = UnivariateSpline(x_origin, signal, s=0)
        new_signal = spline(x_target)
        processed_signals.append(new_signal)

    processed_signals = np.array(processed_signals)
    np.savetxt(preprocessed_data_path, processed_signals)

