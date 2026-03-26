import numpy as np
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from PIL import Image

# OpenCV is optional — not available on Android (no arm64 wheel on PyPI).
# All CV operations fall back to Pillow + NumPy when cv2 is absent.
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class VisionEngine:
    @staticmethod
    def is_valid_chart(image_path: str) -> Tuple[bool, str]:
        """
        Validates that the image looks like a chart.
        Uses OpenCV on desktop; falls back to a Pillow-based heuristic on Android.
        """
        if _CV2_AVAILABLE:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, "Could not read image."
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=100, minLineLength=100, maxLineGap=10
            )
            if lines is None or len(lines) < 2:
                return False, "No clear axes detected."
            return True, "Success"
        else:
            # Fallback: basic sanity check via Pillow
            try:
                img = Image.open(image_path).convert("L")
                arr = np.array(img, dtype=np.float32)
                # Rough edge proxy: standard deviation of gradient
                grad_x = np.diff(arr, axis=1)
                grad_y = np.diff(arr, axis=0)
                edge_energy = np.std(grad_x) + np.std(grad_y)
                if edge_energy < 10.0:
                    return False, "Image has too few edges to be a chart."
                return True, "Success (fallback validator)"
            except Exception as exc:
                return False, f"Could not read image: {exc}"

    @staticmethod
    def extract_time_series(image_path: str, frequency: str = "M") -> pd.DataFrame:
        """
        Extracts time-series data from a chart image.
        Currently returns mock data; swap in Gemini Vision API call here.
        """
        dates = pd.date_range(start="2024-01-01", periods=12, freq=frequency)
        values = [10.0, 12.5, 11.2, 14.8, 15.1, 16.5, 18.0, 17.5, 20.1, 22.4, 25.0, 24.1]
        return pd.DataFrame({"date": dates, "value": values})

    @staticmethod
    def interpolate_data(df: pd.DataFrame, target_freq: str = "D") -> pd.DataFrame:
        """
        Linear interpolation using NumPy only (no scipy dependency).
        """
        df = df.set_index("date").sort_index()
        resampled = df.resample(target_freq).asfreq()

        all_x = np.arange(len(resampled))
        known_x = np.where(~np.isnan(resampled.iloc[:, 0]))[0]
        known_y = resampled.iloc[known_x, 0].values

        resampled.iloc[:, 0] = np.interp(all_x, known_x, known_y)
        return resampled.reset_index()
