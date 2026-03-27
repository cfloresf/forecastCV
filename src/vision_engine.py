import numpy as np
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from PIL import Image
import os
import time

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
    # Configure API key if available
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
except ImportError:
    _GENAI_AVAILABLE = False


MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class VisionEngine:
    @staticmethod
    def is_valid_chart(image_path: str) -> Tuple[bool, str]:
        """
        Validates that the image looks like a chart.
        Uses OpenCV on desktop; falls back to a Pillow-based heuristic on Android.
        Includes better error handling.
        """
        try:
            # First check: file exists and is readable
            if not os.path.exists(image_path):
                return False, "Image file not found."
            
            if _CV2_AVAILABLE:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return False, "Could not read image file (corrupted?)."
                blurred = cv2.GaussianBlur(img, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                lines = cv2.HoughLinesP(
                    edges, 1, np.pi / 180,
                    threshold=100, minLineLength=100, maxLineGap=10
                )
                if lines is None or len(lines) < 2:
                    return False, "No clear axes detected. Please capture a chart with visible axes."
                return True, "Valid chart detected"
            else:
                # Fallback: basic sanity check via Pillow
                img = Image.open(image_path).convert("L")
                arr = np.array(img, dtype=np.float32)
                
                # Check image is not too small
                if arr.size < 1000:
                    return False, "Image too small. Please capture a larger chart."
                
                # Rough edge proxy: standard deviation of gradient
                grad_x = np.diff(arr, axis=1)
                grad_y = np.diff(arr, axis=0)
                edge_energy = np.std(grad_x) + np.std(grad_y)
                if edge_energy < 10.0:
                    return False, "Image has too few edges. Please ensure the chart is clear and well-lit."
                return True, "Chart validation successful"
        except Exception as exc:
            return False, f"Image validation error: {exc}"

    @staticmethod
    def extract_time_series(image_path: str, frequency: str = "M") -> pd.DataFrame:
        """
        Extracts time-series data from a chart image using Gemini Vision API.
        Falls back to mock data if API is unavailable.
        Includes retry logic for network resilience.
        """
        if not _GENAI_AVAILABLE:
            print("⚠️  Gemini API not available. Using mock data for demonstration.")
            return VisionEngine._get_mock_data(frequency)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  GOOGLE_API_KEY not set. Using mock data for demonstration.")
            return VisionEngine._get_mock_data(frequency)
        
        for attempt in range(MAX_RETRIES):
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                
                prompt = f"""Extract numerical time-series data from this chart image.
                
Return a JSON object with this exact format:
{{
    "dates": ["2024-01-01", "2024-02-01", ...],
    "values": [value1, value2, ...]
}}

Important:
- Extract ALL visible data points
- Preserve chronological order
- Use appropriate date format based on visible labels
- Return ONLY the JSON, no other text"""
                
                response = model.generate_content([image_data, prompt])
                
                # Parse response
                json_str = response.text.strip()
                if json_str.startswith("```"):
                    json_str = json_str.split("```")[1].strip()
                    if json_str.startswith("json"):
                        json_str = json_str[4:].strip()
                
                data = json.loads(json_str)
                df = pd.DataFrame({
                    "date": pd.to_datetime(data["dates"]),
                    "value": data["values"]
                })
                return df.sort_values("date").reset_index(drop=True)
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"⚠️  API call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"❌ Failed to extract data after {MAX_RETRIES} attempts. Using mock data.")
                    return VisionEngine._get_mock_data(frequency)
        
        # Fallback
        return VisionEngine._get_mock_data(frequency)

    @staticmethod
    def _get_mock_data(frequency: str = "M") -> pd.DataFrame:
        """Returns mock data for demonstration when API is unavailable."""
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
