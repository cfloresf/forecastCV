import cv2
import numpy as np
import base64
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import interpolate
import google.generativeai as genai
from PIL import Image
import io

# Setup Gemini Vision
# Note: In a real app, API_KEY should be handled securely.
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")  # Change to 2.0-flash once stable/available

class VisionEngine:
    @staticmethod
    def is_valid_chart(image_path: str) -> Tuple[bool, str]:
        """
        Heuristic check with OpenCV to see if the photo likely contains a chart.
        Checks for axes (long perpendicular lines).
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, "Could not read image."
        
        # Blur and edge detection
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough Line Transform to find axes
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 2:
            return False, "No clear axes detected. Ensure the chart is well-lit and contains straight lines for the X and Y axes."
        
        # Add basic count of vertical/horizontal lines
        h_lines = 0
        v_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10: h_lines += 1
            if abs(x2 - x1) < 10: v_lines += 1
            
        if h_lines < 1 or v_lines < 1:
            return False, "Failed to identify both X and Y axes. Please align the photo with the chart's axes."
            
        return True, "Success"

    @staticmethod
    def extract_time_series(image_path: str, frequency: str = "monthly") -> pd.DataFrame:
        """
        Uses Gemini Vision to perform 'Plot Digitization'.
        Extracts data points with dates and numerical values.
        """
        img = Image.open(image_path)
        
        prompt = f"""
        Analyze this chart image and extract the numerical time series data shown. 
        Focus strictly on the main trend line/curve. 
        Determine the frequency (provided as: {frequency}) and identify the axis labels.
        Map the curve's points into a JSON list of objects with 'date' and 'value'.
        Output strictly in JSON format: {{"data": [{{"date": "YYYY-MM-DD", "value": 123.45}}, ...]}}.
        If you cannot extract data, return {{"error": "Reason for failure"}}.
        Ensure point density matches the original plot.
        """
        
        # For demonstration context, we assume the API call would happen here.
        # In a real implementation without API keys, we'll simulate output or use dummy results.
        # response = model.generate_content([prompt, img])
        # data = json.loads(response.text)
        
        # MOCKING for development / demo:
        # Generate a synthetic range based on a standard 'upward with noise' trend
        dates = pd.date_range(start="2023-01-01", periods=12, freq='M')
        values = [100, 110, 105, 120, 130, 125, 140, 155, 150, 170, 185, 200]
        df = pd.DataFrame({"date": dates, "value": values})
        return df

    @staticmethod
    def interpolate_data(df: pd.DataFrame, target_freq: str = 'D') -> pd.DataFrame:
        """
        Ensures the extracted data has a continuous timeline by interpolating gaps.
        """
        df = df.set_index('date').sort_index()
        # Resample to daily (or specified freq) and interpolate
        resampled = df.resample(target_freq).asfreq()
        # Cubic spline for smoothness
        resampled['value'] = resampled['value'].interpolate(method='cubicspline')
        return resampled.reset_index()
