# 📈 ForecastCam - Vision Powered Time-Series Forecaster

**ForecastCam** is a professional-grade mobile application that transforms physical charts into actionable digital intelligence. By combining state-of-the-art **Computer Vision** with robust **Time-Series Machine Learning**, it enables anyone to capture a trend-line and immediately see where it's headed.

---

## ✨ Key Features

- **📸 Intelligent Capture**: Real-time OpenCV validation ensures you only capture high-quality charts with detectable axes.
- **🧠 Neural Digitization**: Uses Gemini 2.0 Flash to extract precise numerical $(x, y)$ data points from photographs, automatically mapping dates and values.
- **🚀 Advanced Forecasting**:
  - **SARIMA**: High-precision statistical modeling for seasonal trends.
  - **LightGBM**: Gradient-boosted recursive forecasting for complex patterns.
- **📊 Interactive Visualization**: Glassmorphism UI with high-fidelity history and forecast charts.
- **📥 Seamless Export**: One-tap CSV generation for further analysis in Excel or Python.
- **📱 Smart Metadata**: User-defined horizon and frequency (Daily, Weekly, Monthly, etc.).

---

## 🛠️ Tech Stack

- **UI Framework**: [Flet](https://flet.dev) (Python-powered Flutter)
- **Computer Vision**: OpenCV, Gemini 2.0 Flash, PIL
- **Machine Learning**: LightGBM, Statsmodels (SARIMA), Scikit-Learn
- **Data Engine**: Pandas, Scipy (Interpolation)

---

## 🚀 How to Run Locally

Requires [uv](https://github.com/astral-sh/uv) or Python 3.10+.

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Run the App**:
   ```bash
   uv run python src/main.py
   ```

---

## 📱 How to Build for Android

ForecastCam is designed to be a native Android app. To compile:

1. **Initialize Flet Build**:
   ```bash
   flet build apk
   ```

2. **Requirements**:
   - Flutter SDK installed.
   - Android Studio / NDK for compilation.
   - For a full 'Wow' experience, provide your `GEMINI_API_KEY` in `src/vision_engine.py`.

---

## 🎨 Design Philosophy
ForecastCam uses a **Premium Dark Mode** with **Glassmorphism** and **Neon Cyan accents**. Every interaction is designed to feel fast, fluid, and professional, ensuring a high-end experience on any smartphone.
