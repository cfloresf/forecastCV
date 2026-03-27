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

### Prerequisites

1. **Set up Google Generative AI API Key** (required for vision processing):
   ```bash
   # On Windows (PowerShell):
   $env:GOOGLE_API_KEY = "your-api-key-here"
   
   # On macOS/Linux:
   export GOOGLE_API_KEY="your-api-key-here"
   ```
   
   Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

3. **Run the App**:
   ```bash
   uv run python main.py
   ```

---

## 📱 How to Build for Android

ForecastCam is designed to be a native Android app. To compile:

### Prerequisites for Android Build

1. **Set environment variable**:
   ```bash
   # You MUST set this before building for Android
   export GOOGLE_API_KEY="your-api-key-here"
   ```

2. **Install Flet CLI**:
   ```bash
   pip install flet-cli
   ```

3. **Build APK**:
   ```bash
   flet build apk
   ```
   
   This creates an unsigned APK in `build/outputs/apk/`.

4. **Sign and Deploy**:
   ```bash
   # Follow Android's official signing guide
   # Then use adb to deploy:
   adb install -r build/outputs/apk/forecastcv.apk
   ```

### Android Permissions

The app requires these permissions in `pubspec.yaml`:
- `CAMERA` - for file picker camera capture
- `READ_EXTERNAL_STORAGE` / `WRITE_EXTERNAL_STORAGE` - for file access
- `INTERNET` - for Gemini API calls

These are automatically requested on app startup on Android 6.0+.

---

## 🐛 Recent Fixes & Improvements

### v1.1.0 (Latest)

**Critical Fixes:**
- ✅ **Fixed black screen on Android** - App now explicitly navigates to home screen on startup
- ✅ **Added API key validation** - App warns if GOOGLE_API_KEY is not set
- ✅ **Improved error handling** - Better error messages throughout the app

**Improvements:**
- ✅ **Retry logic** - API calls now retry 3 times with exponential backoff
- ✅ **Data validation** - Improved chart detection and data quality checks
- ✅ **Better logging** - Console logs now show warnings and errors for debugging
- ✅ **CSV export** - Export functionality now works across platforms
- ✅ **Graceful fallbacks** - Uses mock data when API is unavailable (with warnings)

**Code Quality:**
- ✅ Improved error messages for users
- ✅ Better data validation in ForecastEngine
- ✅ More robust image validation
- ✅ Async error handling in all async operations
   flet build apk
   ```

2. **Requirements**:
   - Flutter SDK installed.
   - Android Studio / NDK for compilation.
   - For a full 'Wow' experience, provide your `GEMINI_API_KEY` in `src/vision_engine.py`.

---

## 🎨 Design Philosophy
ForecastCam uses a **Premium Dark Mode** with **Glassmorphism** and **Neon Cyan accents**. Every interaction is designed to feel fast, fluid, and professional, ensuring a high-end experience on any smartphone.
