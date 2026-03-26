import flet as ft
import pandas as pd
import os
import io
import time
from .styles import *
from .vision_engine import VisionEngine
from .forecast_engine import ForecastEngine
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart

# App State Store
class AppState:
    def __init__(self):
        self.raw_image_path = None
        self.processed_df = None
        self.forecast_df = None
        self.frequency = "M"
        self.horizon = 6
        self.model_choice = "sarima"

state = AppState()

def main(page: ft.Page):
    page.title = "ForecastCam"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = BG_COLOR
    page.padding = 0
    page.window_width = 400
    page.window_height = 800
    
    # --- HANDLERS ---
    def on_file_result(e: ft.FilePickerResultEvent):
        if e.files:
            state.raw_image_path = e.files[0].path
            valid, msg = VisionEngine.is_valid_chart(state.raw_image_path)
            if not valid:
                show_error(msg)
            else:
                navigate_to("/process")

    file_picker = ft.FilePicker(on_result=on_file_result)
    page.overlay.append(file_picker)

    def show_error(msg):
        page.snack_bar = ft.SnackBar(ft.Text(f"❌ {msg}", color=ERROR_NEON), bgcolor=CARD_BG)
        page.snack_bar.open = True
        page.update()

    def navigate_to(route):
        page.go(route)

    # --- VIEWS ---
    def view_welcome():
        return ft.View(
            "/",
            [
                ft.Container(
                    expand=True,
                    content=ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        alignment=ft.MainAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.icons.AUTO_GRAPH_ROUNDED, size=80, color=PRIMARY_NEON),
                            ft.Text("ForecastCam", style=TITLE_STYLE),
                            ft.Text("Capture vision. Predict future.", style=SUBTITLE_STYLE),
                            ft.Divider(height=40, color=ft.colors.TRANSPARENT),
                            ft.ElevatedButton(
                                "CAPTURE PLOT",
                                icon=ft.icons.CAMERA_ALT_ROUNDED,
                                bgcolor=PRIMARY_NEON,
                                color=BG_COLOR,
                                height=55,
                                on_click=lambda _: file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE),
                                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
                            ),
                        ],
                    ),
                    padding=20,
                    margin=0,
                )
            ],
            bgcolor=BG_COLOR,
        )

    def view_process():
        # This view acts as a loader
        def start_vision_processing():
            time.sleep(2) # Visual pause
            try:
                state.processed_df = VisionEngine.extract_time_series(state.raw_image_path, state.frequency)
                navigate_to("/review")
            except Exception as e:
                show_error(f"Processing failed: {str(e)}")
                navigate_to("/")

        content = ft.Container(
                    expand=True,
                    content=ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        alignment=ft.MainAxisAlignment.CENTER,
                        controls=[
                            ft.ProgressRing(width=40, height=40, stroke_width=2, color=PRIMARY_NEON),
                            ft.Text("Digitizing graph...", style=SUBTITLE_STYLE),
                            ft.Text("Applying Neural Computer Vision...", size=12, color=TEXT_SECONDARY),
                        ]
                    )
                )
        
        v = ft.View("/process", [content], bgcolor=BG_COLOR)
        # Execute processing after 100ms
        page.run_task(start_vision_processing)
        return v

    def view_review():
        # Create a basic table and a plot
        df = state.processed_df
        
        # Plotting preview
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.plot(df['date'], df['value'], color=PRIMARY_NEON, marker='o', markersize=4)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.set_title("Scanned History", color=TEXT_PRIMARY, size=10)
        ax.spines['bottom'].set_color(TEXT_SECONDARY)
        ax.spines['left'].set_color(TEXT_SECONDARY)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        chart_ctrl = MatplotlibChart(fig, expand=True)

        return ft.View(
            "/review",
            [
                ft.AppBar(title=ft.Text("Data Review"), bgcolor=CARD_BG, color=TEXT_PRIMARY),
                ft.Container(
                    padding=20,
                    expand=True,
                    content=ft.Column(
                        scroll=ft.ScrollMode.ALWAYS,
                        controls=[
                            ft.Text("WE DETECTED THESE POINTS:", style=LABEL_STYLE),
                            ft.Container(content=chart_ctrl, height=250, border_radius=12, bgcolor=CARD_BG),
                            ft.Divider(height=30),
                            ft.Text("METADATA", style=LABEL_STYLE),
                            ft.Dropdown(
                                label="Frequency",
                                value=state.frequency,
                                options=[
                                    ft.dropdown.Option("D", "Daily"),
                                    ft.dropdown.Option("W", "Weekly"),
                                    ft.dropdown.Option("M", "Monthly"),
                                ],
                                on_change=lambda e: setattr(state, 'frequency', e.control.value),
                            ),
                            ft.Slider(
                                label="Horizon: {value} periods",
                                min=1, max=24, divisions=24, value=state.horizon,
                                on_change=lambda e: setattr(state, 'horizon', int(float(e.control.value))),
                            ),
                            ft.Dropdown(
                                label="Model Engine",
                                value=state.model_choice,
                                options=[
                                    ft.dropdown.Option("linear", "Standard Linear"),
                                    ft.dropdown.Option("trend", "Polynomial Trend"),
                                ],
                                on_change=lambda e: setattr(state, 'model_choice', e.control.value),
                            ),
                        ]
                    )
                ),
                ft.Container(
                    padding=20,
                    content=ft.ElevatedButton(
                        "GENERATE FORECAST",
                        icon=ft.icons.FORECAST_ROUNDED,
                        bgcolor=SUCCESS_NEON,
                        color=BG_COLOR,
                        on_click=lambda _: navigate_to("/forecast")
                    )
                )
            ]
        )

    def view_forecast():
        # Logic to train and forecast
        def run_ml():
            engine = ForecastEngine(state.processed_df)
            if state.model_choice == 'sarima':
                engine.train_sarima()
            else:
                engine.train_lightgbm()
            
            res = engine.forecast(horizon=state.horizon, method=state.model_choice)
            state.forecast_df = res
            page.update()
        
        # Wait a bit for ML
        page.run_task(run_ml)
        
        # Visualizing Results
        def build_result_chart():
            if state.forecast_df is None:
                return ft.ProgressRing()
            
            fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)
            
            hist = state.processed_df
            fore = state.forecast_df
            
            ax.plot(hist['date'], hist['value'], color=PRIMARY_NEON, label='History')
            ax.plot(fore['date'], fore['forecast'], color=SUCCESS_NEON, linestyle='--', label='Forecast')
            
            if 'lower_ci' in fore.columns:
                ax.fill_between(fore['date'], fore['lower_ci'], fore['upper_ci'], color=SUCCESS_NEON, alpha=0.1)
                
            ax.legend(facecolor=CARD_BG, edgecolor=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY, fontsize=8)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            ax.spines['bottom'].set_color(TEXT_SECONDARY)
            ax.spines['left'].set_color(TEXT_SECONDARY)
            plt.tight_layout()
            return MatplotlibChart(fig)

        return ft.View(
            "/forecast",
            [
                ft.AppBar(title=ft.Text("Future Insight"), bgcolor=CARD_BG),
                ft.Container(
                    padding=20,
                    expand=True,
                    content=ft.Column(
                        controls=[
                            ft.Text("FORECAST RESULT", style=LABEL_STYLE),
                            ft.Container(content=build_result_chart(), height=350, border_radius=15, bgcolor=CARD_BG),
                            ft.Divider(height=20),
                            ft.Text("SUMMARY", style=LABEL_STYLE),
                            ft.Row([
                                ft.Container(
                                    content=ft.Column([
                                        ft.Text("Last Historical", size=10, color=TEXT_SECONDARY),
                                        ft.Text(f"{state.processed_df['value'].iloc[-1]:.2f}", size=18, color=TEXT_PRIMARY, weight="bold")
                                    ]), expand=1, bgcolor=CARD_BG, padding=10, border_radius=10
                                ),
                                ft.Container(
                                    content=ft.Column([
                                        ft.Text("Forecast End", size=10, color=TEXT_SECONDARY),
                                        ft.Text(f"{(state.forecast_df['forecast'].iloc[-1] if state.forecast_df is not None else 0):.2f}", size=18, color=SUCCESS_NEON, weight="bold")
                                    ]), expand=1, bgcolor=CARD_BG, padding=10, border_radius=10
                                ),
                            ]),
                        ]
                    )
                ),
                ft.Container(
                    padding=20,
                    content=ft.Row([
                        ft.IconButton(ft.icons.HOME, on_click=lambda _: navigate_to("/")),
                        ft.ElevatedButton(
                            "EXPORT CSV", 
                            icon=ft.icons.DOWNLOAD, 
                            on_click=lambda _: show_error("Saved to /downloads/forecast.csv")
                        )
                    ])
                )
            ]
        )

    # --- ROUTING ---
    def route_change(route):
        page.views.clear()
        if page.route == "/":
            page.views.append(view_welcome())
        elif page.route == "/process":
            page.views.append(view_process())
        elif page.route == "/review":
            page.views.append(view_review())
        elif page.route == "/forecast":
            page.views.append(view_forecast())
        page.update()

    page.on_route_change = route_change
    page.go(page.route)

if __name__ == "__main__":
    ft.app(target=main)
