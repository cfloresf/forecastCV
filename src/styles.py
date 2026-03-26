import flet as ft

# Color Palette - Premium Dark Mode
BG_COLOR = "#0F111A"
CARD_BG = "#1A1D2E"
GLASS_BG = "0x1Affffff"  # Using 0x hex for some glass effects
PRIMARY_NEON = "#00F5FF"  # Electric Cyan
SECONDARY_NEON = "#7B61FF"  # Soft Violet
SUCCESS_NEON = "#00FFC2"  # Emerald
ERROR_NEON = "#FF4B6B"  # Soft Red
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#94A3B8"

# Glassmorphism Styles (approximated for Flet)
GLASS_STYLE = {
    "bgcolor": ft.colors.with_opacity(0.05, ft.colors.WHITE),
    "border": ft.border.all(1, ft.colors.with_opacity(0.1, ft.colors.WHITE)),
    "blur": ft.Blur(10, 10, ft.BlurStyle.INNER),
    "border_radius": 20,
}

NEON_SHADOW = ft.BoxShadow(
    spread_radius=1,
    blur_radius=15,
    color=ft.colors.with_opacity(0.3, PRIMARY_NEON),
    offset=ft.Offset(0, 0),
    blur_style=ft.ShadowBlurStyle.NORMAL,
)

# Typography
TITLE_STYLE = ft.TextStyle(size=28, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY, letter_spacing=1.2)
SUBTITLE_STYLE = ft.TextStyle(size=14, color=TEXT_SECONDARY)
LABEL_STYLE = ft.TextStyle(size=12, weight=ft.FontWeight.W_500, color=TEXT_SECONDARY)

# Transitions
TRANSITION_DURATION = 400
