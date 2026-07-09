from __future__ import annotations

from types import SimpleNamespace


TOKENS = SimpleNamespace(
    palette=SimpleNamespace(
        background="#f4f6f8",
        surface="#ffffff",
        surface_alt="#f8fafc",
        border="#d8dee8",
        border_strong="#b8c2d1",
        text="#1d2433",
        text_muted="#5c6675",
        text_subtle="#7b8494",
        primary="#2457a6",
        primary_hover="#1f4d93",
        primary_soft="#e8f0fe",
        success="#166534",
        success_bg="#dcfce7",
        success_border="#22c55e",
        warning="#854d0e",
        warning_bg="#fef9c3",
        warning_border="#eab308",
        danger="#991b1b",
        danger_bg="#fee2e2",
        danger_border="#ef4444",
        neutral="#4b5563",
        neutral_bg="#eef1f5",
        neutral_border="#c8d0dc",
        info="#0f6785",
        info_bg="#e2f5fb",
        info_border="#78bfd6",
    ),
    spacing=SimpleNamespace(xs=4, sm=8, md=12, lg=16, xl=20),
    radius_sm=4,
    radius_md=6,
    radius_lg=8,
)

STATUS_COLORS: dict[str, tuple[str, str, str]] = {
    "success": (
        TOKENS.palette.success,
        TOKENS.palette.success_bg,
        TOKENS.palette.success_border,
    ),
    "warning": (
        TOKENS.palette.warning,
        TOKENS.palette.warning_bg,
        TOKENS.palette.warning_border,
    ),
    "danger": (
        TOKENS.palette.danger,
        TOKENS.palette.danger_bg,
        TOKENS.palette.danger_border,
    ),
    "error": (
        TOKENS.palette.danger,
        TOKENS.palette.danger_bg,
        TOKENS.palette.danger_border,
    ),
    "info": (
        TOKENS.palette.info,
        TOKENS.palette.info_bg,
        TOKENS.palette.info_border,
    ),
    "neutral": (
        TOKENS.palette.neutral,
        TOKENS.palette.neutral_bg,
        TOKENS.palette.neutral_border,
    ),
}

STATUS_TEXT_COLORS: dict[str, str] = {
    "success": "#1f7a3a",
    "warning": "#9a5a00",
    "danger": "#b00020",
    "error": "#b00020",
    "info": TOKENS.palette.info,
    "neutral": TOKENS.palette.neutral,
}


def status_color(kind: str) -> str:
    return STATUS_TEXT_COLORS.get(kind, STATUS_TEXT_COLORS["neutral"])


def status_badge_qss(kind: str = "neutral") -> str:
    foreground, background, border = STATUS_COLORS.get(kind, STATUS_COLORS["neutral"])
    return (
        "QLabel {"
        f"color: {foreground}; background: {background}; border: 1px solid {border};"
        f"border-radius: {TOKENS.radius_sm}px; padding: 3px 8px; font-weight: 600;"
        "}"
    )


def status_text_qss(kind: str = "neutral") -> str:
    return f"color: {status_color(kind)}; font-weight: 600;"


def muted_text_qss() -> str:
    return f"color: {TOKENS.palette.text_muted};"


def metric_card_qss() -> str:
    palette = TOKENS.palette
    return (
        "QLabel {"
        f"background: {palette.surface}; border: 1px solid {palette.border};"
        f"border-radius: {TOKENS.radius_lg}px; padding: 10px 12px;"
        "}"
    )


def section_card_qss() -> str:
    palette = TOKENS.palette
    return (
        "QFrame#sectionCard {"
        f"background: {palette.surface}; border: 1px solid {palette.border};"
        f"border-radius: {TOKENS.radius_lg}px;"
        "}"
    )


def empty_table_state_qss() -> str:
    palette = TOKENS.palette
    return (
        "QLabel {"
        f"color: {palette.text_muted}; background: rgba(255, 255, 255, 210);"
        f"border: 1px dashed {palette.border_strong};"
        f"border-radius: {TOKENS.radius_md}px; padding: 14px;"
        "}"
    )


def preview_surface_qss() -> str:
    return "QLabel { background: #171717; border: 1px solid #555; }"


def toolbar_button_qss() -> str:
    palette = TOKENS.palette
    return (
        "QToolButton {"
        f"background: {palette.surface}; color: {palette.text};"
        f"border: 1px solid {palette.border}; border-radius: {TOKENS.radius_sm}px;"
        "padding: 6px 10px; font-weight: 600;"
        "}"
        "QToolButton:hover {"
        f"background: {palette.primary_soft}; border-color: {palette.primary};"
        "}"
        "QToolButton:pressed {"
        f"background: {palette.primary}; color: #ffffff;"
        "}"
        "QToolButton:disabled {"
        f"color: {palette.text_subtle}; background: {palette.neutral_bg};"
        "}"
    )


def button_variant_qss(variant: str = "primary") -> str:
    palette = TOKENS.palette
    colors = {
        "primary": (palette.primary, "#ffffff", palette.primary_hover),
        "secondary": (palette.neutral_bg, palette.text, "#e2e7ef"),
        "danger": (palette.danger, "#ffffff", "#93001b"),
        "success": (palette.success, "#ffffff", "#17652f"),
    }
    background, foreground, hover = colors.get(variant, colors["primary"])
    return (
        "QPushButton {"
        f"background: {background}; color: {foreground}; border: none;"
        f"border-radius: {TOKENS.radius_sm}px; padding: 7px 12px; font-weight: 600;"
        "}"
        f"QPushButton:hover {{ background: {hover}; }}"
        f"QPushButton:disabled {{ background: {palette.neutral_border}; color: #ffffff; }}"
    )


APP_STYLE = f"""
QMainWindow, QWidget {{
    background: {TOKENS.palette.background};
    color: {TOKENS.palette.text};
    font-size: 13px;
}}

QMenuBar {{
    background: {TOKENS.palette.surface};
    border-bottom: 1px solid {TOKENS.palette.border};
    padding: 2px;
}}

QMenuBar::item {{
    padding: 5px 9px;
    border-radius: {TOKENS.radius_sm}px;
}}

QMenuBar::item:selected {{
    background: {TOKENS.palette.primary_soft};
}}

QMenu {{
    background: {TOKENS.palette.surface};
    border: 1px solid {TOKENS.palette.border};
    padding: 5px;
}}

QMenu::item {{
    padding: 6px 22px;
}}

QMenu::item:selected {{
    background: {TOKENS.palette.primary_soft};
}}

QStatusBar {{
    background: {TOKENS.palette.surface};
    border-top: 1px solid {TOKENS.palette.border};
    color: {TOKENS.palette.text_muted};
}}

QToolBar {{
    background: {TOKENS.palette.surface};
    border: none;
    border-bottom: 1px solid {TOKENS.palette.border};
    padding: 6px;
    spacing: 6px;
}}

{toolbar_button_qss()}

QTabWidget::pane {{
    border: 1px solid {TOKENS.palette.border};
    background: {TOKENS.palette.surface};
}}

QTabBar::tab {{
    background: #e9edf3;
    border: 1px solid {TOKENS.palette.border};
    border-bottom: none;
    padding: 9px 14px;
    margin-right: 2px;
    font-weight: 600;
    color: {TOKENS.palette.text_muted};
}}

QTabBar::tab:selected {{
    background: {TOKENS.palette.surface};
    color: {TOKENS.palette.text};
}}

QTabBar::tab:hover {{
    background: {TOKENS.palette.primary_soft};
    color: {TOKENS.palette.text};
}}

QPushButton {{
    background: {TOKENS.palette.primary};
    color: #ffffff;
    border: none;
    border-radius: {TOKENS.radius_sm}px;
    padding: 7px 12px;
    font-weight: 600;
}}

QPushButton:hover {{
    background: {TOKENS.palette.primary_hover};
}}

QPushButton:disabled {{
    background: #9aa5b5;
    color: #ffffff;
}}

QPushButton[variant="secondary"] {{
    background: {TOKENS.palette.neutral_bg};
    color: {TOKENS.palette.text};
    border: 1px solid {TOKENS.palette.border};
}}

QPushButton[variant="secondary"]:hover {{
    background: #e2e7ef;
}}

QPushButton[variant="danger"] {{
    background: {TOKENS.palette.danger};
    color: #ffffff;
}}

QPushButton[variant="danger"]:hover {{
    background: #93001b;
}}

QPushButton[variant="success"] {{
    background: {TOKENS.palette.success};
    color: #ffffff;
}}

QPushButton[variant="success"]:hover {{
    background: #17652f;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {TOKENS.palette.surface};
    border: 1px solid #cfd6e2;
    border-radius: {TOKENS.radius_sm}px;
    padding: 4px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus {{
    border: 1px solid {TOKENS.palette.primary};
}}

QTableWidget {{
    background: {TOKENS.palette.surface};
    alternate-background-color: {TOKENS.palette.surface_alt};
    border: 1px solid #cfd6e2;
    border-radius: {TOKENS.radius_sm}px;
    gridline-color: #edf1f5;
    selection-background-color: {TOKENS.palette.primary_soft};
    selection-color: {TOKENS.palette.text};
}}

QTableWidget::item {{
    padding: 5px 7px;
}}

QHeaderView::section {{
    background: #eef1f5;
    border: none;
    border-right: 1px solid {TOKENS.palette.border};
    border-bottom: 1px solid {TOKENS.palette.border};
    padding: 7px;
    font-weight: 700;
    color: {TOKENS.palette.text};
}}

QGroupBox {{
    background: {TOKENS.palette.surface};
    border: 1px solid {TOKENS.palette.border};
    border-radius: {TOKENS.radius_lg}px;
    margin-top: 16px;
    padding-top: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {TOKENS.palette.text_muted};
    font-weight: 700;
}}

QSplitter::handle {{
    background: {TOKENS.palette.border};
}}

QPlainTextEdit#logViewer {{
    font-family: Consolas, "Cascadia Mono", monospace;
    line-height: 1.25em;
}}

QLabel#sectionTitle {{
    background: transparent;
    color: {TOKENS.palette.text};
    font-weight: 700;
    font-size: 14px;
}}

QLabel#sectionSubtitle {{
    background: transparent;
    color: {TOKENS.palette.text_muted};
}}
"""
