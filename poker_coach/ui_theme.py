"""Оформление приложения: две палитры (светлая и тёмная) и таблица стилей Qt.

Тема по умолчанию следует системной; её можно переключить в боковой панели. Свои виджеты
(стол, карты, шкалы) берут цвета из current() при каждой отрисовке.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication


@dataclass(frozen=True)
class Palette:
    dark: bool
    bg: str              # фон окна
    sidebar: str         # фон боковой панели
    panel: str           # фон карточек-панелей
    text: str
    secondary: str
    tertiary: str
    separator: str
    fill: str            # фон обычных кнопок и сегментов
    fill_hover: str
    fill_pressed: str
    field: str           # фон полей ввода
    field_border: str
    selection: str       # выбранный пункт в списках
    accent: str
    accent_hover: str
    accent_tint: str     # прозрачная подложка цвета акцента
    green: str
    green_tint: str
    red: str
    red_tint: str
    orange: str
    orange_tint: str
    felt_center: str
    felt_edge: str
    card_back: str
    card_back_edge: str
    segment_on: str      # выбранный сегмент
    scroll: str          # ползунок прокрутки


LIGHT = Palette(
    dark=False, bg="#F5F5F7", sidebar="#ECECEF", panel="#FFFFFF", text="#1D1D1F", secondary="#6E6E73",
    tertiary="#A1A1A6", separator="#E3E3E8", fill="#ECECF0", fill_hover="#E3E3E8", fill_pressed="#D8D8DE",
    field="#FFFFFF", field_border="#D2D2D7", selection="#DCDCE1", accent="#0071E3", accent_hover="#0077ED",
    accent_tint="rgba(0,113,227,0.10)", green="#248A3D", green_tint="rgba(52,199,89,0.14)", red="#D70015",
    red_tint="rgba(255,59,48,0.12)", orange="#C93400", orange_tint="rgba(255,149,0,0.14)",
    felt_center="#2F7558", felt_edge="#1F5541", card_back="#2B3A5C", card_back_edge="#1B2640", segment_on="#FFFFFF",
    scroll="rgba(0,0,0,0.22)",
)

DARK = Palette(
    dark=True, bg="#1C1C1E", sidebar="#232325", panel="#2A2A2D", text="#F5F5F7", secondary="#98989F",
    tertiary="#636368", separator="#3A3A3D", fill="#3A3A3D", fill_hover="#444448", fill_pressed="#505055",
    field="#1E1E20", field_border="#434347", selection="#3A3A3D", accent="#0A84FF", accent_hover="#409CFF",
    accent_tint="rgba(10,132,255,0.18)", green="#30D158", green_tint="rgba(48,209,88,0.16)", red="#FF453A",
    red_tint="rgba(255,69,58,0.16)", orange="#FF9F0A", orange_tint="rgba(255,159,10,0.16)",
    felt_center="#2A6650", felt_edge="#18412F", card_back="#2E3F66", card_back_edge="#1A2542", segment_on="#636368",
    scroll="rgba(255,255,255,0.22)",
)

# Цвета мастей: четырёхцветная колода — пики и трефы не путаются с червами и бубнами.
SUIT_COLORS = {"s": "#1D1D1F", "h": "#E0352B", "d": "#1F6FE5", "c": "#23854A"}

FONT_FAMILIES = ["SF Pro Text", ".AppleSystemUIFont", "Inter", "Segoe UI Variable", "Segoe UI", "Helvetica Neue",
                 "Arial", "DejaVu Sans"]


def color(value: str, alpha: float | None = None) -> QColor:
    """'#RRGGBB' или 'rgba(r,g,b,a)' -> QColor; alpha (0..1) переопределяет прозрачность."""
    if value.startswith("rgba"):
        r, g, b, a = (float(x) for x in value[5:-1].split(","))
        c = QColor(int(r), int(g), int(b))
        c.setAlphaF(a)
    else:
        c = QColor(value)
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


def font(px: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


class _Theme(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.mode = "auto"           # auto / light / dark
        self.pal = LIGHT
        self._assets = Path(tempfile.mkdtemp(prefix="poker_coach_ui_"))

    def system_dark(self) -> bool:
        app = QGuiApplication.instance()
        if app is None:
            return False
        try:
            return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
        except AttributeError:       # Qt старше 6.5
            return app.palette().window().color().lightness() < 128

    def apply(self, mode: str | None = None) -> None:
        if mode is not None:
            self.mode = mode
        dark = self.system_dark() if self.mode == "auto" else self.mode == "dark"
        self.pal = DARK if dark else LIGHT
        app = QGuiApplication.instance()
        if app is not None:
            app.setFont(font(13))
            app.setStyleSheet(stylesheet(self.pal, self._chevron(self.pal)))
            for w in app.allWidgets():
                w.update()
        self.changed.emit()

    def _chevron(self, p: Palette) -> str:
        path = self._assets / f"chevron_{'dark' if p.dark else 'light'}.svg"
        if not path.exists():
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
                f'<path d="M2 3.5 L5 6.5 L8 3.5" fill="none" stroke="{p.secondary}" stroke-width="1.5" '
                'stroke-linecap="round" stroke-linejoin="round"/></svg>')
        return path.as_posix()


theme = _Theme()


def current() -> Palette:
    return theme.pal


def stylesheet(p: Palette, chevron: str) -> str:
    return f"""
* {{ outline: 0; }}
QWidget {{ color: {p.text}; font-size: 13px; }}
QMainWindow, QWidget#page, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget#scrollBody {{ background: {p.bg}; }}
QWidget#sidebar {{ background: {p.sidebar}; border-right: 1px solid {p.separator}; }}
QFrame#panel {{ background: {p.panel}; border: 1px solid {p.separator}; border-radius: 12px; }}
QFrame#panel QLabel, QFrame#panel QWidget#transparent {{ background: transparent; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 26px; font-weight: 600; }}
QLabel#appTitle {{ font-size: 15px; font-weight: 600; padding: 2px 10px; }}
QLabel#subtitle {{ color: {p.secondary}; font-size: 13px; }}
QLabel#section {{ font-size: 13px; font-weight: 600; }}
QLabel#caption {{ color: {p.secondary}; font-size: 12px; }}
QLabel#fieldLabel {{ color: {p.secondary}; font-size: 12px; }}
QLabel#headline {{ font-size: 22px; font-weight: 600; }}
QLabel#sidebarFooter {{ color: {p.tertiary}; font-size: 11px; padding: 0 10px; }}

QPushButton {{ background: {p.fill}; color: {p.text}; border: none; border-radius: 8px;
              padding: 6px 14px; font-weight: 500; min-height: 18px; }}
QPushButton:hover {{ background: {p.fill_hover}; }}
QPushButton:pressed {{ background: {p.fill_pressed}; }}
QPushButton:disabled {{ color: {p.tertiary}; }}
QPushButton#primary {{ background: {p.accent}; color: #FFFFFF; }}
QPushButton#primary:hover {{ background: {p.accent_hover}; }}
QPushButton#primary:disabled {{ background: {p.fill}; color: {p.tertiary}; }}
QPushButton#tinted {{ background: {p.accent_tint}; color: {p.accent}; }}
QPushButton#tinted:hover {{ background: {p.fill_hover}; }}
QPushButton#tinted:disabled {{ background: {p.fill}; color: {p.tertiary}; }}
QPushButton#plain {{ background: transparent; color: {p.accent}; padding: 6px 8px; }}
QPushButton#plain:hover {{ background: {p.accent_tint}; }}
QPushButton#action {{ min-height: 26px; padding: 8px 18px; border-radius: 10px; font-size: 14px; }}

QFrame#segmented {{ background: {p.fill}; border-radius: 8px; }}
QPushButton#segment {{ background: transparent; border-radius: 6px; padding: 4px 12px; font-weight: 500;
                      color: {p.secondary}; min-height: 16px; }}
QPushButton#segment:hover {{ color: {p.text}; }}
QPushButton#segment:checked {{ background: {p.segment_on}; color: {p.text}; font-weight: 600; }}
QFrame#segmentedSmall {{ background: {p.fill}; border-radius: 7px; }}
QFrame#segmentedSmall QPushButton#segment {{ padding: 3px 4px; font-size: 12px; border-radius: 5px; }}

QPushButton#chip {{ background: {p.panel}; border: 1px solid {p.separator}; border-radius: 6px;
                   padding: 0; font-weight: 600; font-size: 12px; }}
QPushButton#chip:hover {{ border: 1px solid {p.tertiary}; }}

QComboBox, QLineEdit {{ background: {p.field}; border: 1px solid {p.field_border};
    border-radius: 7px; padding: 5px 9px; min-height: 20px; selection-background-color: {p.accent};
    selection-color: #FFFFFF; }}
QComboBox:hover, QLineEdit:hover {{ border: 1px solid {p.tertiary}; }}
QComboBox:focus, QLineEdit:focus {{ border: 1px solid {p.accent}; }}
QComboBox:disabled, QLineEdit:disabled {{ color: {p.tertiary}; background: {p.bg}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({chevron}); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{ background: {p.panel}; border: 1px solid {p.separator}; padding: 4px;
    selection-background-color: {p.accent}; selection-color: #FFFFFF; outline: 0; }}

QTextEdit, QTextBrowser {{ background: transparent; border: none; selection-background-color: {p.accent};
    selection-color: #FFFFFF; }}

QListWidget {{ background: transparent; border: none; }}
QListWidget::item {{ padding: 7px 10px; border-radius: 7px; margin: 1px 0; }}
QListWidget::item:hover {{ background: {p.fill}; }}
QListWidget::item:selected {{ background: {p.selection}; color: {p.text}; }}
QListWidget#nav::item {{ padding: 8px 12px; font-size: 13px; }}
QListWidget#nav::item:selected {{ background: {p.accent}; color: #FFFFFF; font-weight: 600; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.scroll}; border-radius: 3px; min-height: 30px; margin: 0 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p.scroll}; border-radius: 3px; min-width: 30px; margin: 2px 0; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ width: 0; height: 0;
    background: transparent; border: none; }}

QToolTip {{ background: {p.panel}; color: {p.text}; border: 1px solid {p.separator}; border-radius: 6px;
    padding: 5px 8px; }}
QMenu {{ background: {p.panel}; border: 1px solid {p.separator}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 5px; }}
QMenu::item:selected {{ background: {p.accent}; color: #FFFFFF; }}
"""
