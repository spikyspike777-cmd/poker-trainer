"""Виджеты интерфейса, которые рисуются вручную: карты, стол, шкалы, переключатели."""
from __future__ import annotations

import math
from html import escape

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient, QLinearGradient
from PySide6.QtWidgets import (
    QAbstractButton, QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .ui_theme import SUIT_COLORS, color, current, font

RANKS_DESC = "AKQJT98765432"
RANK_TEXT = {"T": "10"}


# ---------- масти и карты ----------

def _suit_path(suit: str) -> QPainterPath:
    """Контур масти в квадрате 1×1."""
    p = QPainterPath()
    if suit == "h":
        p.moveTo(0.5, 0.94)
        p.cubicTo(0.18, 0.70, 0.0, 0.50, 0.0, 0.31)
        p.cubicTo(0.0, 0.14, 0.12, 0.04, 0.27, 0.04)
        p.cubicTo(0.38, 0.04, 0.46, 0.11, 0.5, 0.21)
        p.cubicTo(0.54, 0.11, 0.62, 0.04, 0.73, 0.04)
        p.cubicTo(0.88, 0.04, 1.0, 0.14, 1.0, 0.31)
        p.cubicTo(1.0, 0.50, 0.82, 0.70, 0.5, 0.94)
    elif suit == "d":
        p.moveTo(0.5, 0.0)
        p.cubicTo(0.62, 0.18, 0.76, 0.35, 0.90, 0.5)
        p.cubicTo(0.76, 0.65, 0.62, 0.82, 0.5, 1.0)
        p.cubicTo(0.38, 0.82, 0.24, 0.65, 0.10, 0.5)
        p.cubicTo(0.24, 0.35, 0.38, 0.18, 0.5, 0.0)
    else:
        if suit == "s":
            p.moveTo(0.5, 0.0)
            p.cubicTo(0.82, 0.24, 1.0, 0.42, 1.0, 0.58)
            p.cubicTo(1.0, 0.73, 0.88, 0.82, 0.74, 0.82)
            p.cubicTo(0.63, 0.82, 0.55, 0.77, 0.5, 0.68)
            p.cubicTo(0.45, 0.77, 0.37, 0.82, 0.26, 0.82)
            p.cubicTo(0.12, 0.82, 0.0, 0.73, 0.0, 0.58)
            p.cubicTo(0.0, 0.42, 0.18, 0.24, 0.5, 0.0)
        else:
            r = 0.235
            for cx, cy in ((0.5, 0.26), (0.25, 0.57), (0.75, 0.57)):
                p.addEllipse(QPointF(cx, cy), r, r)
            p.addEllipse(QPointF(0.5, 0.52), 0.13, 0.13)
        stem = QPainterPath()
        stem.moveTo(0.5, 0.60)
        stem.cubicTo(0.52, 0.80, 0.58, 0.92, 0.70, 1.0)
        stem.lineTo(0.30, 1.0)
        stem.cubicTo(0.42, 0.92, 0.48, 0.80, 0.5, 0.60)
        p = p.united(stem)
    p.setFillRule(Qt.FillRule.WindingFill)
    return p


_SUITS = {s: _suit_path(s) for s in "shdc"}


def paint_suit(p: QPainter, suit: str, rect: QRectF, col: QColor | None = None) -> None:
    p.save()
    p.translate(rect.topLeft())
    p.scale(rect.width(), rect.height())
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(col or QColor(SUIT_COLORS[suit]))
    p.drawPath(_SUITS[suit])
    p.restore()


def paint_card(p: QPainter, r: QRectF, card: str | None, dim: bool = False, shadow: bool = True) -> None:
    """Карта лицом (card = 'Ah') или рубашкой (card = None)."""
    pal = current()
    radius = r.width() * 0.13
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if dim:
        p.setOpacity(0.35)
    if shadow:
        for i, a in ((3, 0.05), (2, 0.07), (1, 0.10)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, int(255 * a)))
            p.drawRoundedRect(r.adjusted(-i * 0.3, i * 0.6, i * 0.3, i * 0.9), radius + i * 0.4, radius + i * 0.4)
    if card is None:
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, QColor(pal.card_back))
        g.setColorAt(1, QColor(pal.card_back_edge))
        p.setBrush(g)
        p.setPen(QPen(QColor(0, 0, 0, 60), 0.8))
        p.drawRoundedRect(r, radius, radius)
        inner = r.adjusted(r.width() * 0.12, r.width() * 0.12, -r.width() * 0.12, -r.width() * 0.12)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 55), 1.0))
        p.drawRoundedRect(inner, radius * 0.6, radius * 0.6)
        p.restore()
        return
    p.setBrush(QColor("#FFFFFF"))
    p.setPen(QPen(QColor(0, 0, 0, 38), 0.8))
    p.drawRoundedRect(r, radius, radius)
    rank, suit = card[0], card[1]
    col = QColor(SUIT_COLORS[suit])
    h = r.height()
    p.setPen(col)
    f = font(max(8, int(h * 0.30)), QFont.Weight.DemiBold)
    p.setFont(f)
    text_rect = QRectF(r.left() + r.width() * 0.10, r.top() + h * 0.04, r.width() * 0.8, h * 0.34)
    p.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, RANK_TEXT.get(rank, rank))
    small = h * 0.17
    paint_suit(p, suit, QRectF(r.left() + r.width() * 0.12, r.top() + h * 0.39, small * 0.92, small), col)
    big = h * 0.42
    paint_suit(p, suit, QRectF(r.right() - big * 0.92 - r.width() * 0.10, r.bottom() - big - h * 0.08, big * 0.92, big), col)
    p.restore()


class CardRow(QWidget):
    """Ряд карт; пустые места рисуются пунктирными контурами."""

    def __init__(self, card_h: int = 56, slots: int = 0, parent=None):
        super().__init__(parent)
        self.cards: list[str] = []
        self.card_h, self.slots = card_h, slots
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_cards(self, cards: list[str]) -> None:
        self.cards = list(cards)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        w = self.card_h * 0.7
        n = max(len(self.cards), self.slots)
        return QSize(int(n * (w + 6)) + 4, self.card_h + 6)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.card_h * 0.7, self.card_h
        for i in range(max(len(self.cards), self.slots)):
            r = QRectF(2 + i * (w + 6), 1, w, h)
            if i < len(self.cards):
                paint_card(p, r, self.cards[i])
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)
                pen = QPen(color(current().tertiary, 0.7), 1.2, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawRoundedRect(r.adjusted(1, 1, -1, -1), w * 0.13, w * 0.13)
        p.end()


# ---------- элементы управления ----------

class Switch(QAbstractButton):
    """Переключатель в стиле macOS. Ведёт себя как QCheckBox: isChecked, setChecked, toggled."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0
        self._anim = QVariantAnimation(self, duration=160, easingCurve=QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._moved)
        self.toggled.connect(self._animate)

    def _moved(self, v) -> None:
        self._pos = float(v)
        self.update()

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on: bool) -> None:  # noqa: N802 — имя из Qt
        super().setChecked(on)
        self._pos = 1.0 if on else 0.0

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(34 + (10 + fm.horizontalAdvance(self.text()) + 2 if self.text() else 0), 22)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _e) -> None:
        pal = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.45)
        track = QRectF(0, (self.height() - 20) / 2, 34, 20)
        off, on = QColor(pal.fill_pressed), QColor(pal.accent)
        t = self._pos
        mix = QColor(int(off.red() + (on.red() - off.red()) * t), int(off.green() + (on.green() - off.green()) * t),
                     int(off.blue() + (on.blue() - off.blue()) * t))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(mix)
        p.drawRoundedRect(track, 10, 10)
        knob = QRectF(track.left() + 2 + t * 14, track.top() + 2, 16, 16)
        p.setBrush(QColor(0, 0, 0, 40))
        p.drawEllipse(knob.adjusted(0, 0.8, 0, 0.8))
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(knob)
        if self.text():
            p.setPen(QColor(pal.text))
            p.setFont(self.font())
            p.drawText(QRectF(44, 0, self.width() - 44, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        p.end()


class Segmented(QFrame):
    """Сегментированный переключатель: один вариант из нескольких."""

    changed = Signal(int)

    def __init__(self, labels: list[str], parent=None, small: bool = False):
        super().__init__(parent)
        self.setObjectName("segmentedSmall" if small else "segmented")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons = []
        for i, text in enumerate(labels):
            b = QPushButton(text)
            b.setObjectName("segment")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b)
            self.buttons.append(b)
        self.buttons[0].setChecked(True)
        self.group.idClicked.connect(self.changed)

    def index(self) -> int:
        return self.group.checkedId()

    def set_index(self, i: int) -> None:
        self.buttons[i].setChecked(True)


class Pill(QLabel):
    """Короткий статус на цветной подложке: верно / иначе / нейтрально."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kind = "neutral"
        self.hide()

    def show_status(self, text: str, kind: str = "neutral") -> None:
        self.kind = kind
        self.setText(text)
        self.restyle()
        self.setVisible(bool(text))

    def restyle(self) -> None:
        pal = current()
        fg, bg = {"good": (pal.green, pal.green_tint), "bad": (pal.red, pal.red_tint),
                  "warn": (pal.orange, pal.orange_tint), "accent": (pal.accent, pal.accent_tint)}.get(
            self.kind, (pal.secondary, pal.fill))
        self.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:9px; padding:3px 10px; "
                           f"font-size:12px; font-weight:600;")


def panel(title: str | None = None, caption: str | None = None, margins: int = 16, spacing: int = 10):
    """Карточка-панель с необязательным заголовком. -> (рамка, вертикальный макет внутри)."""
    frame = QFrame()
    frame.setObjectName("panel")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(margins, margins - 2, margins, margins)
    lay.setSpacing(spacing)
    if title:
        head = QLabel(title)
        head.setObjectName("section")
        lay.addWidget(head)
    if caption:
        cap = QLabel(caption)
        cap.setObjectName("caption")
        cap.setWordWrap(True)
        lay.addWidget(cap)
    return frame, lay


def label(text: str, name: str | None = None, wrap: bool = False) -> QLabel:
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


# ---------- шкалы ----------

class EquityBar(QWidget):
    """Эквити руки против отметки «нужно для колла»."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.equity: float | None = None
        self.required = 0.0
        self.setFixedHeight(38)

    def set_values(self, equity: float | None, required: float) -> None:
        self.equity, self.required = equity, required
        self.update()

    def paintEvent(self, _e) -> None:
        pal = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        p.setFont(font(12, QFont.Weight.Medium))
        if self.equity is None:
            p.setPen(QColor(pal.tertiary))
            p.drawText(QRectF(0, 0, w, 16), Qt.AlignmentFlag.AlignLeft, "Эквити появится после расчёта")
        else:
            ok = self.required == 0 or self.equity >= self.required
            p.setPen(QColor(pal.text))
            p.drawText(QRectF(0, 0, w, 16), Qt.AlignmentFlag.AlignLeft, f"Эквити {self.equity:.0%}")
            if self.required > 0:
                p.setPen(QColor(pal.secondary))
                p.drawText(QRectF(0, 0, w, 16), Qt.AlignmentFlag.AlignRight, f"нужно для колла {self.required:.0%}")
        track = QRectF(0, 24, w, 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pal.fill))
        p.drawRoundedRect(track, 4, 4)
        if self.equity is not None:
            ok = self.required == 0 or self.equity >= self.required
            fill = QRectF(track.left(), track.top(), max(8.0, track.width() * self.equity), track.height())
            p.setBrush(QColor(pal.green if ok else pal.orange))
            p.drawRoundedRect(fill, 4, 4)
        if self.required > 0:
            x = track.left() + track.width() * self.required
            p.setBrush(QColor(pal.text))
            p.drawRoundedRect(QRectF(x - 1.25, track.top() - 4, 2.5, track.height() + 8), 1.25, 1.25)
        p.end()


ACTION_RU = {"fold": "Фолд", "check": "Чек", "call": "Колл", "bet": "Бет", "raise": "Рейз", "allin": "Олл-ин"}


class MixBar(QWidget):
    """Смешанная стратегия солвера: доля каждого действия на одной полосе."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mix: list[tuple[str, float | None, float, float]] = []
        self.setFixedHeight(44)
        self.hide()

    def set_mix(self, mix) -> None:
        self.mix = list(mix or [])
        self.setVisible(len(self.mix) > 1)
        self.update()

    def _color(self, kind: str, i: int) -> QColor:
        pal = current()
        if kind == "fold":
            return QColor(pal.tertiary)
        if kind in ("check", "call"):
            return QColor(pal.green)
        c = QColor(pal.accent)
        c.setAlphaF(max(0.45, 1.0 - 0.2 * i))
        return c

    def paintEvent(self, _e) -> None:
        pal = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        track = QRectF(0, 0, w, 8)
        path = QPainterPath()
        path.addRoundedRect(track, 4, 4)
        p.setClipPath(path)
        x = 0.0
        aggressive = 0
        for kind, _amt, f, _ev in self.mix:
            seg = track.width() * f
            idx = aggressive if kind in ("bet", "raise", "allin") else 0
            aggressive += kind in ("bet", "raise", "allin")
            p.fillRect(QRectF(x, 0, seg, 8), self._color(kind, idx))
            x += seg
        p.setClipping(False)
        p.setFont(font(12))
        x = 0.0
        aggressive = 0
        for kind, amt, f, _ev in self.mix:
            idx = aggressive if kind in ("bet", "raise", "allin") else 0
            aggressive += kind in ("bet", "raise", "allin")
            name = ACTION_RU.get(kind, kind) + (f" {amt:g} bb" if amt and kind in ("bet", "raise") else "")
            text = f"{name} · {f:.0%}"
            tw = p.fontMetrics().horizontalAdvance(text) + 22
            if x + tw > w:
                break
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self._color(kind, idx))
            p.drawEllipse(QRectF(x, 23, 8, 8))
            p.setPen(QColor(pal.secondary))
            p.drawText(QRectF(x + 12, 16, tw, 22), Qt.AlignmentFlag.AlignVCenter, text)
            x += tw
        p.end()


class RangeMatrix(QWidget):
    """Сетка 13×13: диапазон оппонента. Одномастные — выше диагонали, разномастные — ниже."""

    def __init__(self, cell: int = 26, parent=None):
        super().__init__(parent)
        self.cell = cell
        self.weights: dict[str, float] = {}
        self.hero_key: str | None = None
        self.setFixedSize(13 * cell, 13 * cell)

    def set_data(self, weights: dict[str, float], hero_key: str | None) -> None:
        self.weights, self.hero_key = weights, hero_key
        self.update()

    def paintEvent(self, _e) -> None:
        pal = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(font(9, QFont.Weight.Medium))
        c = self.cell
        empty = QColor(pal.fill)
        for i, r1 in enumerate(RANKS_DESC):
            for j, r2 in enumerate(RANKS_DESC):
                key = r1 + r2 if i == j else (r1 + r2 + "s" if i < j else r2 + r1 + "o")
                wgt = self.weights.get(key, 0.0)
                rect = QRectF(j * c + 1, i * c + 1, c - 2, c - 2)
                p.setPen(Qt.PenStyle.NoPen)
                if wgt > 0:
                    col = QColor(pal.accent)
                    col.setAlphaF(0.25 + 0.75 * wgt)
                    p.setBrush(col)
                else:
                    p.setBrush(empty)
                p.drawRoundedRect(rect, 3, 3)
                if key == self.hero_key:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(QColor(pal.orange), 2))
                    p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 3, 3)
                p.setPen(QColor("#FFFFFF") if wgt > 0.45 else QColor(pal.secondary if wgt > 0 else pal.tertiary))
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, key)
        p.end()


# ---------- стол ----------

class TableView(QWidget):
    """Стол: игроки по кругу, борд и банк в центре, ставки фишками перед игроками."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.game = None
        self.reveal_style = False
        self.setMinimumSize(560, 360)
        pol = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        pol.setHeightForWidth(True)
        self.setSizePolicy(pol)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return int((w - 96) * 0.56 + 150)

    def sizeHint(self) -> QSize:
        return QSize(700, self.heightForWidth(700))

    def set_game(self, game, reveal_style: bool) -> None:
        self.game, self.reveal_style = game, reveal_style
        self.update()

    def paintEvent(self, _e) -> None:
        pal = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = self.game
        W, H = self.width(), self.height()
        fw = W - 96
        fh = min(H - 150, fw * 0.56)
        felt = QRectF((W - fw) / 2, (H - fh) / 2 - 16, fw, fh)
        # тень и сукно
        for i, a in ((10, 0.05), (6, 0.06), (3, 0.08)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, int(255 * a)))
            p.drawRoundedRect(felt.adjusted(-i * 0.4, i * 0.6, i * 0.4, i), felt.height() / 2, felt.height() / 2)
        grad = QRadialGradient(felt.center(), felt.width() * 0.6)
        grad.setColorAt(0, QColor(pal.felt_center))
        grad.setColorAt(1, QColor(pal.felt_edge))
        p.setBrush(grad)
        p.setPen(QPen(QColor(0, 0, 0, 70), 1))
        p.drawRoundedRect(felt, felt.height() / 2, felt.height() / 2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 22), 1.2))
        p.drawRoundedRect(felt.adjusted(14, 14, -14, -14), felt.height() / 2 - 14, felt.height() / 2 - 14)
        if g is None:
            p.end()
            return
        from .engine import H as HERO
        cx, cy = felt.center().x(), felt.center().y()
        # банк и борд
        pot = g.final_pot if g.finished else g.pot
        p.setPen(QColor(255, 255, 255, 235))
        p.setFont(font(14, QFont.Weight.DemiBold))
        p.drawText(QRectF(cx - 150, cy - 64, 300, 20), Qt.AlignmentFlag.AlignCenter, f"Банк {pot:g} bb")
        ch = min(64.0, felt.height() * 0.28)
        cw = ch * 0.7
        total = 5 * cw + 4 * 7
        for k in range(5):
            r = QRectF(cx - total / 2 + k * (cw + 7), cy - ch / 2 + 4, cw, ch)
            if k < len(g.board):
                paint_card(p, r, g.board[k])
            else:
                p.setBrush(QColor(255, 255, 255, 14))
                p.setPen(QPen(QColor(255, 255, 255, 30), 1))
                p.drawRoundedRect(r, cw * 0.13, cw * 0.13)
        # места
        n = len(g.players)
        rx, ry = felt.width() / 2 - 24, felt.height() / 2 + 4      # таблички сидят на бортике
        for i, pl in enumerate(g.players):
            ang = math.pi / 2 + 2 * math.pi * i / n
            sx, sy = cx + rx * math.cos(ang), cy + ry * math.sin(ang)
            self._seat(p, g, i, pl, QPointF(sx, sy), QPointF(cx, cy), i == HERO)
        p.end()

    def _seat(self, p: QPainter, g, i: int, pl, at: QPointF, center: QPointF, hero: bool) -> None:
        pal = current()
        acting = g.to_act == i and not g.finished
        folded = pl.folded
        # ставка: фишка между игроком и центром
        if pl.inv > 0:
            t = 0.42 if hero else 0.36
            bx, by = at.x() + (center.x() - at.x()) * t, at.y() + (center.y() - at.y()) * t
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(QColor("#F2A93B"))
            p.drawEllipse(QPointF(bx - 18, by), 7, 7)
            p.setPen(QColor(255, 255, 255, 240))
            p.setFont(font(12, QFont.Weight.DemiBold))
            p.drawText(QRectF(bx - 8, by - 9, 70, 18), Qt.AlignmentFlag.AlignVCenter, f"{pl.inv:g}")
        # карты
        shown = i in g.shown
        if not folded and pl.cards:
            upper = at.y() < center.y() - 4          # сидит в верхней половине — карты кладём под табличку
            if hero or shown:
                ch = 62 if hero else 44
                cw = ch * 0.7
                y = at.y() + 20 if upper else at.y() - 26 - ch + (6 if hero else 10)
                for k, c in enumerate(pl.cards):
                    paint_card(p, QRectF(at.x() - cw - 2 + k * (cw + 4), y, cw, ch), c)
            else:
                ch, cw = 34.0, 24.0
                y = at.y() + 12 if upper else at.y() - 26 - ch + 12
                for k in range(2):
                    paint_card(p, QRectF(at.x() - cw + 3 + k * (cw - 6), y, cw, ch), None, shadow=False)
        # табличка игрока
        w, h = (150, 50) if hero else (136, 46)
        box = QRectF(at.x() - w / 2, at.y() - h / 2, w, h)
        p.save()
        if folded:
            p.setOpacity(0.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 45))
        p.drawRoundedRect(box.adjusted(0, 2, 0, 3), 12, 12)
        p.setBrush(QColor(pal.panel))
        p.setPen(QPen(QColor(pal.accent), 2) if acting else QPen(QColor(pal.separator), 1))
        p.drawRoundedRect(box, 12, 12)
        name = "Вы" if hero else pl.name
        style = "" if hero or not self.reveal_style else f" · {pl.style.name.lower()}"
        p.setPen(QColor(pal.text))
        p.setFont(font(13, QFont.Weight.DemiBold))
        p.drawText(QRectF(box.left() + 12, box.top() + 6, w - 24, 18), Qt.AlignmentFlag.AlignLeft, name)
        if style:
            nw = p.fontMetrics().horizontalAdvance(name)
            p.setFont(font(12))
            p.setPen(QColor(pal.secondary))
            p.drawText(QRectF(box.left() + 12 + nw, box.top() + 7, w - 24 - nw, 18),
                       Qt.AlignmentFlag.AlignLeft, p.fontMetrics().elidedText(style, Qt.TextElideMode.ElideRight, int(w - 24 - nw)))
        p.setFont(font(12))
        p.setPen(QColor(pal.secondary))
        status = "сбросил" if folded else ("олл-ин" if pl.stack == 0 and not g.finished else f"{pl.stack:g} bb")
        p.drawText(QRectF(box.left() + 12, box.top() + 25, w - 24, 18), Qt.AlignmentFlag.AlignLeft,
                   f"{pl.label} · {status}")
        p.restore()
        # кнопка дилера
        if pl.pos == "BTN":
            d = QPointF(box.right() - 4, box.top() + 4)
            p.setPen(QPen(QColor(0, 0, 0, 50), 1))
            p.setBrush(QColor("#FFFFFF"))
            p.drawEllipse(d, 9, 9)
            p.setPen(QColor("#1D1D1F"))
            p.setFont(font(10, QFont.Weight.Bold))
            p.drawText(QRectF(d.x() - 9, d.y() - 9, 18, 18), Qt.AlignmentFlag.AlignCenter, "D")


# ---------- разбор тренера ----------

def advice_html(a, extra: str = "") -> str:
    """Причины совета и источник — для QTextBrowser. Заголовок совета показывается отдельно."""
    pal = current()
    rows = "".join(f'<p style="margin:0 0 9px 0; line-height:140%;">{escape(r)}</p>' for r in a.reasons)
    src = (f'<p style="margin:12px 0 0 0; color:{pal.tertiary}; font-size:12px;">'
           f'Источник: {escape(a.source)}</p>') if a.source else ""
    return f'<div style="color:{pal.text}; font-size:13px;">{rows}{src}{extra}</div>'


def summary_html(lines: list[tuple[bool, str]], title: str, result: str) -> str:
    pal = current()
    items = "".join(
        f'<p style="margin:0 0 4px 0;"><span style="color:{pal.green if ok else pal.red};">●</span>&nbsp; {escape(t)}</p>'
        for ok, t in lines)
    return (f'<p style="margin:14px 0 6px 0; font-weight:600;">{escape(title)}</p>'
            f'<p style="margin:0 0 8px 0; color:{pal.secondary};">{escape(result)}</p>{items}')
