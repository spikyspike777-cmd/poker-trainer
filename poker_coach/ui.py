"""Окно приложения. Запуск: python -m poker_coach.ui"""
from __future__ import annotations

import sys
from html import escape
from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import QLocale, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPushButton, QScrollArea,
    QSizePolicy, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from . import charts, ranges, theory
from .bots import STYLES
from .coach import Analysis, analyze
from .engine import STREET_RU, Game, H
from .hh import review_session
from .scenarios import random_state
from .state import LINE_LABELS, POSITIONS, PREFLOP_DECISIONS, GameState, PreflopLine
from .ui_theme import SUIT_COLORS, color, current, font, theme
from .ui_widgets import (
    CardRow, EquityBar, MixBar, Pill, RangeMatrix, Segmented, Switch, TableView, advice_html, label, paint_suit,
    panel, summary_html,
)

RANKS_DESC = "AKQJT98765432"


# ---------- ввод линии ставок ----------

_ACTION_WORDS = {"x": "check", "ч": "check", "чек": "check", "c": "call", "к": "call", "колл": "call"}


def parse_actions(text: str, board_cards: int, hero_ip: bool):
    """'x b3 c | x b6 c | x b9' -> (history, суммы, сколько герой вложил на текущей улице).

    Улицы разделяются «|», на каждой первым ходит игрок без позиции.
    """
    streets = ["flop", "turn", "river"][:max(board_cards - 2, 0)]
    parts = [p.strip() for p in text.replace(";", "|").split("|")]
    if len(parts) != len(streets):
        raise ValueError(f"Борд открыт до улицы «{streets[-1] if streets else 'префлоп'}»: нужно описать "
                         f"улиц — {len(streets)}, а в строке ставок их {len(parts)}")
    history, amounts, hero_inv = [], [], 0.0
    oop, ip = ("villain", "hero") if hero_ip else ("hero", "villain")
    for street, part in zip(streets, parts):
        inv = {"hero": 0.0, "villain": 0.0}
        for i, tok in enumerate(part.replace("/", " ").replace(",", " ").lower().split()):
            actor = oop if i % 2 == 0 else ip
            if tok in _ACTION_WORDS:
                kind, amount = _ACTION_WORDS[tok], 0.0
                if kind == "call":
                    inv[actor] = max(inv.values())
            elif tok[0] in "brбр" and tok[1:].replace(".", "", 1).isdigit():
                kind = "bet" if tok[0] in "bб" else "raise"
                amount = inv[actor] = float(tok[1:])
            else:
                raise ValueError(f"Не понимаю действие «{tok}». Примеры: x, c, b3, r9")
            history.append((street, actor, kind))
            amounts.append(amount)
        hero_inv = inv["hero"]
    return history, amounts, hero_inv


# ---------- общие части страниц ----------

def page_header(title: str, subtitle: str = "") -> tuple[QHBoxLayout, QLabel]:
    row = QHBoxLayout()
    row.setSpacing(12)
    col = QVBoxLayout()
    col.setSpacing(2)
    col.addWidget(label(title, "title"))
    sub = label(subtitle, "subtitle")
    col.addWidget(sub)
    row.addLayout(col, 1)
    return row, sub


def page(widget: QWidget) -> QVBoxLayout:
    widget.setObjectName("page")
    lay = QVBoxLayout(widget)
    lay.setContentsMargins(28, 22, 28, 22)
    lay.setSpacing(16)
    return lay


def text_view() -> QTextBrowser:
    t = QTextBrowser()
    t.setOpenExternalLinks(False)
    t.document().setDocumentMargin(0)
    t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return t


def dot_icon(col: str) -> QIcon:
    pm = QPixmap(QSize(20, 20))
    pm.setDevicePixelRatio(2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(col))
    p.drawEllipse(QRectF(2, 2, 6, 6))
    p.end()
    return QIcon(pm)


class _Job(QThread):
    """Тяжёлый расчёт (ходы ботов, совет тренера) в фоновом потоке, чтобы окно не замирало."""

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn, self.result = fn, None

    def run(self) -> None:
        self.result = self.fn()


class NumberField(QLineEdit):
    """Поле для суммы в bb: принимает число с точкой или запятой; value() и setValue() как у спинбокса."""

    def __init__(self, value: float = 0.0, maximum: float = 1000.0):
        super().__init__()
        self.maximum = maximum
        val = QDoubleValidator(0.0, maximum, 1, self)
        val.setNotation(QDoubleValidator.Notation.StandardNotation)
        val.setLocale(QLocale.c())
        self.setValidator(val)
        self.textEdited.connect(lambda t: self.setText(t.replace(",", ".")) if "," in t else None)
        self.setValue(value)

    def value(self) -> float:
        try:
            return min(self.maximum, max(0.0, float(self.text().replace(",", ".") or 0)))
        except ValueError:
            return 0.0

    def setValue(self, v: float) -> None:  # noqa: N802 — как у QDoubleSpinBox
        self.setText(f"{v:g}")


# ---------- выбор карт ----------

class CardChip(QAbstractButton):
    """Карта в сетке выбора: ранг и масть; выбранная в руку — синяя, на борд — тёмная."""

    def __init__(self, card: str, parent=None):
        super().__init__(parent)
        self.card = card
        self.state = ""                  # "", "hero", "board"
        self.setFixedSize(30, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, _e) -> None:
        pal = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        if self.state == "hero":
            bg, fg = QColor(pal.accent), QColor("#FFFFFF")
        elif self.state == "board":
            bg, fg = QColor(pal.text), QColor(pal.panel)
        else:
            bg, fg = QColor(pal.panel), QColor(SUIT_COLORS[self.card[1]])
            if pal.dark and self.card[1] == "s":
                fg = QColor(pal.text)
        p.setBrush(bg)
        p.setPen(Qt.PenStyle.NoPen if self.state else color(pal.tertiary if self.underMouse() else pal.separator))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(fg)
        p.setFont(font(12, QFont.Weight.DemiBold))
        p.drawText(QRectF(0, 3, self.width(), 16), Qt.AlignmentFlag.AlignCenter, "10" if self.card[0] == "T" else self.card[0])
        paint_suit(p, self.card[1], QRectF(self.width() / 2 - 5, 20, 10, 11), fg)
        p.end()

    def enterEvent(self, e) -> None:
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self.update()
        super().leaveEvent(e)


class CardPicker(QWidget):
    """Сетка из 52 карт. Клик добавляет карту в руку или на борд, повторный — убирает."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("transparent")
        self.hero: list[str] = []
        self.board: list[str] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        top = QHBoxLayout()
        self.target = Segmented(["Моя рука", "Борд"])
        top.addWidget(self.target)
        top.addStretch()
        lay.addLayout(top)
        grid = QGridLayout()
        grid.setSpacing(4)
        self.chips: dict[str, CardChip] = {}
        for r, suit in enumerate("shdc"):
            for c, rank in enumerate(RANKS_DESC):
                chip = CardChip(rank + suit)
                chip.clicked.connect(lambda _=False, card=rank + suit: self.toggle(card))
                self.chips[rank + suit] = chip
                grid.addWidget(chip, r, c)
        lay.addLayout(grid)
        rows = QHBoxLayout()
        rows.setSpacing(20)
        for title, attr, slots in (("Рука", "hero_row", 2), ("Борд", "board_row", 5)):
            col = QVBoxLayout()
            col.setSpacing(6)
            col.addWidget(label(title, "fieldLabel"))
            row = CardRow(52, slots)
            setattr(self, attr, row)
            col.addWidget(row)
            rows.addLayout(col)
        rows.addStretch()
        lay.addLayout(rows)
        self.refresh()

    def toggle(self, card: str) -> None:
        for lst in (self.hero, self.board):
            if card in lst:
                lst.remove(card)
                break
        else:
            if self.target.index() == 0 and len(self.hero) < 2:
                self.hero.append(card)
                if len(self.hero) == 2:
                    self.target.set_index(1)
            elif self.target.index() == 1 and len(self.board) < 5:
                self.board.append(card)
        self.refresh()
        self.changed.emit()

    def set_cards(self, hero: list[str], board: list[str]) -> None:
        self.hero, self.board = list(hero), list(board)
        self.refresh()

    def refresh(self) -> None:
        for card, chip in self.chips.items():
            chip.state = "hero" if card in self.hero else "board" if card in self.board else ""
            chip.update()
        self.hero_row.set_cards(self.hero)
        self.board_row.set_cards(self.board)


# ---------- разбор ситуации ----------

class CoachPanel(QFrame):
    """Правая колонка: вердикт, совет, эквити, смешанная стратегия и объяснение."""

    def __init__(self, title: str = "Тренер", placeholder: str = ""):
        super().__init__()
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(label(title, "section"))
        head.addStretch()
        self.pill = Pill()
        head.addWidget(self.pill)
        lay.addLayout(head)
        self.headline = label(placeholder, "headline", wrap=True)
        lay.addWidget(self.headline)
        self.bar = EquityBar()
        lay.addWidget(self.bar)
        self.mix = MixBar()
        lay.addWidget(self.mix)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setObjectName("separator")
        lay.addWidget(sep)
        self.sep = sep
        self.text = text_view()
        lay.addWidget(self.text, 1)
        self.analysis: Analysis | None = None
        self.extra = ""
        self.placeholder = placeholder
        theme.changed.connect(self.rerender)
        self.rerender()

    def show_analysis(self, a: Analysis | None, headline: str | None = None, extra: str = "") -> None:
        self.analysis, self.extra = a, extra
        if a is None:
            self.headline.setText(headline or self.placeholder)
            self.bar.set_values(None, 0)
            self.mix.set_mix([])
        else:
            self.headline.setText(headline or a.headline)
            self.bar.set_values(a.equity, a.required)
            self.mix.set_mix(a.mix)
        self.rerender()

    def show_message(self, headline: str, text: str = "") -> None:
        self.analysis, self.extra = None, ""
        self.headline.setText(headline)
        self.bar.set_values(None, 0)
        self.mix.set_mix([])
        self.text.setHtml(f'<p style="color:{current().secondary};">{escape(text)}</p>')

    def add_extra(self, html: str) -> None:
        self.extra += html
        self.rerender()

    def rerender(self) -> None:
        pal = current()
        self.sep.setStyleSheet(f"background:{pal.separator};")
        self.pill.restyle()
        if self.analysis is not None:
            self.text.setHtml(advice_html(self.analysis, self.extra))
        elif self.extra:
            self.text.setHtml(self.extra)
        else:
            self.text.setHtml("")


class TrainerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.analysis: Analysis | None = None
        root = page(self)
        head, _sub = page_header("Разбор ситуации", "Выставьте раздачу — тренер подскажет решение и объяснит его")
        self.btn_random = QPushButton("Случайная раздача")
        self.btn_clear = QPushButton("Сбросить")
        head.addWidget(self.btn_clear, 0, Qt.AlignmentFlag.AlignBottom)
        head.addWidget(self.btn_random, 0, Qt.AlignmentFlag.AlignBottom)
        root.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        # --- ввод: колонка с прокруткой ---
        left = QWidget()
        left.setObjectName("scrollBody")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        ll.setSpacing(16)
        box, bl = panel("Карты")
        self.picker = CardPicker()
        bl.addWidget(self.picker)
        ll.addWidget(box)

        box, bl = panel("Ситуация")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        self.line = QComboBox()
        for ln, text in LINE_LABELS.items():
            self.line.addItem(("Префлоп: " if ln in PREFLOP_DECISIONS else "Постфлоп: ") + text, ln)
        self.hero_pos, self.vil_pos = QComboBox(), QComboBox()
        self.hero_pos.addItems(POSITIONS)
        self.vil_pos.addItems(POSITIONS)
        self.pot, self.to_call, self.stack = (self._spin(v) for v in (1.5, 0, 100))
        self.actions_edit = QLineEdit()
        self.actions_edit.setPlaceholderText("x b3 c | x b6 c | x b9")
        fields = [("Что было на префлопе", self.line, 0, 0, 2), ("Моя позиция", self.hero_pos, 1, 0, 1),
                  ("Позиция оппонента", self.vil_pos, 1, 1, 1), ("Банк, bb", self.pot, 2, 0, 1),
                  ("Нужно доставить, bb", self.to_call, 2, 1, 1), ("Мой стек, bb", self.stack, 3, 0, 1),
                  ("Ставки на постфлопе", self.actions_edit, 4, 0, 2)]
        for text, w, r, c, span in fields:
            cell = QVBoxLayout()
            cell.setSpacing(4)
            cell.addWidget(label(text, "fieldLabel"))
            cell.addWidget(w)
            grid.addLayout(cell, r, c, 1, span)
        bl.addLayout(grid)
        bl.addWidget(label("Банк — вместе со ставкой оппонента; если оппонент чекнул, доставлять ничего не нужно. "
                           "Ставки необязательны: улицы через «|», первым ходит игрок без позиции; x — чек, c — колл, "
                           "b3 — бет 3 bb, r9 — рейз до 9 bb. С ними тренер восстанавливает диапазоны по всей раздаче.",
                           "caption", wrap=True))
        ll.addWidget(box)

        box, bl = panel("Ваше решение")
        self.quiz_mode = Switch("Сначала отвечаю сам, потом смотрю разбор")
        self.quiz_mode.setChecked(True)
        bl.addWidget(self.quiz_mode)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.answer_btns = {}
        for key, text in (("fold", "Фолд"), ("passive", "Чек или колл"), ("aggressive", "Бет или рейз")):
            b = QPushButton(text)
            b.setObjectName("tinted" if key == "aggressive" else "")
            b.clicked.connect(lambda _=False, k=key: self.run(k))
            self.answer_btns[key] = b
            row.addWidget(b)
        bl.addLayout(row)
        self.btn_analyze = QPushButton("Показать разбор")
        self.btn_analyze.setObjectName("primary")
        bl.addWidget(self.btn_analyze)
        ll.addWidget(box)
        ll.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(500)
        body.addWidget(scroll)

        # --- разбор ---
        right = QVBoxLayout()
        right.setSpacing(16)
        self.coach = CoachPanel("Разбор тренера", "Выставьте раздачу или нажмите «Случайная раздача»")
        right.addWidget(self.coach, 1)
        rbox, rl = panel("Диапазон оппонента")
        mrow = QHBoxLayout()
        self.matrix = RangeMatrix(24)
        mrow.addWidget(self.matrix)
        self.range_label = label("Появится после расчёта.", "caption", wrap=True)
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        mrow.addSpacing(8)
        mrow.addWidget(self.range_label, 1)
        rl.addLayout(mrow)
        right.addWidget(rbox, 0)
        body.addLayout(right, 1)

        self.btn_analyze.clicked.connect(lambda: self.run(None))
        self.btn_random.clicked.connect(self.load_random)
        self.btn_clear.clicked.connect(self.clear)
        self.line.currentIndexChanged.connect(self._sync)
        self.picker.changed.connect(self.reset_result)
        self.quiz_mode.toggled.connect(self._sync)
        self._sync()

    @staticmethod
    def _spin(value: float) -> "NumberField":
        return NumberField(value)

    def _sync(self) -> None:
        self.vil_pos.setEnabled(self.line.currentData() != PreflopLine.UNOPENED)
        for b in self.answer_btns.values():
            b.setEnabled(self.quiz_mode.isChecked())

    def state(self) -> GameState:
        line = self.line.currentData()
        base = dict(
            hero_cards=self.picker.hero, board=self.picker.board, hero_pos=self.hero_pos.currentText(),
            villain_pos=None if line == PreflopLine.UNOPENED else self.vil_pos.currentText(),
            preflop_line=line, pot=self.pot.value(), to_call=self.to_call.value(), stack=self.stack.value())
        text = self.actions_edit.text().strip()
        if not text:
            return GameState(**base)
        hero_ip = GameState(**base).in_position
        history, amounts, hero_inv = parse_actions(text, len(self.picker.board), hero_ip)
        return GameState(**base, history=history, history_amounts=amounts, hero_invested=hero_inv)

    def set_state(self, s: GameState) -> None:
        self.picker.set_cards(s.hero_cards, s.board)
        self.line.setCurrentIndex(self.line.findData(s.preflop_line))
        self.hero_pos.setCurrentText(s.hero_pos)
        if s.villain_pos:
            self.vil_pos.setCurrentText(s.villain_pos)
        self.pot.setValue(s.pot)
        self.to_call.setValue(s.to_call)
        self.stack.setValue(s.stack)
        self.actions_edit.clear()
        self.reset_result()

    def load_random(self) -> None:
        self.set_state(random_state())

    def clear(self) -> None:
        self.picker.set_cards([], [])
        self.picker.target.set_index(0)
        self.reset_result()

    def reset_result(self) -> None:
        self.analysis = None
        self.coach.pill.show_status("")
        self.coach.show_analysis(None, "Ваш ход. Что будете делать?")
        self.matrix.set_data({}, None)
        self.range_label.setText("Появится после расчёта.")

    # для совместимости со скриптами проверки
    @property
    def headline(self) -> QLabel:
        return self.coach.headline

    @property
    def text(self) -> QTextBrowser:
        return self.coach.text

    def run(self, answer: str | None) -> None:
        try:
            st = self.state()
            a = analyze(st)
        except ValidationError as e:
            self._error("; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors()))
            return
        except ValueError as e:
            self._error(str(e))
            return
        self.analysis = a
        self.coach.show_analysis(a)
        self.matrix.set_data(ranges.matrix(a.villain), charts.hand_key(st.hero_cards))
        if a.villain:
            self.range_label.setText(f"{a.villain.label[:1].upper()}{a.villain.label[1:]}.\n\n"
                                     f"Комбинаций: {len(a.villain.combos)}.\nОранжевая рамка — ваша рука.")
        else:
            self.range_label.setText("До вас все сбросили — оппонента пока нет.")
        if answer is None:
            self.coach.pill.show_status("")
            return
        kind = {"fold": "fold", "check": "passive", "call": "passive"}.get(a.action, "aggressive")
        ok = kind == answer
        self.coach.pill.show_status("Совпало с тренером" if ok else "Тренер сыграл бы иначе", "good" if ok else "bad")

    def _error(self, msg: str) -> None:
        self.reset_result()
        self.coach.pill.show_status("Проверьте ввод", "warn")
        self.coach.show_message("Не получается посчитать", msg)


# ---------- игра ----------

class GameTab(QWidget):
    """Стол на 2–9 игроков: решение на каждой улице и комментарий тренера."""

    def __init__(self):
        super().__init__()
        self.game = Game(n_players=6, styles="random")
        self.hands = 0
        self.total = 0.0
        self.counted = False
        self.revealed = False
        self.sync = False          # True — считать прямо в потоке окна (для скриптов и тестов)
        self.busy = False
        self._job = None
        self._hint = None          # совет тренера на текущий ход (считается один раз)
        root = page(self)
        head, self.score = page_header("Игра", "")
        self.count = QComboBox()
        for n in range(2, 10):
            self.count.addItem(f"{n} игроков" if n > 4 else f"{n} игрока", n)
        self.count.setCurrentIndex(4)
        self.style = QComboBox()
        self.style.addItem("Случайные характеры", "random")
        for st in STYLES.values():
            self.style.addItem(f"Все: {st.name.lower()}", st.key)
        self.btn_reveal = QPushButton("Показать характеры")
        self.btn_reveal.setObjectName("plain")
        for w in (self.btn_reveal, self.count, self.style):
            head.addWidget(w, 0, Qt.AlignmentFlag.AlignBottom)
        root.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)
        left = QVBoxLayout()
        left.setSpacing(14)
        body.addLayout(left, 1)
        self.table = TableView()
        left.addStretch(1)
        left.addWidget(self.table)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        actions_row = QHBoxLayout()
        actions_row.addStretch()
        actions_row.addLayout(self.actions)
        actions_row.addStretch()
        left.addLayout(actions_row)
        self.waiting = label("", "caption")
        self.waiting.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.waiting)
        left.addStretch(1)
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        bottom.addStretch()
        self.btn_overlay = QPushButton("Оверлей")
        self.btn_overlay.setToolTip("Компактное окно тренера поверх остальных окон")
        self.btn_hint = QPushButton("Подсказка")
        self.btn_new = QPushButton("Новая раздача")
        self.btn_new.setObjectName("primary")
        for b in (self.btn_overlay, self.btn_hint, self.btn_new):
            bottom.addWidget(b)
        left.addLayout(bottom)
        self.overlay = Overlay()

        right_w = QWidget()
        right_w.setFixedWidth(360)
        right = QVBoxLayout(right_w)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(16)
        self.coach = CoachPanel("Тренер", "Сделайте ход — тренер прокомментирует решение")
        right.addWidget(self.coach, 3)
        pbox, pl = panel("Соперники")
        self.use_profile = Switch("Тренер учитывает их профили")
        self.use_profile.setChecked(True)
        self.adaptive = Switch("Боты подстраиваются под меня")
        pl.addWidget(self.use_profile)
        pl.addWidget(self.adaptive)
        self.profile = text_view()
        self.profile.setFixedHeight(120)
        pl.addWidget(self.profile)
        right.addWidget(pbox, 0)
        lbox, lbl = panel("Ход раздачи")
        self.log = text_view()
        self.log.setFixedHeight(104)
        lbl.addWidget(self.log)
        right.addWidget(lbox, 0)
        body.addWidget(right_w)

        self.btn_new.clicked.connect(self.new_hand)
        self.btn_hint.clicked.connect(self.hint)
        self.btn_overlay.clicked.connect(self.toggle_overlay)
        self.count.currentIndexChanged.connect(self.reseat)
        self.style.currentIndexChanged.connect(self.reseat)
        self.use_profile.toggled.connect(lambda on: setattr(self.game, "use_profile", on))
        self.adaptive.toggled.connect(lambda on: setattr(self.game, "adaptive", on))
        self.btn_reveal.clicked.connect(self.reveal)
        self.overlay.live.toggled.connect(lambda _on: self.update_overlay())
        theme.changed.connect(self.refresh)
        self.refresh()

    # для совместимости со скриптами проверки
    @property
    def verdict(self) -> QLabel:
        return self.coach.headline

    def _run(self, fn, done, wait_text: str) -> None:
        """Выполнить fn в фоне и затем done(результат); пока считается — кнопки выключены."""
        if self.busy:
            return
        if self.sync:
            done(fn())
            return
        self.busy = True
        self.waiting.setText(wait_text)
        self._set_enabled(False)
        job = _Job(fn, self)

        def finish() -> None:
            self.busy = False
            self.waiting.setText("")
            self._set_enabled(True)
            done(job.result)

        job.finished.connect(finish)
        self._job = job
        job.start()

    def _set_enabled(self, on: bool) -> None:
        for w in (self.btn_new, self.btn_hint, self.count, self.style, self.use_profile, self.adaptive):
            w.setEnabled(on)
        for i in range(self.actions.count()):
            w = self.actions.itemAt(i).widget()
            if w:
                w.setEnabled(on)

    def reseat(self) -> None:
        """Сменить число игроков или характеры: профили и счёт начинаются заново."""
        self.revealed = False
        n, style = self.count.currentData(), self.style.currentData()

        def done(_r) -> None:
            self.hands, self.total = 0, 0.0
            self.new_hand_ui()

        self._run(lambda: self.game.setup(n, style), done, "Рассаживаю игроков…")

    def reveal(self) -> None:
        self.revealed = True
        self.refresh()

    def toggle_overlay(self) -> None:
        if self.overlay.isVisible():
            self.overlay.hide()
            return
        self.overlay.show()
        self.update_overlay()

    def update_overlay(self) -> None:
        """Оверлей показывает оценку последнего решения, а при включённом переключателе — совет на текущий ход."""
        if not self.overlay.isVisible():
            return
        g = self.game
        if not g.finished and self.overlay.live.isChecked():
            if self._hint is None:
                self.hint()                      # посчитается в фоне и вернётся сюда
                return
            self.overlay.show_advice(self._hint.headline, self._hint, "Совет", "accent")
        elif g.last is not None:
            d = g.last
            self.overlay.show_advice(d.advice.headline, d.advice, "Верно" if d.matched else f"Вы: {d.chosen}",
                                     "good" if d.matched else "bad")
        else:
            self.overlay.show_text("Ваш ход" if not g.finished else g.result)

    def new_hand(self) -> None:
        self._run(self.game.new_hand, lambda _r: self.new_hand_ui(), "Соперники думают…")

    def new_hand_ui(self) -> None:
        self.counted = False
        self._hint = None
        self.coach.pill.show_status("")
        self.coach.show_analysis(None)
        self.refresh()

    def hint(self) -> None:
        if self.game.finished:
            return

        def done(a) -> None:
            self._hint = a
            self.coach.pill.show_status("Подсказка", "accent")
            self.coach.show_analysis(a)
            self.update_overlay()

        if self._hint is not None:
            done(self._hint)
        else:
            self._run(lambda: self.game.advice(H), done, "Тренер считает…")

    def act(self, opt) -> None:
        g = self.game

        def done(_r) -> None:
            self._hint = None
            d = g.last
            street = STREET_RU[d.street]
            if d.matched:
                self.coach.pill.show_status(f"{street}: верно", "good")
                self.coach.show_analysis(d.advice, f"{d.chosen} — как у тренера")
            else:
                self.coach.pill.show_status(f"{street}: вы — {d.chosen}", "bad")
                self.coach.show_analysis(d.advice, f"Тренер: {d.advice.headline}")
            self.refresh()

        self._run(lambda: g.hero_act(opt), done, "Соперники думают…")

    def refresh(self) -> None:
        g = self.game
        pal = current()
        while self.actions.count():
            w = self.actions.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        caption_added = False
        for opt in g.options():
            words = opt.label.split()
            is_size = opt.kind == "raise" and words[0] == "Бет" and len(words) == 3
            if is_size and not caption_added:            # «Бет» один раз, на кнопках — только размер
                cap = label("Бет", "caption")
                cap.setContentsMargins(6, 0, 0, 0)
                self.actions.addWidget(cap)
                caption_added = True
            if is_size:
                text = f"{words[1]}\n{words[2]} bb"
            elif opt.kind in ("raise",) and words[-1].replace(".", "", 1).isdigit():
                text = f"{' '.join(words[:-1])}\n{words[-1]} bb"
            else:
                text = opt.label
            b = QPushButton(text)
            b.setObjectName("tinted" if opt.kind == "raise" else "")
            b.setMinimumHeight(46)
            b.setMinimumWidth(64 if is_size else 84)
            b.setStyleSheet("padding: 5px 12px; border-radius: 10px; font-size: 13px;")
            b.setEnabled(not self.busy)
            b.clicked.connect(lambda _=False, o=opt: self.act(o))
            self.actions.addWidget(b)
        if not self.busy:
            self.waiting.setText(g.result if g.finished else "")
        self.btn_hint.setEnabled(not g.finished and not self.busy)
        hidden = self.style.currentData() == "random" and not self.revealed
        self.btn_reveal.setVisible(hidden)
        self.table.set_game(g, not hidden)
        lines = []
        for entry in g.log:
            muted = entry.startswith(("Префлоп", "Флоп", "Тёрн", "Ривер")) and "Вы" not in entry
            lines.append(f'<p style="margin:0 0 3px 0; color:{pal.secondary if muted else pal.text};">{escape(entry)}</p>')
        self.log.setHtml("".join(lines))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if g.finished and not self.counted:
            self.counted = True
            self.hands += 1
            self.total += g.net
            if g.decisions:
                ok = sum(d.matched for d in g.decisions)
                items = [(d.matched, f"{STREET_RU[d.street]}: вы — {d.chosen}; тренер — {d.advice.headline}")
                         for d in g.decisions]
                self.coach.add_extra(summary_html(items, f"Итог раздачи: по совету {ok} из {len(g.decisions)}", g.result))
        self.score.setText(f"Сыграно раздач: {self.hands} · итог {self.total:+g} bb" if self.hands
                           else "Тренер оценивает каждое ваше решение")
        notes = []
        for i, p in enumerate(g.players):
            if i == H:
                continue
            tag = p.stats.adjust().label
            tag_html = (f'<span style="color:{pal.accent}; font-weight:600;">{escape(tag)}</span>' if tag
                        else f'<span style="color:{pal.tertiary};">пока без выводов</span>')
            notes.append(f'<p style="margin:0 0 7px 0;"><span style="font-weight:600;">{escape(p.name)}</span> · '
                         f'{tag_html}<br><span style="color:{pal.secondary}; font-size:12px;">'
                         f'{escape(p.stats.summary())}</span></p>')
        self.profile.setHtml("".join(notes))
        self.update_overlay()


class Overlay(QWidget):
    """Компактное окно тренера поверх остальных окон. Данные получает только от своего стола."""

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(340)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label("Тренер", "caption"))
        self.pill = Pill()
        head.addWidget(self.pill)
        head.addStretch()
        close = QPushButton("Закрыть")
        close.setObjectName("plain")
        close.clicked.connect(self.hide)
        head.addWidget(close)
        lay.addLayout(head)
        self.headline = label("—", "section", wrap=True)
        self.headline.setStyleSheet("font-size:17px;")
        self.bar = EquityBar()
        self.body = label("", "caption", wrap=True)
        self.live = Switch("Подсказывать до хода")
        lay.addWidget(self.headline)
        lay.addWidget(self.bar)
        lay.addWidget(self.body)
        lay.addWidget(self.live)
        self._drag = None
        theme.changed.connect(self.update)

    def paintEvent(self, _e) -> None:
        pal = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(color(pal.separator))
        p.setBrush(QColor(pal.panel))
        p.drawRoundedRect(r, 14, 14)
        p.end()

    def show_advice(self, title: str, a: Analysis, status: str = "", kind: str = "neutral") -> None:
        self.headline.setText(title)
        self.pill.show_status(status, kind)
        self.bar.set_values(a.equity, a.required)
        self.body.setText("\n\n".join(a.reasons[:2]))
        self._fit()

    def show_text(self, title: str) -> None:
        self.headline.setText(title)
        self.pill.show_status("")
        self.bar.set_values(None, 0)
        self.body.setText("")
        self._fit()

    def _fit(self) -> None:
        """Высота окна под текст: перенос строк зависит от ширины, поэтому считаем по ней."""
        lay = self.layout()
        lay.activate()
        self.resize(self.width(), lay.totalHeightForWidth(self.width()))

    def mousePressEvent(self, e) -> None:
        self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = None


# ---------- разбор сессии ----------

class SessionTab(QWidget):
    """Разбор сыгранной сессии по hand history (формат PokerStars)."""

    SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "pokerstars_sample.txt"

    def __init__(self):
        super().__init__()
        self.items: list = []
        self.hands: list = []
        self.loaded = False
        root = page(self)
        head, self.summary = page_header("Разбор сессии", "Загрузите hand history PokerStars — файл или текст из буфера")
        for text, fn, name in (("Пример", self.load_sample, "plain"), ("Вставить", self.paste, ""),
                               ("Открыть файл…", self.open_file, "primary")):
            b = QPushButton(text)
            b.setObjectName(name)
            b.clicked.connect(fn)
            head.addWidget(b, 0, Qt.AlignmentFlag.AlignBottom)
        root.addLayout(head)
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        lbox, ll = panel("Решения")
        self.only_errors = Switch("Только расхождения с тренером")
        ll.addWidget(self.only_errors)
        self.list = QListWidget()
        self.list.setIconSize(QSize(10, 10))
        self.list.setWordWrap(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ll.addWidget(self.list, 1)
        lbox.setFixedWidth(420)
        body.addWidget(lbox)

        right = QVBoxLayout()
        right.setSpacing(16)
        cbox, cl = panel()
        crow = QHBoxLayout()
        crow.setSpacing(24)
        for title, attr, slots in (("Рука", "hero_row", 2), ("Борд", "board_row", 5)):
            col = QVBoxLayout()
            col.setSpacing(6)
            col.addWidget(label(title, "fieldLabel"))
            row = CardRow(52, slots)
            setattr(self, attr, row)
            col.addWidget(row)
            crow.addLayout(col)
        crow.addStretch()
        self.result = label("", "caption")
        crow.addWidget(self.result, 0, Qt.AlignmentFlag.AlignBottom)
        cl.addLayout(crow)
        right.addWidget(cbox)
        mid = QHBoxLayout()
        mid.setSpacing(16)
        self.coach = CoachPanel("Разбор тренера", "Выберите решение в списке")
        mid.addWidget(self.coach, 1)
        rbox, rl = panel("Диапазон оппонента")
        self.matrix = RangeMatrix(22)
        rl.addWidget(self.matrix)
        rl.addStretch()
        mid.addWidget(rbox, 0)
        right.addLayout(mid, 1)
        body.addLayout(right, 1)
        self.list.currentRowChanged.connect(self.show_item)
        self.only_errors.toggled.connect(self.fill)
        theme.changed.connect(self.fill)

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Hand history", "", "Текст (*.txt);;Все файлы (*)")
        if path:
            self.load_text(Path(path).read_text(encoding="utf-8", errors="replace"))

    def paste(self) -> None:
        self.load_text(QApplication.clipboard().text())

    def load_sample(self) -> None:
        self.load_text(self.SAMPLE.read_text(encoding="utf-8"))

    def load_text(self, text: str) -> None:
        self.hands = review_session(text)
        self.loaded = True
        self.fill()

    def fill(self) -> None:
        pal = current()
        keep = self.list.currentRow()
        self.list.clear()
        self.items = []
        if not self.hands:
            if self.loaded:
                self.summary.setText("Раздач не найдено. Нужен текст hand history PokerStars (Hold'em No Limit).")
            return
        rated = [d for h in self.hands for d in h.decisions if d.matched is not None]
        ok = sum(d.matched for d in rated)
        net = sum(h.net_bb for h in self.hands)
        self.summary.setText(f"Раздач: {len(self.hands)} · итог {net:+g} bb · по совету тренера {ok} из {len(rated)}"
                             f" ({100 * ok / max(len(rated), 1):.0f}%)")
        icons = {True: dot_icon(pal.green), False: dot_icon(pal.red), None: dot_icon(pal.tertiary)}
        for n, h in enumerate(self.hands, 1):
            for d in h.decisions:
                if self.only_errors.isChecked() and d.matched is not False:
                    continue
                tail = f"тренер: {d.advice.headline}" if d.advice else "без оценки"
                cards = " ".join(c[0].replace("T", "10") + {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}[c[1]]
                                 for c in h.hero_cards)
                item = QListWidgetItem(icons[d.matched], f"{STREET_RU[d.street]}: {d.actual}\n"
                                                         f"№{n} · {cards} · {h.hero_pos} · {tail}")
                self.list.addItem(item)
                self.items.append((h, d))
        if self.items:
            self.list.setCurrentRow(min(max(keep, 0), len(self.items) - 1))

    def show_item(self, row: int) -> None:
        if not 0 <= row < len(self.items):
            return
        h, d = self.items[row]
        self.hero_row.set_cards(h.hero_cards)
        self.board_row.set_cards(d.state.board if d.state else [])
        self.result.setText(f"Итог раздачи {h.net_bb:+g} bb")
        if d.advice is None:
            self.coach.pill.show_status("Без оценки")
            self.coach.show_message("Тренер не оценивает это решение", d.note)
            self.matrix.set_data({}, None)
            return
        street = STREET_RU[d.street]
        if d.matched:
            self.coach.pill.show_status(f"{street}: верно", "good")
            self.coach.show_analysis(d.advice, f"{d.actual[:1].upper()}{d.actual[1:]} — как у тренера")
        else:
            self.coach.pill.show_status(f"{street}: вы — {d.actual}", "bad")
            self.coach.show_analysis(d.advice, f"Тренер: {d.advice.headline}")
        self.matrix.set_data(ranges.matrix(d.advice.villain), charts.hand_key(h.hero_cards))


# ---------- обучение ----------

class LearnTab(QWidget):
    """Учебник: список глав слева, текст справа."""

    def __init__(self):
        super().__init__()
        root = page(self)
        head, _sub = page_header("Обучение", "Теория покера от правил до устройства солвера")
        root.addLayout(head)
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)
        cbox, cl = panel(margins=8)
        self.chapters = QListWidget()
        for title, _html in theory.CHAPTERS:
            self.chapters.addItem(title)
        self.chapters.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chapters.setWordWrap(True)
        cl.addWidget(self.chapters)
        cbox.setFixedWidth(300)
        body.addWidget(cbox)
        tbox, tl = panel(margins=28)
        self.text = text_view()
        tl.addWidget(self.text)
        body.addWidget(tbox, 1)
        self.chapters.currentRowChanged.connect(self.render)
        theme.changed.connect(lambda: self.render(self.chapters.currentRow()))
        self.chapters.setCurrentRow(0)

    def render(self, i: int) -> None:
        if i >= 0:
            self.text.setHtml(theory.html(i, current()))


# ---------- окно ----------

class Nav(QStackedWidget):
    """Страницы приложения; setCurrentIndex синхронизирован с боковой панелью."""

    def __init__(self, sidebar: QListWidget):
        super().__init__()
        self.sidebar = sidebar
        sidebar.currentRowChanged.connect(super().setCurrentIndex)

    def setCurrentIndex(self, i: int) -> None:  # noqa: N802 — имя из Qt
        self.sidebar.setCurrentRow(i)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Покерный тренер")
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(216)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(12, 22, 12, 16)
        sl.setSpacing(14)
        sl.addWidget(label("Покерный тренер", "appTitle"))
        self.nav = QListWidget()
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setObjectName("nav")
        for name in ("Игра", "Разбор ситуации", "Разбор сессии", "Обучение"):
            self.nav.addItem(name)
        sl.addWidget(self.nav, 1)
        sl.addWidget(label("Оформление", "sidebarFooter"))
        self.appearance = Segmented(["Авто", "Светлое", "Тёмное"], small=True)
        self.appearance.changed.connect(lambda i: theme.apply(["auto", "light", "dark"][i]))
        sl.addWidget(self.appearance)
        sl.addWidget(label("Безлимитный холдем · 100 bb", "sidebarFooter", wrap=True))
        lay.addWidget(side)

        self.tabs = Nav(self.nav)
        self.game_tab = GameTab()
        self.trainer = TrainerTab()
        self.session = SessionTab()
        self.learn = LearnTab()
        for w in (self.game_tab, self.trainer, self.session, self.learn):
            self.tabs.addWidget(w)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.nav.setCurrentRow(0)
        self.resize(1360, 860)
        self.setMinimumSize(1180, 760)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Покерный тренер")
    theme.apply("auto")
    try:
        app.styleHints().colorSchemeChanged.connect(lambda _s: theme.apply() if theme.mode == "auto" else None)
    except AttributeError:
        pass
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
