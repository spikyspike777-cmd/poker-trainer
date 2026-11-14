"""Окно приложения. Запуск: python -m poker_coach.ui"""
from __future__ import annotations

import sys

from pydantic import ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QRadioButton, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

from . import charts, ranges
from .coach import Analysis, analyze, explain
from .scenarios import random_state
from .state import LINE_LABELS, POSITIONS, PREFLOP_DECISIONS, RANKS, GameState, PreflopLine

SUIT_SYM = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_COLOR = {"s": "#1d2530", "h": "#c62828", "d": "#1565c0", "c": "#2e7d32"}
RANKS_DESC = RANKS[::-1]

STYLE = """
QWidget { background: #17212b; color: #e6edf3; font-size: 13px; }
QGroupBox { border: 1px solid #2f3d4c; border-radius: 6px; margin-top: 10px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #8fa3b8; }
QPushButton { background: #263545; border: 1px solid #3a4c5f; border-radius: 4px; padding: 6px 10px; }
QPushButton:hover { background: #2f4257; }
QPushButton#primary { background: #2e7d32; border-color: #2e7d32; font-weight: bold; }
QPushButton:disabled { color: #5c6b7a; }
QComboBox, QDoubleSpinBox, QTextEdit { background: #101820; border: 1px solid #2f3d4c; border-radius: 4px; padding: 4px; }
QTabBar::tab { padding: 8px 16px; background: #101820; }
QTabBar::tab:selected { background: #17212b; border-bottom: 2px solid #4caf50; }
"""


def card_html(card: str, size: int = 22) -> str:
    return (f'<span style="background:#f5f5f5; color:{SUIT_COLOR[card[1]]}; font-size:{size}px; '
            f'font-weight:bold;">&nbsp;{card[0]}{SUIT_SYM[card[1]]}&nbsp;</span>')


class CardPicker(QGroupBox):
    """Сетка из 52 карт. Клик добавляет карту в руку или на борд, повторный — убирает."""

    changed = Signal()

    def __init__(self):
        super().__init__("Карты")
        self.hero: list[str] = []
        self.board: list[str] = []
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.to_hero, self.to_board = QRadioButton("Моя рука"), QRadioButton("Борд")
        self.to_hero.setChecked(True)
        row.addWidget(QLabel("Клик кладёт карту в:"))
        row.addWidget(self.to_hero)
        row.addWidget(self.to_board)
        row.addStretch()
        lay.addLayout(row)
        grid = QGridLayout()
        grid.setSpacing(2)
        self.buttons: dict[str, QPushButton] = {}
        for r, suit in enumerate("shdc"):
            for c, rank in enumerate(RANKS_DESC):
                b = QPushButton(rank + SUIT_SYM[suit])
                b.setFixedSize(34, 30)
                b.clicked.connect(lambda _=False, card=rank + suit: self.toggle(card))
                self.buttons[rank + suit] = b
                grid.addWidget(b, r, c)
        lay.addLayout(grid)
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setMinimumHeight(40)
        lay.addWidget(self.summary)
        self.refresh()

    def toggle(self, card: str) -> None:
        for lst in (self.hero, self.board):
            if card in lst:
                lst.remove(card)
                break
        else:
            if self.to_hero.isChecked() and len(self.hero) < 2:
                self.hero.append(card)
                if len(self.hero) == 2:
                    self.to_board.setChecked(True)
            elif self.to_board.isChecked() and len(self.board) < 5:
                self.board.append(card)
        self.refresh()
        self.changed.emit()

    def set_cards(self, hero: list[str], board: list[str]) -> None:
        self.hero, self.board = list(hero), list(board)
        self.refresh()

    def refresh(self) -> None:
        for card, b in self.buttons.items():
            base = f"color:{SUIT_COLOR[card[1]]}; background:#e9edf1; font-weight:bold; padding:0;"
            if card in self.hero:
                base = "color:white; background:#2e7d32; font-weight:bold; padding:0;"
            elif card in self.board:
                base = "color:white; background:#b26a00; font-weight:bold; padding:0;"
            b.setStyleSheet(base)
        hero = " ".join(card_html(c) for c in self.hero) or "—"
        board = " ".join(card_html(c) for c in self.board) or "— (префлоп)"
        self.summary.setText(f"Рука: {hero} &nbsp;&nbsp;&nbsp; Борд: {board}")


class EquityBar(QWidget):
    """Полоса «ваше эквити» с отметкой эквити, нужного для колла."""

    def __init__(self):
        super().__init__()
        self.equity: float | None = None
        self.required = 0.0
        self.setFixedHeight(46)

    def set_values(self, equity: float | None, required: float) -> None:
        self.equity, self.required = equity, required
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        w, top, h = self.width() - 2, 4, 20
        p.fillRect(1, top, w, h, QColor("#101820"))
        if self.equity is not None:
            ok = self.equity >= self.required
            p.fillRect(1, top, int(w * self.equity), h, QColor("#2e7d32" if ok else "#b23b3b"))
            p.setPen(QColor("#e6edf3"))
            p.drawText(8, top + 15, f"Ваше эквити {self.equity:.0%}")
        if self.required > 0:
            x = 1 + int(w * self.required)
            p.setPen(QPen(QColor("#ffd54f"), 2))
            p.drawLine(x, top - 3, x, top + h + 3)
            p.drawText(min(x + 4, w - 150), top + h + 16, f"нужно для колла {self.required:.0%}")
        p.end()


class RangeMatrix(QWidget):
    """Сетка 13×13: диапазон оппонента. Одномастные — выше диагонали, разномастные — ниже."""

    CELL = 27

    def __init__(self):
        super().__init__()
        self.weights: dict[str, float] = {}
        self.hero_key: str | None = None
        self.setFixedSize(13 * self.CELL + 1, 13 * self.CELL + 1)

    def set_data(self, weights: dict[str, float], hero_key: str | None) -> None:
        self.weights, self.hero_key = weights, hero_key
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        f = QFont(self.font())
        f.setPixelSize(10)
        p.setFont(f)
        c = self.CELL
        for i, r1 in enumerate(RANKS_DESC):
            for j, r2 in enumerate(RANKS_DESC):
                key = r1 + r2 if i == j else (r1 + r2 + "s" if i < j else r2 + r1 + "o")
                wgt = self.weights.get(key, 0.0)
                col = QColor("#1f2b38")
                if wgt > 0:
                    col = QColor(46, 125, 50, int(90 + 165 * wgt))
                p.fillRect(j * c, i * c, c - 1, c - 1, QColor("#101820"))
                p.fillRect(j * c, i * c, c - 1, c - 1, col)
                if key == self.hero_key:
                    p.setPen(QPen(QColor("#ffd54f"), 2))
                    p.drawRect(j * c + 1, i * c + 1, c - 3, c - 3)
                p.setPen(QColor("#e6edf3" if wgt > 0 else "#5c6b7a"))
                p.drawText(j * c, i * c, c - 1, c - 1, Qt.AlignCenter, key)
        p.end()


class TrainerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.analysis: Analysis | None = None
        root = QHBoxLayout(self)
        left, right = QVBoxLayout(), QVBoxLayout()
        root.addLayout(left, 0)
        root.addLayout(right, 1)

        # --- ввод ---
        self.picker = CardPicker()
        left.addWidget(self.picker)
        box = QGroupBox("Ситуация")
        form = QFormLayout(box)
        self.line = QComboBox()
        for ln, label in LINE_LABELS.items():
            self.line.addItem(("Префлоп: " if ln in PREFLOP_DECISIONS else "Постфлоп: ") + label, ln)
        self.hero_pos, self.vil_pos = QComboBox(), QComboBox()
        self.hero_pos.addItems(POSITIONS)
        self.vil_pos.addItems(POSITIONS)
        self.pot, self.to_call, self.stack = (self._spin(v) for v in (1.5, 0, 100))
        form.addRow("Что было на префлопе", self.line)
        form.addRow("Моя позиция", self.hero_pos)
        form.addRow("Позиция оппонента", self.vil_pos)
        form.addRow("Банк сейчас, bb", self.pot)
        form.addRow("Нужно доставить, bb", self.to_call)
        form.addRow("Мой стек, bb", self.stack)
        hint = QLabel("Банк — вместе со ставкой оппонента. Если оппонент чекнул, «доставить» = 0.")
        hint.setStyleSheet("color:#8fa3b8; font-size:11px;")
        hint.setWordWrap(True)
        form.addRow(hint)
        left.addWidget(box)

        btns = QHBoxLayout()
        self.btn_random = QPushButton("Случайная раздача")
        self.btn_clear = QPushButton("Сбросить")
        btns.addWidget(self.btn_random)
        btns.addWidget(self.btn_clear)
        left.addLayout(btns)

        quiz = QGroupBox("Ваше решение")
        ql = QVBoxLayout(quiz)
        self.quiz_mode = QCheckBox("Сначала отвечаю сам, потом смотрю разбор")
        self.quiz_mode.setChecked(True)
        ql.addWidget(self.quiz_mode)
        row = QHBoxLayout()
        self.answer_btns = {}
        for key, text in (("fold", "Фолд"), ("passive", "Чек / Колл"), ("aggressive", "Бет / Рейз")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=key: self.run(k))
            self.answer_btns[key] = b
            row.addWidget(b)
        ql.addLayout(row)
        self.btn_analyze = QPushButton("Показать разбор")
        self.btn_analyze.setObjectName("primary")
        ql.addWidget(self.btn_analyze)
        left.addWidget(quiz)
        left.addStretch()

        # --- разбор ---
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.headline = QLabel("Выставьте раздачу слева или нажмите «Случайная раздача»")
        self.headline.setStyleSheet("font-size:24px; font-weight:bold;")
        self.headline.setWordWrap(True)
        self.bar = EquityBar()
        right.addWidget(self.verdict)
        right.addWidget(self.headline)
        right.addWidget(self.bar)
        mid = QHBoxLayout()
        rbox = QGroupBox("Диапазон оппонента")
        rl = QVBoxLayout(rbox)
        self.matrix = RangeMatrix()
        self.range_label = QLabel("—")
        self.range_label.setWordWrap(True)
        self.range_label.setStyleSheet("color:#8fa3b8; font-size:11px;")
        rl.addWidget(self.matrix)
        rl.addWidget(self.range_label)
        rl.addStretch()
        rbox.setFixedWidth(13 * RangeMatrix.CELL + 30)
        mid.addWidget(rbox, 0)
        tbox = QGroupBox("Разбор тренера")
        tl = QVBoxLayout(tbox)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        tl.addWidget(self.text)
        mid.addWidget(tbox, 1)
        right.addLayout(mid, 1)

        self.btn_analyze.clicked.connect(lambda: self.run(None))
        self.btn_random.clicked.connect(self.load_random)
        self.btn_clear.clicked.connect(self.clear)
        self.line.currentIndexChanged.connect(self._sync)
        self.picker.changed.connect(self.reset_result)
        self.quiz_mode.toggled.connect(self._sync)
        self._sync()

    @staticmethod
    def _spin(value: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0, 1000)
        s.setDecimals(1)
        s.setSingleStep(0.5)
        s.setValue(value)
        return s

    def _sync(self) -> None:
        self.vil_pos.setEnabled(self.line.currentData() != PreflopLine.UNOPENED)
        for b in self.answer_btns.values():
            b.setEnabled(self.quiz_mode.isChecked())

    def state(self) -> GameState:
        line = self.line.currentData()
        return GameState(
            hero_cards=self.picker.hero, board=self.picker.board, hero_pos=self.hero_pos.currentText(),
            villain_pos=None if line == PreflopLine.UNOPENED else self.vil_pos.currentText(),
            preflop_line=line, pot=self.pot.value(), to_call=self.to_call.value(), stack=self.stack.value())

    def set_state(self, s: GameState) -> None:
        self.picker.set_cards(s.hero_cards, s.board)
        self.line.setCurrentIndex(self.line.findData(s.preflop_line))
        self.hero_pos.setCurrentText(s.hero_pos)
        if s.villain_pos:
            self.vil_pos.setCurrentText(s.villain_pos)
        self.pot.setValue(s.pot)
        self.to_call.setValue(s.to_call)
        self.stack.setValue(s.stack)
        self.reset_result()

    def load_random(self) -> None:
        self.set_state(random_state())
        self.headline.setText("Ваш ход. Что будете делать?")

    def clear(self) -> None:
        self.picker.set_cards([], [])
        self.picker.to_hero.setChecked(True)
        self.reset_result()

    def reset_result(self) -> None:
        self.analysis = None
        self.verdict.setText("")
        self.headline.setText("Ваш ход. Что будете делать?")
        self.bar.set_values(None, 0)
        self.matrix.set_data({}, None)
        self.range_label.setText("—")
        self.text.clear()

    def run(self, answer: str | None) -> None:
        try:
            st = self.state()
            a = analyze(st)
        except ValidationError as e:
            msgs = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
            self._error(msgs)
            return
        except ValueError as e:
            self._error(str(e))
            return
        self.analysis = a
        self.headline.setText(a.headline)
        self.bar.set_values(a.equity, a.required)
        self.matrix.set_data(ranges.matrix(a.villain), charts.hand_key(st.hero_cards))
        if a.villain:
            self.range_label.setText(f"{a.villain.label}. Комбинаций: {len(a.villain.combos)}. "
                                     "Жёлтая рамка — ваша рука.")
        else:
            self.range_label.setText("До вас все сбросили — оппонента пока нет.")
        self.text.setPlainText(explain(a))
        if answer is None:
            self.verdict.setText("")
            return
        kind = {"fold": "fold", "check": "passive", "call": "passive"}.get(a.action, "aggressive")
        ok = kind == answer
        self.verdict.setText("✔ Совпало с советом тренера" if ok else "✘ Тренер сыграл бы иначе")
        self.verdict.setStyleSheet(f"font-size:15px; font-weight:bold; color:{'#66bb6a' if ok else '#ef5350'};")

    def _error(self, msg: str) -> None:
        self.reset_result()
        self.headline.setText("Проверьте ввод")
        self.text.setPlainText(msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Покерный тренер — 6-max NLHE, 100bb")
        tabs = QTabWidget()
        self.trainer = TrainerTab()
        tabs.addTab(self.trainer, "Тренажёр")
        soon = QLabel("Разбор сессии по скриншотам и записи экрана — следующий этап.")
        soon.setAlignment(Qt.AlignCenter)
        tabs.addTab(soon, "Разбор сессии")
        self.setCentralWidget(tabs)
        self.resize(1180, 760)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
