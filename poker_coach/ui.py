"""Окно приложения. Запуск: python -m poker_coach.ui"""
from __future__ import annotations

import random
import sys

from pydantic import ValidationError
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMainWindow, QPushButton, QRadioButton,
    QTabWidget, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from . import charts, ranges, theory
from .coach import Analysis, analyze, explain
from .bots import STYLES
from .engine import STREET_RU, Game, H
from .hh import review_session
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


class GameTab(QWidget):
    """Стол на 2–6 игроков: решение на каждой улице и комментарий тренера."""

    def __init__(self):
        super().__init__()
        self.game = Game(n_players=6, styles="random")
        self.hands = 0
        self.total = 0.0
        self.counted = False
        self.revealed = False
        root = QHBoxLayout(self)
        left, right = QVBoxLayout(), QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 1)

        obox = QGroupBox("Стол и соперники")
        ol = QVBoxLayout(obox)
        orow = QHBoxLayout()
        self.count = QComboBox()
        for n in range(2, 7):
            self.count.addItem(f"{n} игроков" if n > 4 else f"{n} игрока", n)
        self.count.setCurrentIndex(4)
        self.style = QComboBox()
        self.style.addItem("Случайные характеры (скрыты)", "random")
        for st in STYLES.values():
            self.style.addItem(f"Все: {st.name.lower()}", st.key)
        self.btn_reveal = QPushButton("Раскрыть характеры")
        orow.addWidget(self.count)
        orow.addWidget(self.style, 1)
        orow.addWidget(self.btn_reveal)
        ol.addLayout(orow)
        crow = QHBoxLayout()
        self.use_profile = QCheckBox("Тренер учитывает профили ботов")
        self.use_profile.setChecked(True)
        self.adaptive = QCheckBox("Боты подстраиваются под меня")
        crow.addWidget(self.use_profile)
        crow.addWidget(self.adaptive)
        crow.addStretch()
        ol.addLayout(crow)
        left.addWidget(obox)

        table = QGroupBox("Стол")
        tl = QVBoxLayout(table)
        self.table = QLabel()
        self.table.setTextFormat(Qt.RichText)
        self.table.setMinimumHeight(250)
        self.table.setAlignment(Qt.AlignTop)
        tl.addWidget(self.table)
        left.addWidget(table)
        self.actions = QHBoxLayout()
        left.addLayout(self.actions)
        row = QHBoxLayout()
        self.btn_hint = QPushButton("Подсказка тренера")
        self.btn_overlay = QPushButton("Оверлей")
        self.btn_overlay.setToolTip("Компактное окно тренера поверх остальных окон")
        self.overlay = Overlay()
        self.btn_new = QPushButton("Новая раздача")
        self.btn_new.setObjectName("primary")
        row.addWidget(self.btn_hint)
        row.addWidget(self.btn_overlay)
        row.addWidget(self.btn_new)
        left.addLayout(row)
        lbox = QGroupBox("Ход раздачи")
        ll = QVBoxLayout(lbox)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        ll.addWidget(self.log)
        left.addWidget(lbox, 1)
        self.score = QLabel()
        self.score.setStyleSheet("color:#8fa3b8;")
        left.addWidget(self.score)

        self.verdict = QLabel("Сделайте ход — тренер прокомментирует решение.")
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet("font-size:17px; font-weight:bold;")
        self.bar = EquityBar()
        right.addWidget(self.verdict)
        right.addWidget(self.bar)
        cbox = QGroupBox("Комментарий тренера")
        cl = QVBoxLayout(cbox)
        self.coach = QTextEdit()
        self.coach.setReadOnly(True)
        cl.addWidget(self.coach)
        right.addWidget(cbox, 2)
        pbox = QGroupBox("Что тренер заметил за соперниками")
        pl = QVBoxLayout(pbox)
        self.profile = QLabel()
        self.profile.setWordWrap(True)
        self.profile.setTextFormat(Qt.RichText)
        self.profile.setStyleSheet("font-size:11px;")
        pl.addWidget(self.profile)
        right.addWidget(pbox, 0)

        self.btn_new.clicked.connect(self.new_hand)
        self.btn_hint.clicked.connect(self.hint)
        self.btn_overlay.clicked.connect(self.toggle_overlay)
        self.count.currentIndexChanged.connect(self.reseat)
        self.style.currentIndexChanged.connect(self.reseat)
        self.use_profile.toggled.connect(lambda on: setattr(self.game, "use_profile", on))
        self.adaptive.toggled.connect(lambda on: setattr(self.game, "adaptive", on))
        self.btn_reveal.clicked.connect(self.reveal)
        self.overlay.live.toggled.connect(lambda _on: self.update_overlay())
        self.refresh()

    def reseat(self) -> None:
        """Сменить число игроков или характеры: профили и счёт начинаются заново."""
        self.revealed = False
        self.game.setup(self.count.currentData(), self.style.currentData())
        self.hands, self.total = 0, 0.0
        self.new_hand_ui()

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
        """Оверлей показывает оценку последнего решения, а при включённой галочке — совет на текущий ход."""
        if not self.overlay.isVisible():
            return
        g = self.game
        if not g.finished and self.overlay.live.isChecked():
            a = g.advice(H)
            self.overlay.show_advice(f"Совет: {a.headline}", a, "#ffd54f")
        elif g.last is not None:
            d = g.last
            title = (f"✔ {d.chosen} — верно" if d.matched else f"✘ Вы — {d.chosen}, тренер — {d.advice.headline}")
            self.overlay.show_advice(title, d.advice, "#66bb6a" if d.matched else "#ef5350")
        else:
            self.overlay.show_text("Ваш ход" if not g.finished else g.result)

    def new_hand(self) -> None:
        self.game.new_hand()
        self.new_hand_ui()

    def new_hand_ui(self) -> None:
        self.counted = False
        self.verdict.setText("Сделайте ход — тренер прокомментирует решение.")
        self.verdict.setStyleSheet("font-size:17px; font-weight:bold;")
        self.bar.set_values(None, 0)
        self.coach.clear()
        self.refresh()

    def hint(self) -> None:
        a = self.game.advice(H)
        self.verdict.setText(f"Подсказка: {a.headline}")
        self.verdict.setStyleSheet("font-size:17px; font-weight:bold; color:#ffd54f;")
        self.bar.set_values(a.equity, a.required)
        self.coach.setPlainText(explain(a))

    def act(self, opt) -> None:
        g = self.game
        g.hero_act(opt)
        d = g.last
        if d.matched:
            self.verdict.setText(f"✔ {STREET_RU[d.street]}: {d.chosen} — совпало с тренером")
            color = "#66bb6a"
        else:
            self.verdict.setText(f"✘ {STREET_RU[d.street]}: вы — {d.chosen}, тренер — {d.advice.headline}")
            color = "#ef5350"
        self.verdict.setStyleSheet(f"font-size:17px; font-weight:bold; color:{color};")
        self.bar.set_values(d.advice.equity, d.advice.required)
        self.coach.setPlainText(explain(d.advice))
        self.refresh()

    def _style_name(self, p) -> str:
        hidden = self.style.currentData() == "random" and not self.revealed
        return "характер скрыт" if hidden else p.style.name.lower()

    def refresh(self) -> None:
        g = self.game
        while self.actions.count():
            w = self.actions.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        for opt in g.options():
            b = QPushButton(opt.label)
            b.setMinimumHeight(38)
            b.clicked.connect(lambda _=False, o=opt: self.act(o))
            self.actions.addWidget(b)
        self.btn_hint.setEnabled(not g.finished)
        self.btn_reveal.setVisible(self.style.currentData() == "random" and not self.revealed)
        back = '<span style="background:#3a4c5f; color:#3a4c5f;">&nbsp;??&nbsp;</span>'
        rows = []
        for i, p in enumerate(g.players):
            if i == H:
                continue
            cards = " ".join(card_html(c, 15) for c in p.cards) if i in g.shown else f"{back} {back}"
            state = "сбросил" if p.folded else (f"ставка {p.inv:g}" if p.inv else "в игре")
            turn = " ◀ ходит" if g.to_act == i else ""
            color = "#5c6b7a" if p.folded else "#e6edf3"
            rows.append(f"<tr style='color:{color};'><td>{p.name}</td><td>&nbsp;{p.pos}&nbsp;</td><td>{cards}</td>"
                        f"<td>&nbsp;стек {p.stack:g}&nbsp;</td><td>{state}{turn}</td>"
                        f"<td style='color:#8fa3b8;'>&nbsp;{self._style_name(p)}</td></tr>")
        hero = g.players[H]
        board = " ".join(card_html(c, 28) for c in g.board) or '<span style="color:#5c6b7a;">борд ещё не открыт</span>'
        mine = f"<br><span style='color:#8fa3b8;'>ваша ставка на улице: {hero.inv:g}</span>" if hero.inv else ""
        self.table.setText(
            f"<table cellpadding='2'>{''.join(rows)}</table>"
            f"<p align='center'><br>{board}<br><br><span style='font-size:18px; color:#ffd54f;'>"
            f"Банк {(g.final_pot if g.finished else g.pot):g} bb</span></p>"
            f"<p align='center'>{' '.join(card_html(c, 28) for c in hero.cards)}<br><br>"
            f"Вы · {hero.pos} · стек {hero.stack:g} bb{mine}</p>")
        self.log.setPlainText("\n".join(g.log))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if g.finished and not self.counted:
            self.counted = True
            self.hands += 1
            self.total += g.net
            if g.decisions:
                ok = sum(d.matched for d in g.decisions)
                lines = [f"{'✔' if d.matched else '✘'} {STREET_RU[d.street]}: вы — {d.chosen}; "
                         f"тренер — {d.advice.headline}" for d in g.decisions]
                self.coach.append(f"\n— Итог раздачи —\n{g.result}\nРешений по совету тренера: "
                                  f"{ok} из {len(g.decisions)}\n" + "\n".join(lines))
        self.score.setText(f"Сыграно раздач: {self.hands} · общий итог: {self.total:+g} bb")
        notes = []
        for i, p in enumerate(g.players):
            if i == H:
                continue
            label = p.stats.adjust().label
            verdict = f"<b>{label}</b>" if label else "отклонений пока не видно"
            notes.append(f"{p.name}: {verdict} <span style='color:#8fa3b8;'>({p.stats.summary()})</span>")
        self.profile.setText("<br>".join(notes))
        self.update_overlay()


class Overlay(QWidget):
    """Компактное окно тренера поверх остальных окон. Данные получает только от своего стола."""

    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setObjectName("overlay")
        self.setStyleSheet("#overlay { border: 1px solid #4caf50; }")
        self.setFixedWidth(360)
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("Тренер · свой стол")
        title.setStyleSheet("color:#8fa3b8; font-size:11px;")
        self.live = QCheckBox("подсказывать до хода")
        self.live.setStyleSheet("font-size:11px;")
        close = QPushButton("x")
        close.setFixedSize(22, 22)
        close.clicked.connect(self.hide)
        head.addWidget(title)
        head.addStretch()
        head.addWidget(self.live)
        head.addWidget(close)
        lay.addLayout(head)
        self.headline = QLabel("—")
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-size:16px; font-weight:bold;")
        self.bar = EquityBar()
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setStyleSheet("font-size:12px;")
        lay.addWidget(self.headline)
        lay.addWidget(self.bar)
        lay.addWidget(self.body)
        self._drag = None

    def show_advice(self, title: str, a: Analysis, color: str = "#e6edf3") -> None:
        self.headline.setText(title)
        self.headline.setStyleSheet(f"font-size:16px; font-weight:bold; color:{color};")
        self.bar.set_values(a.equity, a.required)
        self.body.setText("\n".join(f"• {r}" for r in a.reasons[:3]))
        self.adjustSize()

    def show_text(self, title: str) -> None:
        self.headline.setText(title)
        self.headline.setStyleSheet("font-size:16px; font-weight:bold;")
        self.bar.set_values(None, 0)
        self.body.setText("")
        self.adjustSize()

    def mousePressEvent(self, e) -> None:
        self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = None


class SessionTab(QWidget):
    """Разбор сыгранной сессии по hand history (формат PokerStars)."""

    SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "pokerstars_sample.txt"

    def __init__(self):
        super().__init__()
        self.items: list = []
        root = QHBoxLayout(self)
        left, right = QVBoxLayout(), QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 1)
        row = QHBoxLayout()
        for text, fn in (("Открыть файл…", self.open_file), ("Вставить из буфера", self.paste),
                         ("Пример сессии", self.load_sample)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        left.addLayout(row)
        self.summary = QLabel("Загрузите hand history PokerStars: файл .txt или текст из буфера обмена.")
        self.summary.setWordWrap(True)
        left.addWidget(self.summary)
        self.list = QListWidget()
        self.list.setStyleSheet("QListWidget { background:#101820; border:1px solid #2f3d4c; } "
                                "QListWidget::item { padding:5px; } QListWidget::item:selected { background:#2f4257; }")
        left.addWidget(self.list, 1)
        self.only_errors = QCheckBox("Показывать только расхождения с тренером")
        left.addWidget(self.only_errors)

        self.cards = QLabel()
        self.cards.setTextFormat(Qt.RichText)
        self.cards.setMinimumHeight(40)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet("font-size:17px; font-weight:bold;")
        self.bar = EquityBar()
        right.addWidget(self.cards)
        right.addWidget(self.verdict)
        right.addWidget(self.bar)
        mid = QHBoxLayout()
        rbox = QGroupBox("Диапазон оппонента")
        rl = QVBoxLayout(rbox)
        self.matrix = RangeMatrix()
        rl.addWidget(self.matrix)
        rl.addStretch()
        rbox.setFixedWidth(13 * RangeMatrix.CELL + 30)
        tbox = QGroupBox("Разбор тренера")
        tl = QVBoxLayout(tbox)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        tl.addWidget(self.text)
        mid.addWidget(rbox, 0)
        mid.addWidget(tbox, 1)
        right.addLayout(mid, 1)
        self.hands: list = []
        self.list.currentRowChanged.connect(self.show_item)
        self.only_errors.toggled.connect(self.fill)

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
        self.fill()

    def fill(self) -> None:
        self.list.clear()
        self.items = []
        rated = [d for h in self.hands for d in h.decisions if d.matched is not None]
        skipped = sum(1 for h in self.hands for d in h.decisions if d.matched is None)
        if not self.hands:
            self.summary.setText("Раздач не найдено. Нужен текст hand history PokerStars (Hold'em No Limit).")
            return
        ok = sum(d.matched for d in rated)
        net = sum(h.net_bb for h in self.hands)
        self.summary.setText(
            f"Раздач: {len(self.hands)} · итог {net:+g} bb · решений с оценкой: {len(rated)}, "
            f"по совету тренера {ok} ({100 * ok / max(len(rated), 1):.0f}%) · без оценки: {skipped}")
        for n, h in enumerate(self.hands, 1):
            for d in h.decisions:
                if self.only_errors.isChecked() and d.matched is not False:
                    continue
                mark = "✔" if d.matched else "✘" if d.matched is False else "?"
                tail = f"тренер: {d.advice.headline}" if d.advice else "без оценки"
                self.list.addItem(f"{mark}  №{n} · {' '.join(h.hero_cards)} · {h.hero_pos} · "
                                  f"{STREET_RU[d.street]}: {d.actual} — {tail}")
                self.items.append((h, d))
        if self.items:
            self.list.setCurrentRow(0)

    def show_item(self, row: int) -> None:
        if not 0 <= row < len(self.items):
            return
        h, d = self.items[row]
        board = d.state.board if d.state else []
        self.cards.setText(f"Рука: {' '.join(card_html(c) for c in h.hero_cards)} &nbsp;&nbsp; Борд: "
                           + (" ".join(card_html(c) for c in board) or "— (префлоп)")
                           + f" &nbsp;&nbsp; итог раздачи {h.net_bb:+g} bb")
        if d.advice is None:
            self.verdict.setText("Без оценки")
            self.verdict.setStyleSheet("font-size:17px; font-weight:bold; color:#8fa3b8;")
            self.bar.set_values(None, 0)
            self.matrix.set_data({}, None)
            self.text.setPlainText(d.note)
            return
        if d.matched:
            self.verdict.setText(f"✔ {STREET_RU[d.street]}: {d.actual} — совпало с тренером")
        else:
            self.verdict.setText(f"✘ {STREET_RU[d.street]}: вы — {d.actual}, тренер — {d.advice.headline}")
        self.verdict.setStyleSheet(f"font-size:17px; font-weight:bold; color:{'#66bb6a' if d.matched else '#ef5350'};")
        self.bar.set_values(d.advice.equity, d.advice.required)
        self.matrix.set_data(ranges.matrix(d.advice.villain), charts.hand_key(h.hero_cards))
        self.text.setPlainText(explain(d.advice))


class LearnTab(QWidget):
    """Учебник: список глав слева, текст справа."""

    def __init__(self):
        super().__init__()
        root = QHBoxLayout(self)
        self.chapters = QListWidget()
        self.chapters.setFixedWidth(270)
        self.chapters.setStyleSheet("QListWidget { background:#101820; border:1px solid #2f3d4c; } "
                                    "QListWidget::item { padding:7px; } QListWidget::item:selected { background:#2f4257; }")
        for title, _html in theory.CHAPTERS:
            self.chapters.addItem(title)
        self.text = QTextBrowser()
        self.text.setStyleSheet("QTextBrowser { background:#101820; border:1px solid #2f3d4c; padding:10px; font-size:14px; }")
        root.addWidget(self.chapters)
        root.addWidget(self.text, 1)
        self.chapters.currentRowChanged.connect(lambda i: self.text.setHtml(theory.html(i)) if i >= 0 else None)
        self.chapters.setCurrentRow(0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Покерный тренер — 6-max NLHE, 100bb")
        tabs = QTabWidget()
        self.trainer = TrainerTab()
        self.game_tab = GameTab()
        tabs.addTab(self.game_tab, "Игра против ботов")
        tabs.addTab(self.trainer, "Разбор ситуации")
        self.session = SessionTab()
        tabs.addTab(self.session, "Разбор сессии")
        self.learn = LearnTab()
        tabs.addTab(self.learn, "Обучение")
        self.setCentralWidget(tabs)
        self.tabs = tabs
        self.resize(1180, 760)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
