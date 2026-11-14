"""Разбор раздачи из консоли: python -m poker_coach.cli --hero "As Kh" --pos BTN --line unopened --pot 1.5"""
import argparse

from .coach import analyze, explain
from .state import GameState, PreflopLine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hero", required=True)
    p.add_argument("--board", default="")
    p.add_argument("--pos", required=True)
    p.add_argument("--vpos")
    p.add_argument("--line", required=True, choices=[x.value for x in PreflopLine])
    p.add_argument("--pot", type=float, required=True)
    p.add_argument("--call", type=float, default=0)
    p.add_argument("--stack", type=float, default=100)
    a = p.parse_args()
    st = GameState(hero_cards=a.hero, board=a.board, hero_pos=a.pos, villain_pos=a.vpos,
                   preflop_line=a.line, pot=a.pot, to_call=a.call, stack=a.stack)
    print(explain(analyze(st)))


if __name__ == "__main__":
    main()
