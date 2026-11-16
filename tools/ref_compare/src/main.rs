//! Решает споты из JSON (stdin) открытым солвером postflop-solver и печатает JSON (stdout):
//! стратегию и EV всех рук в корне (ход игрока без позиции) и после его чека (ход игрока в позиции).
use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::time::Instant;

#[derive(Deserialize)]
struct Spot {
    board: Vec<u8>,          // 4 или 5 карт, номер = ранг*4 + масть
    oop: Vec<(u8, u8)>,
    ip: Vec<(u8, u8)>,
    w_oop: Vec<f32>,
    w_ip: Vec<f32>,
    pot: f64,                // в bb
    stack: f64,
    street_sizes: String,    // например "33%,75%"
    river_sizes: String,     // для спотов тёрна
    max_iters: u32,
    target_pct: f32,         // целевая эксплуатируемость, % банка
}

#[derive(Serialize)]
struct NodeOut {
    actions: Vec<(String, f64)>,
    hands: Vec<(u8, u8)>,
    strategy: Vec<Vec<f32>>, // [действие][рука]
    ev: Vec<Vec<f32>>,       // [действие][рука], bb
}

#[derive(Serialize)]
struct Out {
    exploitability_bb: f32,
    seconds: f64,
    oop: NodeOut,
    ip: Option<NodeOut>,
}

const SCALE: f64 = 100.0;

fn node(game: &mut PostFlopGame) -> NodeOut {
    game.cache_normalized_weights();
    let p = game.current_player();
    let hands = game.private_cards(p).to_vec();
    let n = hands.len();
    let acts = game.available_actions();
    let strat = game.strategy();
    let ev = game.expected_values_detail(p);
    let name = |a: &Action| match a {
        Action::Fold => ("fold".to_string(), 0.0),
        Action::Check => ("check".to_string(), 0.0),
        Action::Call => ("call".to_string(), 0.0),
        Action::Bet(x) => ("bet".to_string(), *x as f64 / SCALE),
        Action::Raise(x) => ("raise".to_string(), *x as f64 / SCALE),
        Action::AllIn(x) => ("allin".to_string(), *x as f64 / SCALE),
        _ => ("?".to_string(), 0.0),
    };
    NodeOut {
        actions: acts.iter().map(name).collect(),
        hands,
        strategy: (0..acts.len()).map(|a| strat[a * n..(a + 1) * n].to_vec()).collect(),
        ev: (0..acts.len()).map(|a| ev[a * n..(a + 1) * n].iter().map(|v| v / SCALE as f32).collect()).collect(),
    }
}

fn solve_spot(s: &Spot) -> Out {
    let range = |h: &Vec<(u8, u8)>, w: &Vec<f32>| Range::from_hands_weights(h, w).unwrap();
    let river = s.board.len() == 5;
    let card_config = CardConfig {
        range: [range(&s.oop, &s.w_oop), range(&s.ip, &s.w_ip)],
        flop: [s.board[0], s.board[1], s.board[2]],
        turn: s.board[3],
        river: if river { s.board[4] } else { NOT_DEALT },
    };
    let sizes = |b: &str| BetSizeOptions::try_from((b, "")).unwrap();     // без рейзов — как в нашем дереве сравнения
    let street = sizes(&s.street_sizes);
    let rv = sizes(&s.river_sizes);
    let none = sizes("");
    let tree_config = TreeConfig {
        initial_state: if river { BoardState::River } else { BoardState::Turn },
        starting_pot: (s.pot * SCALE).round() as i32,
        effective_stack: (s.stack * SCALE).round() as i32,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [none.clone(), none.clone()],
        turn_bet_sizes: if river { [none.clone(), none.clone()] } else { [street.clone(), street.clone()] },
        river_bet_sizes: if river { [street.clone(), street.clone()] } else { [rv.clone(), rv.clone()] },
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
    };
    let tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
    game.allocate_memory(false);
    let t = Instant::now();
    let target = (s.pot * SCALE) as f32 * s.target_pct / 100.0;
    let expl = solve(&mut game, s.max_iters, target, false);
    let seconds = t.elapsed().as_secs_f64();
    let oop = node(&mut game);
    let ip = if oop.actions[0].0 == "check" {
        game.play(0);
        Some(node(&mut game))
    } else {
        None
    };
    Out { exploitability_bb: expl / SCALE as f32, seconds, oop, ip }
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let spots: Vec<Spot> = serde_json::from_str(&input).unwrap();
    let outs: Vec<Out> = spots.iter().map(solve_spot).collect();
    println!("{}", serde_json::to_string(&outs).unwrap());
}
