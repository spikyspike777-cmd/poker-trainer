//! CFR+ солвер тёрна и ривера для игры один на один.
//!
//! Игрок 0 — без позиции (ходит первым), игрок 1 — в позиции. Силы рук на каждой возможной
//! карте ривера считает Python (eval7) и передаёт сюда; здесь только дерево и CFR.
//! Вскрытие считается за O(n) на узел: руки отсортированы по силе, а пересечения по картам
//! вычитаются через суммы «по карте».

use pyo3::prelude::*;
use rayon::prelude::*;

const FOLD: u8 = 0;
const CHECK: u8 = 1;
const CALL: u8 = 2;
const BET: u8 = 3;
const RAISE: u8 = 4;
const ALLIN: u8 = 5;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Decision,
    Fold,
    Close, // торговля на улице закончена: вскрытие (ривер) или выход карты ривера (тёрн)
}

#[derive(Clone)]
struct Node {
    kind: Kind,
    player: usize,
    c: [f64; 2], // вложено на улице игроками 0 и 1
    acts: Vec<(u8, f64)>,
    children: Vec<usize>,
}

struct Tree {
    nodes: Vec<Node>,
    pot: f64,
}

#[derive(Clone, Copy)]
struct Cfg {
    raise_mult: f64,
    allow_raise: bool,
}

fn bets(pot: f64, stack: f64, sizes: &[f64], extra: f64) -> Vec<f64> {
    let mut v: Vec<f64> = sizes.iter().map(|f| ((pot * f).min(stack) * 10.0).round() / 10.0).collect();
    if stack <= 1.5 * pot {
        v.push((stack * 100.0).round() / 100.0);
    }
    if extra > 0.0 {
        v.retain(|s| (s - extra).abs() > 0.15 * extra);
        v.push(extra);
    }
    v.retain(|s| *s > 0.0);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    v
}

fn push(t: &mut Tree, kind: Kind, player: usize, c: [f64; 2]) -> usize {
    t.nodes.push(Node { kind, player, c, acts: vec![], children: vec![] });
    t.nodes.len() - 1
}

fn facing(t: &mut Tree, player: usize, c: [f64; 2], raised: bool, stack: f64, cfg: &Cfg) -> usize {
    let id = push(t, Kind::Decision, player, c);
    let (me, opp) = (c[player], c[1 - player]);
    let mut acts = vec![(FOLD, me), (CALL, opp)];
    let f = push(t, Kind::Fold, player, c);
    let s = push(t, Kind::Close, 0, [opp, opp]);
    let mut children = vec![f, s];
    if cfg.allow_raise && !raised && opp < stack - 1e-9 {
        let mut to = (opp * cfg.raise_mult).min(stack);
        if to > 0.6 * stack {
            to = stack;
        }
        let mut nc = c;
        nc[player] = to;
        let ch = facing(t, 1 - player, nc, true, stack, cfg);
        acts.push((if to >= stack - 1e-9 { ALLIN } else { RAISE }, to));
        children.push(ch);
    }
    t.nodes[id].acts = acts;
    t.nodes[id].children = children;
    id
}

fn opening(t: &mut Tree, player: usize, stack: f64, sizes: &[f64], extra: f64, lead: bool, cfg: &Cfg) -> usize {
    let id = push(t, Kind::Decision, player, [0.0, 0.0]);
    let mut acts = vec![(CHECK, 0.0)];
    let first = if player == 0 {
        opening(t, 1, stack, sizes, extra, lead, cfg)
    } else {
        push(t, Kind::Close, 0, [0.0, 0.0])
    };
    let mut children = vec![first];
    if stack > 1e-9 && (player == 1 || lead) {
        for b in bets(t.pot, stack, sizes, extra) {
            let mut c = [0.0, 0.0];
            c[player] = b;
            let ch = facing(t, 1 - player, c, false, stack, cfg);
            acts.push((if b >= stack - 1e-9 { ALLIN } else { BET }, b));
            children.push(ch);
        }
    }
    t.nodes[id].acts = acts;
    t.nodes[id].children = children;
    id
}

fn build(pot: f64, stack: f64, sizes: &[f64], extra: f64, lead: bool, cfg: &Cfg) -> Tree {
    let mut t = Tree { nodes: vec![], pot };
    opening(&mut t, 0, stack, sizes, extra, lead, cfg);
    t
}

/// Накопленные сожаления и суммы стратегий для одного экземпляра дерева.
struct Store {
    regret: Vec<Vec<f64>>,
    ssum: Vec<Vec<f64>>,
}

impl Store {
    fn new(t: &Tree, n: [usize; 2]) -> Store {
        let mut regret = Vec::with_capacity(t.nodes.len());
        let mut ssum = Vec::with_capacity(t.nodes.len());
        for nd in &t.nodes {
            let len = if nd.kind == Kind::Decision { n[nd.player] * nd.acts.len() } else { 0 };
            regret.push(vec![0.0; len]);
            ssum.push(vec![0.0; len]);
        }
        Store { regret, ssum }
    }
}

/// Данные вскрытия для одной карты ривера.
struct Show {
    strength: [Vec<u32>; 2],
    order: [Vec<usize>; 2], // индексы рук по возрастанию силы
    valid: [Vec<bool>; 2],  // рука не содержит карту ривера
}

/// Неизменяемая часть задачи: диапазоны, силы рук, деревья.
struct Ctx {
    hands: [Vec<(u8, u8)>; 2],
    same: [Vec<Option<usize>>; 2], // индекс такой же комбинации в диапазоне оппонента
    shows: Vec<Show>,
    stack: f64,
    street: Tree,
    leaf_of: Vec<usize>,     // узел дерева тёрна -> номер листа (usize::MAX, если не лист)
    river_trees: Vec<Tree>,  // по одному на лист тёрна
    river_weight: f64,
}

/// Где идёт обход: дерево тёрна (с листами-«выходом ривера») или дерево ривера на конкретной карте.
enum Level<'a> {
    Turn { store: &'a mut Store, rivers: &'a mut [Vec<Store>] },
    River { tree: &'a Tree, store: &'a mut Store, ri: usize },
}

type Util = (Vec<f64>, Vec<f64>);

impl Ctx {
    fn n(&self, p: usize) -> usize {
        self.hands[p].len()
    }

    /// Для каждой руки игрока `me`: суммарный вес рук оппонента, не пересекающихся с ней по картам.
    fn compat(&self, me: usize, opp_reach: &[f64]) -> Vec<f64> {
        let opp = 1 - me;
        let mut cs = [0.0f64; 52];
        let mut total = 0.0;
        for (j, &(a, b)) in self.hands[opp].iter().enumerate() {
            let r = opp_reach[j];
            total += r;
            cs[a as usize] += r;
            cs[b as usize] += r;
        }
        self.hands[me]
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| {
                let s = self.same[me][i].map_or(0.0, |j| opp_reach[j]);
                total - cs[a as usize] - cs[b as usize] + s
            })
            .collect()
    }

    /// Для каждой руки игрока `me`: вес рук оппонента слабее и сильнее неё (без пересечений по картам).
    fn win_lose(&self, me: usize, sh: &Show, opp_reach: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let opp = 1 - me;
        let (n_me, n_opp) = (self.n(me), self.n(opp));
        let mut win = vec![0.0; n_me];
        let mut lose = vec![0.0; n_me];
        let (om, oo) = (&sh.order[me], &sh.order[opp]);
        let (sm, so) = (&sh.strength[me], &sh.strength[opp]);
        let mut cw = [0.0f64; 52];
        let mut sum = 0.0;
        let mut j = 0;
        for &h in om.iter() {
            while j < n_opp && so[oo[j]] < sm[h] {
                let k = oo[j];
                let r = opp_reach[k];
                let (a, b) = self.hands[opp][k];
                sum += r;
                cw[a as usize] += r;
                cw[b as usize] += r;
                j += 1;
            }
            let (a, b) = self.hands[me][h];
            win[h] = sum - cw[a as usize] - cw[b as usize];
        }
        let mut cw = [0.0f64; 52];
        let mut sum = 0.0;
        let mut j = n_opp;
        for &h in om.iter().rev() {
            while j > 0 && so[oo[j - 1]] > sm[h] {
                let k = oo[j - 1];
                let r = opp_reach[k];
                let (a, b) = self.hands[opp][k];
                sum += r;
                cw[a as usize] += r;
                cw[b as usize] += r;
                j -= 1;
            }
            let (a, b) = self.hands[me][h];
            lose[h] = sum - cw[a as usize] - cw[b as usize];
        }
        (win, lose)
    }

    fn showdown(&self, ri: usize, pot: f64, c: f64, r0: &[f64], r1: &[f64]) -> Util {
        let sh = &self.shows[ri];
        let mut out: [Vec<f64>; 2] = [vec![], vec![]];
        for me in 0..2 {
            let opp_reach = if me == 0 { r1 } else { r0 };
            let comp = self.compat(me, opp_reach);
            let (win, lose) = self.win_lose(me, sh, opp_reach);
            out[me] = (0..self.n(me))
                .map(|h| {
                    if !sh.valid[me][h] {
                        return 0.0;
                    }
                    let tie = comp[h] - win[h] - lose[h];
                    (pot + c) * win[h] - c * lose[h] + pot / 2.0 * tie
                })
                .collect();
        }
        let [a, b] = out;
        (a, b)
    }

    fn fold(&self, folder: usize, pot: f64, c: [f64; 2], r0: &[f64], r1: &[f64]) -> Util {
        let c0 = self.compat(0, r1);
        let c1 = self.compat(1, r0);
        if folder == 0 {
            (c0.iter().map(|m| -c[0] * m).collect(), c1.iter().map(|m| (pot + c[0]) * m).collect())
        } else {
            (c0.iter().map(|m| (pot + c[1]) * m).collect(), c1.iter().map(|m| -c[1] * m).collect())
        }
    }

    /// Лист тёрна: все карты ривера считаются параллельно, у каждой своё хранилище стратегий.
    fn chance(&self, rivers: &mut [Store], leaf: usize, c: f64, r0: &[f64], r1: &[f64], iter: usize) -> Util {
        let pot = self.street.pot + 2.0 * c;
        let allin = self.stack - c < 1e-9;
        let tree = &self.river_trees[leaf];
        let parts: Vec<Util> = rivers
            .par_iter_mut()
            .enumerate()
            .map(|(ri, store)| {
                let sh = &self.shows[ri];
                let q0: Vec<f64> = r0.iter().zip(&sh.valid[0]).map(|(r, v)| if *v { *r } else { 0.0 }).collect();
                let q1: Vec<f64> = r1.iter().zip(&sh.valid[1]).map(|(r, v)| if *v { *r } else { 0.0 }).collect();
                let (mut a, mut b) = if allin {
                    self.showdown(ri, pot, 0.0, &q0, &q1)
                } else {
                    self.cfr(&mut Level::River { tree, store, ri }, 0, &q0, &q1, iter)
                };
                // полезность ривера считается от банка на начало ривера; переводим к началу тёрна
                let m0 = self.compat(0, &q1);
                let m1 = self.compat(1, &q0);
                for h in 0..a.len() {
                    a[h] = if sh.valid[0][h] { (a[h] - c * m0[h]) * self.river_weight } else { 0.0 };
                }
                for h in 0..b.len() {
                    b[h] = if sh.valid[1][h] { (b[h] - c * m1[h]) * self.river_weight } else { 0.0 };
                }
                (a, b)
            })
            .collect();
        let mut u0 = vec![0.0; self.n(0)];
        let mut u1 = vec![0.0; self.n(1)];
        for (a, b) in parts {
            for (x, y) in u0.iter_mut().zip(a) {
                *x += y;
            }
            for (x, y) in u1.iter_mut().zip(b) {
                *x += y;
            }
        }
        (u0, u1)
    }

    /// Один проход CFR+. iter == 0 — только оценка по средней стратегии, без обновления.
    fn cfr(&self, level: &mut Level, node: usize, r0: &[f64], r1: &[f64], iter: usize) -> Util {
        let (kind, player, c, n_acts, children, pot) = {
            let t = match level {
                Level::Turn { .. } => &self.street,
                Level::River { tree, .. } => *tree,
            };
            let nd = &t.nodes[node];
            (nd.kind, nd.player, nd.c, nd.acts.len(), nd.children.clone(), t.pot)
        };
        match kind {
            Kind::Fold => return self.fold(player, pot, c, r0, r1),
            Kind::Close => {
                return match level {
                    Level::River { ri, .. } => self.showdown(*ri, pot, c[0], r0, r1),
                    Level::Turn { rivers, .. } => {
                        let leaf = self.leaf_of[node];
                        self.chance(&mut rivers[leaf], leaf, c[0], r0, r1, iter)
                    }
                };
            }
            Kind::Decision => {}
        }
        let n_me = self.n(player);
        let strat = {
            let st: &Store = match level {
                Level::Turn { store, .. } => store,
                Level::River { store, .. } => store,
            };
            strategy(st, node, n_me, n_acts, iter == 0)
        };
        let mut utils: Vec<Vec<f64>> = Vec::with_capacity(n_acts);
        let mut other = vec![0.0; self.n(1 - player)];
        let reach_me = if player == 0 { r0 } else { r1 };
        for (a, &ch) in children.iter().enumerate() {
            let q: Vec<f64> = (0..n_me).map(|h| reach_me[h] * strat[h * n_acts + a]).collect();
            let (u0, u1) = if player == 0 { self.cfr(level, ch, &q, r1, iter) } else { self.cfr(level, ch, r0, &q, iter) };
            let (mine, theirs) = if player == 0 { (u0, u1) } else { (u1, u0) };
            for (o, t) in other.iter_mut().zip(theirs) {
                *o += t;
            }
            utils.push(mine);
        }
        let mut mine = vec![0.0; n_me];
        for h in 0..n_me {
            for a in 0..n_acts {
                mine[h] += strat[h * n_acts + a] * utils[a][h];
            }
        }
        if iter > 0 {
            let st: &mut Store = match level {
                Level::Turn { store, .. } => store,
                Level::River { store, .. } => store,
            };
            let w = iter as f64;
            for h in 0..n_me {
                for a in 0..n_acts {
                    let i = h * n_acts + a;
                    st.regret[node][i] = (st.regret[node][i] + utils[a][h] - mine[h]).max(0.0);
                    st.ssum[node][i] += w * reach_me[h] * strat[i];
                }
            }
        }
        if player == 0 { (mine, other) } else { (other, mine) }
    }
}

/// Текущая (по сожалениям) или средняя стратегия в узле: [рука × действие].
fn strategy(st: &Store, node: usize, n_hands: usize, n_acts: usize, average: bool) -> Vec<f64> {
    let src = if average { &st.ssum[node] } else { &st.regret[node] };
    let mut s = vec![0.0; n_hands * n_acts];
    for h in 0..n_hands {
        let row = &src[h * n_acts..(h + 1) * n_acts];
        let tot: f64 = row.iter().map(|x| x.max(0.0)).sum();
        for a in 0..n_acts {
            s[h * n_acts + a] = if tot > 0.0 { row[a].max(0.0) / tot } else { 1.0 / n_acts as f64 };
        }
    }
    s
}

/// Обход от узла дерева текущей улицы: ривер решается сразу, тёрн — с перебором карт ривера.
#[allow(clippy::too_many_arguments)]
fn run(s: &Ctx, is_river: bool, store: &mut Store, rivers: &mut [Vec<Store>], node: usize, r0: &[f64], r1: &[f64], it: usize) -> Util {
    if is_river {
        s.cfr(&mut Level::River { tree: &s.street, store, ri: 0 }, node, r0, r1, it)
    } else {
        s.cfr(&mut Level::Turn { store, rivers }, node, r0, r1, it)
    }
}

/// Решить спот и вернуть стратегию для руки героя.
///
/// strengths[p][r][h] — сила руки h игрока p на карте ривера r (для ривера — один «ривер»);
/// river_cards[r] — номер карты (0..51) или 255, если борд уже полный;
/// path — действия на текущей улице до хода героя: (тип, сумма), тип 1 = чек, 3 = бет;
/// возвращает (действия: (тип, сумма, частота, EV), эквити героя, веса рук оппонента).
#[pyfunction]
#[pyo3(signature = (hands0, hands1, river_cards, strengths0, strengths1, pot, stack, sizes, river_sizes,
                    raise_mult, river_raise, extra_bet, oop_can_lead, iterations, hero_player, hero_index, path))]
#[allow(clippy::too_many_arguments)]
fn solve(
    py: Python<'_>,
    hands0: Vec<(u8, u8)>,
    hands1: Vec<(u8, u8)>,
    river_cards: Vec<u8>,
    strengths0: Vec<Vec<u32>>,
    strengths1: Vec<Vec<u32>>,
    pot: f64,
    stack: f64,
    sizes: Vec<f64>,
    river_sizes: Vec<f64>,
    raise_mult: f64,
    river_raise: bool,
    extra_bet: f64,
    oop_can_lead: bool,
    iterations: usize,
    hero_player: usize,
    hero_index: usize,
    path: Vec<(u8, f64)>,
) -> PyResult<(Vec<(u8, f64, f64, f64)>, f64, Vec<f64>)> {
    let cfg = Cfg { raise_mult, allow_raise: true };
    let river_cfg = Cfg { raise_mult, allow_raise: river_raise };
    let is_river = river_cards.len() == 1 && river_cards[0] == 255;
    let n = [hands0.len(), hands1.len()];
    let hands = [hands0, hands1];
    let mut same: [Vec<Option<usize>>; 2] = [vec![None; n[0]], vec![None; n[1]]];
    for (i, h) in hands[0].iter().enumerate() {
        if let Some(j) = hands[1].iter().position(|g| (g.0 == h.0 && g.1 == h.1) || (g.0 == h.1 && g.1 == h.0)) {
            same[0][i] = Some(j);
            same[1][j] = Some(i);
        }
    }
    let mut shows = Vec::with_capacity(river_cards.len());
    for (ri, &rc) in river_cards.iter().enumerate() {
        let strength = [strengths0[ri].clone(), strengths1[ri].clone()];
        let mut order: [Vec<usize>; 2] = [(0..n[0]).collect(), (0..n[1]).collect()];
        let mut valid: [Vec<bool>; 2] = [vec![], vec![]];
        for p in 0..2 {
            order[p].sort_by_key(|&h| strength[p][h]);
            valid[p] = hands[p].iter().map(|&(a, b)| a != rc && b != rc).collect();
        }
        shows.push(Show { strength, order, valid });
    }
    let street = build(pot, stack, &sizes, extra_bet, oop_can_lead, &cfg);
    let mut street_store = Store::new(&street, n);
    let mut leaf_of = vec![usize::MAX; street.nodes.len()];
    let mut river_trees = vec![];
    let mut river_stores: Vec<Vec<Store>> = vec![];
    if !is_river {
        for (id, nd) in street.nodes.iter().enumerate() {
            if nd.kind == Kind::Close {
                leaf_of[id] = river_trees.len();
                let t = build(pot + 2.0 * nd.c[0], (stack - nd.c[0]).max(0.0), &river_sizes, 0.0, true, &river_cfg);
                river_stores.push((0..river_cards.len()).map(|_| Store::new(&t, n)).collect());
                river_trees.push(t);
            }
        }
    }
    // на тёрне у каждой пары рук 44 возможных карты ривера (52 − 4 борда − 4 закрытых)
    let river_weight = if is_river { 1.0 } else { 1.0 / 44.0 };
    let s = Ctx { hands, same, shows, stack, street, leaf_of, river_trees, river_weight };

    let ones0 = vec![1.0; n[0]];
    let ones1 = vec![1.0; n[1]];
    py.allow_threads(|| {
        for it in 1..=iterations {
            run(&s, is_river, &mut street_store, &mut river_stores, 0, &ones0, &ones1, it);
        }
    });

    // идём по сыгранным действиям к узлу героя
    let mut reach = [ones0, ones1];
    let mut node = 0usize;
    for (kind, amount) in path {
        let nd = s.street.nodes[node].clone();
        let idx = if kind == CHECK {
            0
        } else {
            let mut best = usize::MAX;
            let mut dist = f64::MAX;
            for (i, &(k, a)) in nd.acts.iter().enumerate() {
                if (k == BET || k == RAISE || k == ALLIN) && (a - amount).abs() < dist {
                    dist = (a - amount).abs();
                    best = i;
                }
            }
            if best == usize::MAX {
                return Err(pyo3::exceptions::PyValueError::new_err("в дереве нет такой ставки"));
            }
            best
        };
        let na = nd.acts.len();
        let strat = strategy(&street_store, node, n[nd.player], na, true);
        for h in 0..n[nd.player] {
            reach[nd.player][h] *= strat[h * na + idx];
        }
        node = nd.children[idx];
    }
    let nd = s.street.nodes[node].clone();
    if nd.kind != Kind::Decision || nd.player != hero_player {
        return Err(pyo3::exceptions::PyValueError::new_err("путь не приводит к ходу героя"));
    }
    let na = nd.acts.len();
    let strat = strategy(&street_store, node, n[hero_player], na, true);
    let opp = 1 - hero_player;
    let comp = s.compat(hero_player, &reach[opp]);
    let norm = if comp[hero_index] > 0.0 { comp[hero_index] } else { 1.0 };
    let mut actions = vec![];
    for (a, &ch) in nd.children.iter().enumerate() {
        // герой играет действие `a` со всеми руками — ценность считается для каждой руки отдельно
        let (r0, r1) = (reach[0].clone(), reach[1].clone());
        let (u0, u1) = run(&s, is_river, &mut street_store, &mut river_stores, ch, &r0, &r1, 0);
        let u = if hero_player == 0 { u0 } else { u1 };
        actions.push((nd.acts[a].0, nd.acts[a].1, strat[hero_index * na + a], u[hero_index] / norm));
    }
    // эквити героя против рук, с которыми оппонент дошёл до этой точки
    let (mut num, mut den) = (0.0, 0.0);
    for ri in 0..s.shows.len() {
        if !s.shows[ri].valid[hero_player][hero_index] {
            continue;
        }
        let q: Vec<f64> = reach[opp].iter().zip(&s.shows[ri].valid[opp]).map(|(r, v)| if *v { *r } else { 0.0 }).collect();
        let m = s.compat(hero_player, &q);
        let (win, lose) = s.win_lose(hero_player, &s.shows[ri], &q);
        let (w, l, c) = (win[hero_index], lose[hero_index], m[hero_index]);
        num += w + 0.5 * (c - w - l);
        den += c;
    }
    let equity = if den > 0.0 { num / den } else { 0.0 };
    let (ha, hb) = s.hands[hero_player][hero_index];
    let weights: Vec<f64> = s.hands[opp]
        .iter()
        .zip(&reach[opp])
        .map(|(&(a, b), r)| if a == ha || a == hb || b == ha || b == hb { 0.0 } else { *r })
        .collect();
    Ok((actions, equity, weights))
}

#[pymodule]
fn solver_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
