//! CFR+ солвер флопа, тёрна и ривера для игры один на один.
//!
//! Игрок 0 — без позиции (ходит первым), игрок 1 — в позиции. Силы рук на каждом возможном
//! продолжении борда считает Python (eval7) и передаёт сюда; здесь только деревья и CFR.
//! Улицы вложены рекурсивно: у каждого листа улицы свой набор деревьев следующей улицы,
//! по одному на каждую карту, которая может выйти.
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
    max_raises: u8, // сколько рейзов подряд разрешено на улице (0 — без рейзов)
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

fn facing(t: &mut Tree, player: usize, c: [f64; 2], raises: u8, stack: f64, cfg: &Cfg) -> usize {
    let id = push(t, Kind::Decision, player, c);
    let (me, opp) = (c[player], c[1 - player]);
    let mut acts = vec![(FOLD, me), (CALL, opp)];
    let f = push(t, Kind::Fold, player, c);
    let s = push(t, Kind::Close, 0, [opp, opp]);
    let mut children = vec![f, s];
    if raises < cfg.max_raises && opp < stack - 1e-9 {
        let mut to = (opp * cfg.raise_mult).min(stack);
        if to > 0.6 * stack {
            to = stack;
        }
        let mut nc = c;
        nc[player] = to;
        let ch = facing(t, 1 - player, nc, raises + 1, stack, cfg);
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
            let ch = facing(t, 1 - player, c, 0, stack, cfg);
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

/// Данные вскрытия для одного полного борда.
struct Show {
    strength: [Vec<u32>; 2],
    order: [Vec<usize>; 2], // индексы рук по возрастанию силы
}

/// Что происходит, когда торговля на улице закончена.
#[derive(Clone, Copy)]
enum Next {
    Show(usize), // вскрытие с данными shows[i]
    Deal(usize), // выходит следующая карта: deals[i]
}

/// Выход карты: какие карты перебираем, с каким весом и что дальше после каждой.
struct Deal {
    cards: Vec<u8>,
    weight: f64,
    next: Vec<Next>,
}

/// Настройки дерева улицы глубже корня.
struct StreetCfg {
    sizes: Vec<f64>,
    max_raises: u8,
}

/// Экземпляр улицы: дерево, стратегии и вложенные улицы для каждого листа и каждой карты.
struct Street {
    tree: Tree,
    store: Store,
    after: Next,
    leaf_of: Vec<usize>,    // узел -> номер листа (usize::MAX, если не лист)
    kids: Vec<Vec<Street>>, // [лист][карта]
}

/// Неизменяемая часть задачи.
struct Ctx {
    hands: [Vec<(u8, u8)>; 2],
    same: [Vec<Option<usize>>; 2], // индекс такой же комбинации в диапазоне оппонента
    shows: Vec<Show>,
    deals: Vec<Deal>,
    raise_mult: f64,
}

type Util = (Vec<f64>, Vec<f64>);

fn has(h: (u8, u8), card: u8) -> bool {
    h.0 == card || h.1 == card
}

impl Ctx {
    fn n(&self, p: usize) -> usize {
        self.hands[p].len()
    }

    #[allow(clippy::too_many_arguments)]
    fn make_street(&self, pot: f64, stack: f64, sizes: &[f64], extra: f64, lead: bool, max_raises: u8,
                   after: Next, deeper: &[StreetCfg]) -> Street {
        let n = [self.n(0), self.n(1)];
        let tree = build(pot, stack, sizes, extra, lead, &Cfg { raise_mult: self.raise_mult, max_raises });
        let store = Store::new(&tree, n);
        let mut leaf_of = vec![usize::MAX; tree.nodes.len()];
        let mut kids = vec![];
        if let Next::Deal(d) = after {
            for (id, nd) in tree.nodes.iter().enumerate() {
                if nd.kind == Kind::Close {
                    leaf_of[id] = kids.len();
                    let c = nd.c[0];
                    let row: Vec<Street> = self.deals[d]
                        .next
                        .iter()
                        .map(|nx| self.make_street(pot + 2.0 * c, (stack - c).max(0.0), &deeper[0].sizes, 0.0, true,
                                                   deeper[0].max_raises, *nx, &deeper[1..]))
                        .collect();
                    kids.push(row);
                }
            }
        }
        Street { tree, store, after, leaf_of, kids }
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

    fn showdown(&self, id: usize, pot: f64, c: f64, r0: &[f64], r1: &[f64]) -> Util {
        let sh = &self.shows[id];
        let mut out: [Vec<f64>; 2] = [vec![], vec![]];
        for me in 0..2 {
            let opp_reach = if me == 0 { r1 } else { r0 };
            let comp = self.compat(me, opp_reach);
            let (win, lose) = self.win_lose(me, sh, opp_reach);
            out[me] = (0..self.n(me))
                .map(|h| (pot + c) * win[h] - c * lose[h] + pot / 2.0 * (comp[h] - win[h] - lose[h]))
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

    /// Лист улицы, после которого выходит карта: все карты считаются параллельно.
    fn chance(&self, kids: &mut [Street], d: usize, c: f64, r0: &[f64], r1: &[f64], iter: usize) -> Util {
        let deal = &self.deals[d];
        let parts: Vec<Util> = kids
            .par_iter_mut()
            .enumerate()
            .map(|(i, kid)| {
                let card = deal.cards[i];
                let q0: Vec<f64> = r0.iter().zip(&self.hands[0]).map(|(r, h)| if has(*h, card) { 0.0 } else { *r }).collect();
                let q1: Vec<f64> = r1.iter().zip(&self.hands[1]).map(|(r, h)| if has(*h, card) { 0.0 } else { *r }).collect();
                let (mut a, mut b) = self.cfr(kid, 0, &q0, &q1, iter);
                // полезность следующей улицы считается от её стартового банка; переводим к началу этой
                let m0 = self.compat(0, &q1);
                let m1 = self.compat(1, &q0);
                for h in 0..a.len() {
                    a[h] = if has(self.hands[0][h], card) { 0.0 } else { (a[h] - c * m0[h]) * deal.weight };
                }
                for h in 0..b.len() {
                    b[h] = if has(self.hands[1][h], card) { 0.0 } else { (b[h] - c * m1[h]) * deal.weight };
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
    fn cfr(&self, st: &mut Street, node: usize, r0: &[f64], r1: &[f64], iter: usize) -> Util {
        let (kind, player, c, n_acts, children, pot) = {
            let nd = &st.tree.nodes[node];
            (nd.kind, nd.player, nd.c, nd.acts.len(), nd.children.clone(), st.tree.pot)
        };
        match kind {
            Kind::Fold => return self.fold(player, pot, c, r0, r1),
            Kind::Close => {
                return match st.after {
                    Next::Show(id) => self.showdown(id, pot, c[0], r0, r1),
                    Next::Deal(d) => {
                        let leaf = st.leaf_of[node];
                        self.chance(&mut st.kids[leaf], d, c[0], r0, r1, iter)
                    }
                };
            }
            Kind::Decision => {}
        }
        let n_me = self.n(player);
        let strat = strategy(&st.store, node, n_me, n_acts, iter == 0);
        let mut utils: Vec<Vec<f64>> = Vec::with_capacity(n_acts);
        let mut other = vec![0.0; self.n(1 - player)];
        let reach_me = if player == 0 { r0 } else { r1 };
        for (a, &ch) in children.iter().enumerate() {
            let q: Vec<f64> = (0..n_me).map(|h| reach_me[h] * strat[h * n_acts + a]).collect();
            let (u0, u1) = if player == 0 { self.cfr(st, ch, &q, r1, iter) } else { self.cfr(st, ch, r0, &q, iter) };
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
            let w = iter as f64;
            for h in 0..n_me {
                for a in 0..n_acts {
                    let i = h * n_acts + a;
                    st.store.regret[node][i] = (st.store.regret[node][i] + utils[a][h] - mine[h]).max(0.0);
                    st.store.ssum[node][i] += w * reach_me[h] * strat[i];
                }
            }
        }
        if player == 0 { (mine, other) } else { (other, mine) }
    }

    /// (сумма выигрышей, сумма весов) руки героя против достижимостей оппонента по всем продолжениям борда.
    fn equity(&self, next: Next, me: usize, hand: usize, opp_reach: &[f64]) -> (f64, f64) {
        match next {
            Next::Show(id) => {
                let comp = self.compat(me, opp_reach);
                let (win, lose) = self.win_lose(me, &self.shows[id], opp_reach);
                let (w, l, c) = (win[hand], lose[hand], comp[hand]);
                (w + 0.5 * (c - w - l), c)
            }
            Next::Deal(d) => {
                let deal = &self.deals[d];
                let (mut num, mut den) = (0.0, 0.0);
                for (i, &card) in deal.cards.iter().enumerate() {
                    if has(self.hands[me][hand], card) {
                        continue;
                    }
                    let q: Vec<f64> = opp_reach.iter().zip(&self.hands[1 - me]).map(|(r, h)| if has(*h, card) { 0.0 } else { *r }).collect();
                    let (a, b) = self.equity(deal.next[i], me, hand, &q);
                    num += a;
                    den += b;
                }
                (num, den)
            }
        }
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

fn to_next(x: i64) -> Next {
    if x >= 0 { Next::Show(x as usize) } else { Next::Deal((-x - 1) as usize) }
}

type Answer = (Vec<(u8, f64, f64, f64)>, f64, Vec<f64>, Vec<f64>, Vec<f64>);

/// Решить спот.
///
/// - hands*, w*: руки (номера карт 0..51) и их начальные веса;
/// - strengths*[вскрытие][рука]: силы рук на каждом полном борде;
/// - deal_*: описание выходов карт; next >= 0 — номер вскрытия, next < 0 — выход карты с номером -next-1;
/// - root_next: что происходит после торговли на текущей улице (в том же кодировании);
/// - root_raises, deeper_sizes / deeper_raises: число рейзов подряд и размеры ставок на текущей и следующих улицах;
/// - path: действия на текущей улице до нужного узла: (тип, сумма), типы 0 фолд, 1 чек, 2 колл, 3 ставка или рейз;
/// - want_reach: вернуть только веса рук обоих игроков в узле после path (hero_* игнорируются).
///
/// Возвращает (действия героя: (тип, сумма, частота, EV), эквити героя, веса рук оппонента,
/// веса рук игрока 0 в узле, веса рук игрока 1 в узле).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve(
    py: Python<'_>,
    hands0: Vec<(u8, u8)>,
    hands1: Vec<(u8, u8)>,
    w0: Vec<f64>,
    w1: Vec<f64>,
    strengths0: Vec<Vec<u32>>,
    strengths1: Vec<Vec<u32>>,
    deal_cards: Vec<Vec<u8>>,
    deal_next: Vec<Vec<i64>>,
    deal_weight: Vec<f64>,
    root_next: i64,
    pot: f64,
    stack: f64,
    sizes: Vec<f64>,
    root_raises: u8,
    deeper_sizes: Vec<Vec<f64>>,
    deeper_raises: Vec<u8>,
    raise_mult: f64,
    extra_bet: f64,
    oop_can_lead: bool,
    iterations: usize,
    hero_player: usize,
    hero_index: usize,
    path: Vec<(u8, f64)>,
    want_reach: bool,
) -> PyResult<Answer> {
    let n = [hands0.len(), hands1.len()];
    let hands = [hands0, hands1];
    let mut same: [Vec<Option<usize>>; 2] = [vec![None; n[0]], vec![None; n[1]]];
    for (i, h) in hands[0].iter().enumerate() {
        if let Some(j) = hands[1].iter().position(|g| (g.0 == h.0 && g.1 == h.1) || (g.0 == h.1 && g.1 == h.0)) {
            same[0][i] = Some(j);
            same[1][j] = Some(i);
        }
    }
    let mut shows = Vec::with_capacity(strengths0.len());
    for (s0, s1) in strengths0.into_iter().zip(strengths1) {
        let strength = [s0, s1];
        let mut order: [Vec<usize>; 2] = [(0..n[0]).collect(), (0..n[1]).collect()];
        for p in 0..2 {
            order[p].sort_by_key(|&h| strength[p][h]);
        }
        shows.push(Show { strength, order });
    }
    let deals: Vec<Deal> = deal_cards
        .into_iter()
        .zip(deal_next)
        .zip(deal_weight)
        .map(|((cards, next), weight)| Deal { cards, weight, next: next.into_iter().map(to_next).collect() })
        .collect();
    let deeper: Vec<StreetCfg> = deeper_sizes.into_iter().zip(deeper_raises).map(|(sizes, max_raises)| StreetCfg { sizes, max_raises }).collect();
    let ctx = Ctx { hands, same, shows, deals, raise_mult };
    let mut root = ctx.make_street(pot, stack, &sizes, extra_bet, oop_can_lead, root_raises, to_next(root_next), &deeper);

    py.allow_threads(|| {
        for it in 1..=iterations {
            ctx.cfr(&mut root, 0, &w0, &w1, it);
        }
    });

    // идём по сыгранным действиям к нужному узлу
    let mut reach = [w0, w1];
    let mut node = 0usize;
    for (kind, amount) in path {
        let nd = root.tree.nodes[node].clone();
        if nd.kind != Kind::Decision {
            return Err(pyo3::exceptions::PyValueError::new_err("путь длиннее дерева"));
        }
        let want = match kind {
            0 => FOLD,
            1 => CHECK,
            2 => CALL,
            _ => BET,
        };
        let mut best = usize::MAX;
        let mut dist = f64::MAX;
        for (i, &(k, a)) in nd.acts.iter().enumerate() {
            let d = if want == BET {
                if k == BET || k == RAISE || k == ALLIN { (a - amount).abs() } else { f64::MAX }
            } else if k == want {
                0.0
            } else {
                f64::MAX
            };
            if d < dist {
                dist = d;
                best = i;
            }
        }
        if best == usize::MAX {
            return Err(pyo3::exceptions::PyValueError::new_err("в дереве нет такого действия"));
        }
        let na = nd.acts.len();
        let strat = strategy(&root.store, node, n[nd.player], na, true);
        for h in 0..n[nd.player] {
            reach[nd.player][h] *= strat[h * na + best];
        }
        node = nd.children[best];
    }
    if want_reach {
        let [a, b] = reach;
        return Ok((vec![], 0.0, vec![], a, b));
    }
    let nd = root.tree.nodes[node].clone();
    if nd.kind != Kind::Decision || nd.player != hero_player {
        return Err(pyo3::exceptions::PyValueError::new_err("путь не приводит к ходу героя"));
    }
    let na = nd.acts.len();
    let strat = strategy(&root.store, node, n[hero_player], na, true);
    let opp = 1 - hero_player;
    let comp = ctx.compat(hero_player, &reach[opp]);
    let norm = if comp[hero_index] > 0.0 { comp[hero_index] } else { 1.0 };
    let mut actions = vec![];
    for (a, &ch) in nd.children.iter().enumerate() {
        let (r0, r1) = (reach[0].clone(), reach[1].clone());
        let (u0, u1) = ctx.cfr(&mut root, ch, &r0, &r1, 0);
        let u = if hero_player == 0 { u0 } else { u1 };
        actions.push((nd.acts[a].0, nd.acts[a].1, strat[hero_index * na + a], u[hero_index] / norm));
    }
    let (num, den) = ctx.equity(root.after, hero_player, hero_index, &reach[opp]);
    let equity = if den > 0.0 { num / den } else { 0.0 };
    let hero_hand = ctx.hands[hero_player][hero_index];
    let weights: Vec<f64> = ctx.hands[opp]
        .iter()
        .zip(&reach[opp])
        .map(|(&h, r)| if has(h, hero_hand.0) || has(h, hero_hand.1) { 0.0 } else { *r })
        .collect();
    let [a, b] = reach;
    Ok((actions, equity, weights, a, b))
}

#[pymodule]
fn solver_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
