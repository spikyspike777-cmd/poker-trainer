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
    stack: f64,
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
    let mut t = Tree { nodes: vec![], pot, stack };
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
    Net(usize),  // остаток раздачи оценивает нейросеть; deals[i] нужен только для листьев-олл-инов
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
    kids: Vec<Vec<Street>>, // [лист][карта]; пусто, если лист оценивает нейросеть
    board: Vec<u8>,         // карты борда на этой улице
}

const F_HAND: usize = 7; // признаков на руку
const C_CTX: usize = 41; // признаков общего контекста

/// Сеть одного игрока: оценивает каждую руку по её признакам и общему контексту.
struct Head {
    wf: Vec<f32>, // [h][F_HAND]
    wc: Vec<f32>, // [h][hc]
    b: Vec<f32>,
    blocks: Vec<(Vec<f32>, Vec<f32>)>, // остаточные блоки: (вес [h][h], сдвиг)
    wo: Vec<f32>,
    bo: f32,
}

/// Веса сети-оценщика: контекст -> два слоя; затем для каждой руки своя оценка общей сетью игрока.
struct Net {
    hc: usize,
    h: usize,
    wc1: Vec<f32>, // [hc][C_CTX]
    bc1: Vec<f32>,
    wc2: Vec<f32>, // [hc][hc]
    bc2: Vec<f32>,
    heads: [Head; 2],
}

/// Обученная сеть-оценщик. Создаётся из Python: все веса одним блоком байтов float32.
#[pyclass]
struct ValueNet {
    net: std::sync::Arc<Net>,
}

#[pymethods]
impl ValueNet {
    #[new]
    fn new(hc: usize, h: usize, n_blocks: usize, blob: &[u8]) -> PyResult<Self> {
        let all: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let head_len = h * F_HAND + h * hc + h + n_blocks * (h * h + h) + h + 1;
        if all.len() != hc * C_CTX + hc + hc * hc + hc + 2 * head_len {
            return Err(pyo3::exceptions::PyValueError::new_err("размер весов не соответствует архитектуре сети"));
        }
        let mut at = 0;
        let mut take = |n: usize| {
            let v = all[at..at + n].to_vec();
            at += n;
            v
        };
        let (wc1, bc1, wc2, bc2) = (take(hc * C_CTX), take(hc), take(hc * hc), take(hc));
        let mut head = || {
            let (wf, wc, b) = (take(h * F_HAND), take(h * hc), take(h));
            let blocks = (0..n_blocks).map(|_| (take(h * h), take(h))).collect();
            let (wo, bo) = (take(h), take(1)[0]);
            Head { wf, wc, b, blocks, wo, bo }
        };
        let heads = [head(), head()];
        Ok(ValueNet { net: std::sync::Arc::new(Net { hc, h, wc1, bc1, wc2, bc2, heads }) })
    }
}

/// Всё, что нужно сети в листе: контекст, признаки рук и точная «ценность без ставок».
struct LeafFeat {
    ctx: Vec<f32>,
    f: [Vec<f32>; 2], // [рука][F_HAND]
    e: [Vec<f64>; 2], // ценность без ставок: в долях банка, без половины банка, с весами оппонента как есть
    m: [Vec<f64>; 2], // суммарный вес рук оппонента, совместимых с рукой
}

/// Скалярное произведение с восемью независимыми сумматорами: так компилятор превращает цикл
/// в векторные инструкции (обычная сумма чисел с плавающей точкой не векторизуется).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    acc.iter().sum::<f32>() + tail
}

/// Неизменяемая часть задачи.
struct Ctx {
    hands: [Vec<(u8, u8)>; 2],
    same: [Vec<Option<usize>>; 2], // индекс такой же комбинации в диапазоне оппонента
    shows: Vec<Show>,
    deals: Vec<Deal>,
    raise_mult: f64,
    quadratic: bool,
    net: Option<std::sync::Arc<Net>>,
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
    fn make_street(&self, board: &[u8], pot: f64, stack: f64, sizes: &[f64], extra: f64, lead: bool, max_raises: u8,
                   after: Next, deeper: &[StreetCfg]) -> Street {
        let n = [self.n(0), self.n(1)];
        let tree = build(pot, stack, sizes, extra, lead, &Cfg { raise_mult: self.raise_mult, max_raises });
        let store = Store::new(&tree, n);
        let mut leaf_of = vec![usize::MAX; tree.nodes.len()];
        let mut kids = vec![];
        let deal = match after {
            Next::Deal(d) => Some((d, false)),
            Next::Net(d) => Some((d, true)),
            Next::Show(_) => None,
        };
        if let Some((d, by_net)) = deal {
            for (id, nd) in tree.nodes.iter().enumerate() {
                if nd.kind == Kind::Close {
                    leaf_of[id] = kids.len();
                    let c = nd.c[0];
                    let rest = (stack - c).max(0.0);
                    if by_net && rest > 1e-9 {
                        kids.push(vec![]); // этот лист оценит сеть; олл-ины считаются точно, перебором карт
                        continue;
                    }
                    let row: Vec<Street> = self.deals[d]
                        .next
                        .iter()
                        .zip(&self.deals[d].cards)
                        .map(|(nx, card)| {
                            let mut b = board.to_vec();
                            b.push(*card);
                            // после олл-ина торговли нет: дерево из одних чеков, дальше — только выход карт и вскрытие
                            let nx = if by_net { Self::exact(*nx) } else { *nx };
                            self.make_street(&b, pot + 2.0 * c, rest, &deeper[0].sizes, 0.0, true, deeper[0].max_raises, nx, &deeper[1..])
                        })
                        .collect();
                    kids.push(row);
                }
            }
        }
        Street { tree, store, after, leaf_of, kids, board: board.to_vec() }
    }

    fn exact(nx: Next) -> Next {
        match nx {
            Next::Net(d) => Next::Deal(d),
            other => other,
        }
    }

    /// Признаки листа «улица закончена, следующая карта не вышла». Перебирает все карты выхода `d`
    /// и на каждой считает вскрытие без ставок. Используется и при обучении, и при игре.
    fn leaf_features(&self, d: usize, board: &[u8], pot: f64, stack: f64, r0: &[f64], r1: &[f64]) -> LeafFeat {
        let deal = &self.deals[d];
        let n = [self.n(0), self.n(1)];
        // по каждой карте: (ценность без ставок, эквити руки) для обоих игроков
        let per_card: Vec<[(Vec<f64>, Vec<f64>); 2]> = (0..deal.cards.len())
            .into_par_iter()
            .map(|i| {
                let card = deal.cards[i];
                let id = match deal.next[i] {
                    Next::Show(id) => id,
                    _ => panic!("сеть-оценщик поддерживает только точку перед последней картой"),
                };
                let q0: Vec<f64> = r0.iter().zip(&self.hands[0]).map(|(r, h)| if has(*h, card) { 0.0 } else { *r }).collect();
                let q1: Vec<f64> = r1.iter().zip(&self.hands[1]).map(|(r, h)| if has(*h, card) { 0.0 } else { *r }).collect();
                let (u0, u1) = self.showdown(id, 1.0, 0.0, &q0, &q1);
                let mm = [self.compat(0, &q1), self.compat(1, &q0)];
                let us = [u0, u1];
                let mut out: [(Vec<f64>, Vec<f64>); 2] = [(vec![], vec![]), (vec![], vec![])];
                for p in 0..2 {
                    let mut e = vec![f64::NAN; n[p]]; // NAN — рука содержит эту карту
                    let mut q = vec![f64::NAN; n[p]];
                    for h in 0..n[p] {
                        if !has(self.hands[p][h], card) {
                            e[h] = us[p][h] - 0.5 * mm[p][h];
                            q[h] = if mm[p][h] > 1e-12 { e[h] / mm[p][h] } else { 0.0 };
                        }
                    }
                    out[p] = (e, q);
                }
                out
            })
            .collect();
        let reach = [r0, r1];
        let m = [self.compat(0, r1), self.compat(1, r0)];
        let sums = [r0.iter().sum::<f64>().max(1e-300), r1.iter().sum::<f64>().max(1e-300)];
        let mut e = [vec![0.0; n[0]], vec![0.0; n[1]]];
        let mut f = [vec![0.0f32; n[0] * F_HAND], vec![0.0f32; n[1] * F_HAND]];
        let mut ctx = vec![0.0f32; C_CTX];
        ctx[0] = (pot / (pot + 2.0 * stack)) as f32;
        ctx[1] = ((1.0 + (stack / pot).min(200.0)).ln() / 4.0) as f32;
        let mut suits = [0.0f32; 4];
        for &c in board {
            ctx[2 + (c / 4) as usize] += 0.5; // сколько карт каждого ранга
            suits[(c % 4) as usize] += 0.25;
        }
        suits.sort_by(|a, b| b.partial_cmp(a).unwrap()); // масти по убыванию числа карт: неважно, как они названы
        ctx[15..19].copy_from_slice(&suits);
        for p in 0..2 {
            let opp_sum = sums[1 - p];
            let mut q = vec![0.0; n[p]];
            for h in 0..n[p] {
                let (mut s1, mut s2, mut cnt, mut hi, mut lo) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for card in &per_card {
                    let (ev, qv) = (&card[p].0[h], &card[p].1[h]);
                    if ev.is_nan() {
                        continue;
                    }
                    e[p][h] += ev * deal.weight;
                    s1 += qv;
                    s2 += qv * qv;
                    cnt += 1.0;
                    if *qv > 0.35 {
                        hi += 1.0;
                    }
                    if *qv < -0.35 {
                        lo += 1.0;
                    }
                }
                q[h] = if m[p][h] > 1e-12 { e[p][h] / m[p][h] } else { 0.0 };
                let cnt = f64::max(cnt, 1.0);
                let sd = (s2 / cnt - (s1 / cnt) * (s1 / cnt)).max(0.0).sqrt();
                let row = &mut f[p][h * F_HAND..(h + 1) * F_HAND];
                row[0] = (2.0 * q[h]) as f32; // эквити против диапазона оппонента, от −1 до 1
                row[1] = (2.0 * sd) as f32; // насколько эквити зависит от карты
                row[2] = (hi / cnt) as f32; // доля карт, на которых рука почти всегда выигрывает
                row[3] = (lo / cnt) as f32; // доля карт, на которых почти всегда проигрывает
                row[6] = ((m[p][h] / opp_sum - 0.85) * 5.0) as f32; // сколько рук оппонента рука блокирует
            }
            // место руки в своём диапазоне по силе и её относительный вес
            let mut order: Vec<usize> = (0..n[p]).collect();
            order.sort_by(|&a, &b| q[a].partial_cmp(&q[b]).unwrap());
            let positive = reach[p].iter().filter(|r| **r > 0.0).count().max(1) as f64;
            let mut below = 0.0;
            let mut mean_q = 0.0;
            for &h in &order {
                let w = reach[p][h] / sums[p];
                f[p][h * F_HAND + 4] = (below + 0.5 * w) as f32;
                below += w;
                f[p][h * F_HAND + 5] = if w > 0.0 { ((w * positive).ln().clamp(-4.0, 4.0) / 4.0) as f32 } else { -1.0 };
                let bin = (((q[h] + 0.5) * 10.0) as usize).min(9);
                ctx[19 + p * 10 + bin] += w as f32; // распределение диапазона по силе
                mean_q += w * q[h];
            }
            ctx[39 + p] = (2.0 * mean_q) as f32;
        }
        LeafFeat { ctx, f, e, m }
    }

    /// Прямой проход сети: поправка за торговлю для каждой руки, в долях банка (диапазон оппонента с суммой 1).
    fn net_forward(&self, net: &Net, lf: &LeafFeat) -> [Vec<f64>; 2] {
        let (hc, h) = (net.hc, net.h);
        let c1: Vec<f32> = (0..hc).map(|i| (net.bc1[i] + dot(&net.wc1[i * C_CTX..(i + 1) * C_CTX], &lf.ctx)).max(0.0)).collect();
        let c2: Vec<f32> = (0..hc).map(|i| (net.bc2[i] + dot(&net.wc2[i * hc..(i + 1) * hc], &c1)).max(0.0)).collect();
        let mut out: [Vec<f64>; 2] = [vec![], vec![]];
        for p in 0..2 {
            let head = &net.heads[p];
            let base: Vec<f32> = (0..h).map(|i| head.b[i] + dot(&head.wc[i * hc..(i + 1) * hc], &c2)).collect();
            out[p] = lf.f[p]
                .par_chunks(F_HAND)
                .map(|row| {
                    let mut z: Vec<f32> = (0..h).map(|i| (base[i] + dot(&head.wf[i * F_HAND..(i + 1) * F_HAND], row)).max(0.0)).collect();
                    for (w, b) in &head.blocks {
                        let y: Vec<f32> = (0..h).map(|i| (b[i] + dot(&w[i * h..(i + 1) * h], &z)).max(0.0)).collect();
                        for i in 0..h {
                            z[i] += y[i];
                        }
                    }
                    (head.bo + dot(&head.wo, &z)) as f64
                })
                .collect();
        }
        out
    }

    /// Оценка остатка раздачи: точная «ценность без ставок» плюс поправка за торговлю от сети.
    /// Возвращает полезности в тех же единицах, что и решение следующей улицы с банком `pot`.
    fn net_leaf(&self, net: &Net, d: usize, board: &[u8], pot: f64, stack: f64, r0: &[f64], r1: &[f64]) -> Util {
        let s0: f64 = r0.iter().sum();
        let s1: f64 = r1.iter().sum();
        if s0 <= 1e-12 || s1 <= 1e-12 {
            // у одного из игроков пустой диапазон: делим банк поровну с теми руками, что остались
            let m0 = self.compat(0, r1);
            let m1 = self.compat(1, r0);
            return (m0.iter().map(|m| pot / 2.0 * m).collect(), m1.iter().map(|m| pot / 2.0 * m).collect());
        }
        let lf = self.leaf_features(d, board, pot, stack, r0, r1);
        let [mut t0, mut t1] = self.net_forward(net, &lf);
        // игра с нулевой суммой: взвешенные по диапазонам поправки двух игроков должны в сумме давать ноль
        let err: f64 = t0.iter().zip(r0).map(|(t, r)| t * r / s0).sum::<f64>() + t1.iter().zip(r1).map(|(t, r)| t * r / s1).sum::<f64>();
        for t in t0.iter_mut() {
            *t -= err / 2.0;
        }
        for t in t1.iter_mut() {
            *t -= err / 2.0;
        }
        let u0 = (0..t0.len()).map(|i| (t0[i] * s1 + lf.e[0][i]) * pot + pot / 2.0 * lf.m[0][i]).collect();
        let u1 = (0..t1.len()).map(|i| (t1[i] * s0 + lf.e[1][i]) * pot + pot / 2.0 * lf.m[1][i]).collect();
        (u0, u1)
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
    #[allow(clippy::too_many_arguments)]
    fn chance(&self, kids: &mut [Street], d: usize, c: f64, r0: &[f64], r1: &[f64], iter: usize, upd: usize) -> Util {
        let deal = &self.deals[d];
        let parts: Vec<Util> = kids
            .par_iter_mut()
            .enumerate()
            .map(|(i, kid)| {
                let card = deal.cards[i];
                let q0: Vec<f64> = r0.iter().zip(&self.hands[0]).map(|(r, h)| if has(*h, card) { 0.0 } else { *r }).collect();
                let q1: Vec<f64> = r1.iter().zip(&self.hands[1]).map(|(r, h)| if has(*h, card) { 0.0 } else { *r }).collect();
                let (mut a, mut b) = self.cfr(kid, 0, &q0, &q1, iter, upd);
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
    /// upd — чьи сожаления обновляем на этом проходе (0 или 1; 2 — обоих).
    fn cfr(&self, st: &mut Street, node: usize, r0: &[f64], r1: &[f64], iter: usize, upd: usize) -> Util {
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
                        self.chance(&mut st.kids[leaf], d, c[0], r0, r1, iter, upd)
                    }
                    Next::Net(d) => {
                        let leaf = st.leaf_of[node];
                        if !st.kids[leaf].is_empty() {
                            return self.chance(&mut st.kids[leaf], d, c[0], r0, r1, iter, upd);
                        }
                        let net = self.net.as_ref().expect("для листа с сетью не передана сеть");
                        let (mut u0, mut u1) = self.net_leaf(net, d, &st.board, pot + 2.0 * c[0], st.tree.stack - c[0], r0, r1);
                        // оценка считается от банка следующей улицы; переводим к началу этой
                        let m0 = self.compat(0, r1);
                        let m1 = self.compat(1, r0);
                        for h in 0..u0.len() {
                            u0[h] -= c[0] * m0[h];
                        }
                        for h in 0..u1.len() {
                            u1[h] -= c[0] * m1[h];
                        }
                        (u0, u1)
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
            let (u0, u1) = if player == 0 { self.cfr(st, ch, &q, r1, iter, upd) } else { self.cfr(st, ch, r0, &q, iter, upd) };
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
        if iter > 0 && (upd == 2 || upd == player) {
            let w = if self.quadratic { (iter * iter) as f64 } else { iter as f64 }; // вес итерации в средней стратегии
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

    /// Лучший ответ игрока `br` на среднюю стратегию оппонента: ценность каждой его руки
    /// (взвешена достижимостями оппонента, как и в cfr).
    fn best_response(&self, st: &Street, node: usize, br: usize, r_opp: &[f64]) -> Vec<f64> {
        let nd = &st.tree.nodes[node];
        let pot = st.tree.pot;
        let ones = vec![1.0; self.n(br)];
        let pick = |u: Util| if br == 0 { u.0 } else { u.1 };
        let pair = |q: &[f64]| if br == 0 { (ones.clone(), q.to_vec()) } else { (q.to_vec(), ones.clone()) };
        match nd.kind {
            Kind::Fold => {
                let (r0, r1) = pair(r_opp);
                pick(self.fold(nd.player, pot, nd.c, &r0, &r1))
            }
            Kind::Close => match st.after {
                Next::Show(id) => {
                    let (r0, r1) = pair(r_opp);
                    pick(self.showdown(id, pot, nd.c[0], &r0, &r1))
                }
                Next::Deal(d) | Next::Net(d) => {
                    let deal = &self.deals[d];
                    let c = nd.c[0];
                    let kids = &st.kids[st.leaf_of[node]];
                    let parts: Vec<Vec<f64>> = kids
                        .par_iter()
                        .enumerate()
                        .map(|(i, kid)| {
                            let card = deal.cards[i];
                            let q: Vec<f64> = r_opp.iter().zip(&self.hands[1 - br]).map(|(r, h)| if has(*h, card) { 0.0 } else { *r }).collect();
                            let mut v = self.best_response(kid, 0, br, &q);
                            let m = self.compat(br, &q);
                            for h in 0..v.len() {
                                v[h] = if has(self.hands[br][h], card) { 0.0 } else { (v[h] - c * m[h]) * deal.weight };
                            }
                            v
                        })
                        .collect();
                    let mut out = vec![0.0; self.n(br)];
                    for v in parts {
                        for (x, y) in out.iter_mut().zip(v) {
                            *x += y;
                        }
                    }
                    out
                }
            },
            Kind::Decision => {
                let na = nd.acts.len();
                if nd.player == br {
                    let mut best = vec![f64::MIN; self.n(br)];
                    for &ch in &nd.children {
                        let v = self.best_response(st, ch, br, r_opp);
                        for (b, x) in best.iter_mut().zip(v) {
                            if x > *b {
                                *b = x;
                            }
                        }
                    }
                    best
                } else {
                    let n_opp = self.n(1 - br);
                    let strat = strategy(&st.store, node, n_opp, na, true);
                    let mut out = vec![0.0; self.n(br)];
                    for (a, &ch) in nd.children.iter().enumerate() {
                        let q: Vec<f64> = (0..n_opp).map(|h| r_opp[h] * strat[h * na + a]).collect();
                        let v = self.best_response(st, ch, br, &q);
                        for (x, y) in out.iter_mut().zip(v) {
                            *x += y;
                        }
                    }
                    out
                }
            }
        }
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
            Next::Deal(d) | Next::Net(d) => {
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
    if x >= 0 {
        Next::Show(x as usize)
    } else if x > -1000 {
        Next::Deal((-x - 1) as usize)
    } else {
        Next::Net((-x - 1000) as usize)
    }
}

type Answer = (Vec<(u8, f64, f64, f64)>, f64, Vec<f64>, Vec<f64>, Vec<f64>, f64, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

/// Решить спот.
///
/// - hands*, w*: руки (номера карт 0..51) и их начальные веса;
/// - strengths*[вскрытие][рука]: силы рук на каждом полном борде;
/// - deal_*: описание выходов карт; next >= 0 — номер вскрытия, next < 0 — выход карты с номером -next-1;
/// - root_next: что происходит после торговли на текущей улице (в том же кодировании);
/// - root_raises, deeper_sizes / deeper_raises: число рейзов подряд и размеры ставок на текущей и следующих улицах;
/// - path: действия на текущей улице до нужного узла: (тип, сумма), типы 0 фолд, 1 чек, 2 колл, 3 ставка или рейз;
/// - want_reach: вернуть только веса рук обоих игроков в узле после path (hero_* игнорируются);
/// - deal_next и root_next <= -1000: остаток раздачи оценивает сеть `net` (выход карты -x-1000 — для олл-инов);
/// - board: карты борда текущей улицы (нужны сети); want_values: вернуть ценности всех рук в корне.
///
/// Возвращает (действия героя: (тип, сумма, частота, EV), эквити героя, веса рук оппонента,
/// веса рук игрока 0 в узле, веса рук игрока 1 в узле).
#[pyfunction]
#[pyo3(signature = (hands0, hands1, w0, w1, strengths0, strengths1, deal_cards, deal_next, deal_weight, root_next,
                    pot, stack, sizes, root_raises, deeper_sizes, deeper_raises, raise_mult, extra_bet, oop_can_lead,
                    iterations, hero_player, hero_index, path, want_reach, want_expl, want_all, alternate, board,
                    want_values, net=None))]
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
    want_expl: bool,
    want_all: bool,
    alternate: bool,
    board: Vec<u8>,
    want_values: bool,
    net: Option<PyRef<ValueNet>>,
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
    let uses_net = root_next <= -1000 || deals.iter().any(|d| d.next.iter().any(|n| matches!(n, Next::Net(_))));
    let net = net.map(|n| n.net.clone());
    if uses_net && net.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err("в раскладке есть листья с сетью, но сеть не передана"));
    }
    let ctx = Ctx { hands, same, shows, deals, raise_mult, quadratic: alternate, net };
    let mut root = ctx.make_street(&board, pot, stack, &sizes, extra_bet, oop_can_lead, root_raises, to_next(root_next), &deeper);

    py.allow_threads(|| {
        for it in 1..=iterations {
            if alternate {
                ctx.cfr(&mut root, 0, &w0, &w1, it, 0); // поочерёдные обновления: сначала один игрок,
                ctx.cfr(&mut root, 0, &w0, &w1, it, 1); // затем другой — уже против его новой стратегии
            } else {
                ctx.cfr(&mut root, 0, &w0, &w1, it, 2); // оба игрока за один проход
            }
        }
    });

    // эксплуатируемость: сколько в среднем теряет наша стратегия против лучшего ответа (в bb)
    let mut expl = -1.0;
    if want_expl && !uses_net {
        let br0 = ctx.best_response(&root, 0, 0, &w1);
        let br1 = ctx.best_response(&root, 0, 1, &w0);
        let pairs: f64 = ctx.compat(0, &w1).iter().zip(&w0).map(|(m, w)| m * w).sum();
        if pairs > 0.0 {
            let v0: f64 = br0.iter().zip(&w0).map(|(v, w)| v * w).sum::<f64>() / pairs;
            let v1: f64 = br1.iter().zip(&w1).map(|(v, w)| v * w).sum::<f64>() / pairs;
            expl = (v0 + v1 - pot) / 2.0;
        }
    }

    // ценности всех рук обоих игроков в корне при игре по средней стратегии
    let (mut val0, mut val1) = (vec![], vec![]);
    if want_values {
        let (a, b) = ctx.cfr(&mut root, 0, &w0, &w1, 0, 2);
        val0 = a;
        val1 = b;
    }

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
        return Ok((vec![], 0.0, vec![], a, b, expl, vec![], vec![], val0, val1));
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
    let (mut all_freq, mut all_ev) = (vec![], vec![]);
    for (a, &ch) in nd.children.iter().enumerate() {
        let (r0, r1) = (reach[0].clone(), reach[1].clone());
        let (u0, u1) = ctx.cfr(&mut root, ch, &r0, &r1, 0, 2);
        let u = if hero_player == 0 { u0 } else { u1 };
        actions.push((nd.acts[a].0, nd.acts[a].1, strat[hero_index * na + a], u[hero_index] / norm));
        if want_all {
            all_freq.push((0..n[hero_player]).map(|h| strat[h * na + a]).collect());
            all_ev.push((0..n[hero_player]).map(|h| if comp[h] > 0.0 { u[h] / comp[h] } else { 0.0 }).collect());
        }
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
    Ok((actions, equity, weights, a, b, expl, all_freq, all_ev, val0, val1))
}

/// Признаки листа для генератора обучающих данных — тот же расчёт, что внутри солвера.
/// Возвращает (контекст, признаки рук игрока 0, признаки рук игрока 1, ценность без ставок для игроков 0 и 1).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn leaf_features(
    hands0: Vec<(u8, u8)>,
    hands1: Vec<(u8, u8)>,
    w0: Vec<f64>,
    w1: Vec<f64>,
    strengths0: Vec<Vec<u32>>,
    strengths1: Vec<Vec<u32>>,
    cards: Vec<u8>,
    board: Vec<u8>,
    pot: f64,
    stack: f64,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>, Vec<f64>)> {
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
    let next = (0..cards.len()).map(Next::Show).collect();
    let weight = 1.0 / (52 - board.len() - 4) as f64;
    let ctx = Ctx { hands, same, shows, deals: vec![Deal { cards, weight, next }], raise_mult: 3.0, quadratic: false, net: None };
    let lf = ctx.leaf_features(0, &board, pot, stack, &w0, &w1);
    let [f0, f1] = lf.f;
    let [e0, e1] = lf.e;
    Ok((lf.ctx, f0, f1, e0, e1))
}

#[pymodule]
fn solver_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_class::<ValueNet>()?;
    m.add_function(wrap_pyfunction!(leaf_features, m)?)?;
    Ok(())
}
