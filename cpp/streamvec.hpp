// StreamVec: in-memory streaming maintenance for HNSW-style ANN graphs.
// C++17, NEON SIMD distance, OpenMP-parallel build. Single-header reference.
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace streamvec {

using idx_t = int32_t;

// ---------------------- SIMD distance (L2) ---------------------------------
inline float l2sqr(const float* a, const float* b, int d) {
#if defined(__ARM_NEON)
  float32x4_t acc = vdupq_n_f32(0.f);
  int i = 0;
  for (; i + 4 <= d; i += 4) {
    float32x4_t va = vld1q_f32(a + i), vb = vld1q_f32(b + i);
    float32x4_t df = vsubq_f32(va, vb);
    acc = vmlaq_f32(acc, df, df);
  }
  float s = vaddvq_f32(acc);
  for (; i < d; ++i) { float t = a[i] - b[i]; s += t * t; }
  return s;
#else
  float s = 0;
  for (int i = 0; i < d; ++i) { float t = a[i] - b[i]; s += t * t; }
  return s;
#endif
}

struct Stats {
  long dist_calls = 0, neighbor_checks = 0, heap_ops = 0;
  int n_live = 0, n_tomb = 0, n_dirty = 0, rebuilds = 0, repairs = 0;
  double tombstone_ratio = 0;
};

// per-thread scratch: version-stamped visited set (avoids hash-set alloc per query
// and makes concurrent queries thread-safe)
struct Scratch {
  std::vector<uint32_t> stamp; uint32_t cur = 0;
  long dist_calls = 0, neighbor_checks = 0, heap_ops = 0;
  void ensure(int n) { if ((int)stamp.size() < n) stamp.resize(n, 0); }
  void reset() { if (++cur == 0) { std::fill(stamp.begin(), stamp.end(), 0); cur = 1; } }
  inline bool seen(idx_t i) { return stamp[i] == cur; }
  inline void mark(idx_t i) { stamp[i] = cur; }
};

// ---------------------- HNSW graph with StreamVec maintenance --------------
struct Spin {
  std::atomic_flag f = ATOMIC_FLAG_INIT;
  void lock() { while (f.test_and_set(std::memory_order_acquire)) {} }
  void unlock() { f.clear(std::memory_order_release); }
};

struct HNSWGraph {
  int dim, M, M0, efc;
  double mL;
  std::mt19937 rng;
  std::vector<float> vec;                  // n_cap*dim, row-major
  int n = 0, cap = 0;
  std::vector<int> level;                  // top level per node
  std::vector<std::vector<std::vector<idx_t>>> nbr;  // nbr[node][layer] = neighbors
  std::vector<char> tomb;                  // tombstone flag
  idx_t ep = -1; int maxlevel = -1;
  Stats st;
  bool concurrent = false;                 // when true, neighbor reads/writes are locked
  std::unique_ptr<Spin[]> locks;
  inline void lock_node(idx_t i) { if (concurrent) locks[i].lock(); }
  inline void unlock_node(idx_t i) { if (concurrent) locks[i].unlock(); }

  HNSWGraph(int dim_, int M_, int efc_, uint64_t seed = 42)
      : dim(dim_), M(M_), M0(2 * M_), efc(efc_), rng(seed) {
    mL = 1.0 / std::log(std::max(2, M));
  }

  void reserve(int c) {
    cap = c; vec.resize((size_t)c * dim);
    level.reserve(c); nbr.reserve(c); tomb.reserve(c);
  }
  void enable_concurrency(int c) { locks.reset(new Spin[c]); concurrent = true; }
  inline const float* vptr(idx_t i) const { return &vec[(size_t)i * dim]; }
  inline float dist(const float* q, idx_t i) { st.dist_calls++; return l2sqr(q, vptr(i), dim); }
  int rand_level() { std::uniform_real_distribution<double> u(0, 1);
    return (int)(-std::log(std::max(1e-12, u(rng))) * mL); }

  idx_t add_storage(const float* v) {
    if (n >= cap) { cap = std::max(16, cap * 2); vec.resize((size_t)cap * dim); }
    std::memcpy(&vec[(size_t)n * dim], v, dim * sizeof(float));
    if ((int)level.size() <= n) { level.resize(n + 1); nbr.resize(n + 1); tomb.resize(n + 1, 0); }
    return n++;
  }

  Scratch _scr;  // default single-thread scratch

  // beam search within one layer; skip_t => never traverse/return tombstones
  std::vector<std::pair<float, idx_t>>
  search_layer(const float* q, const std::vector<idx_t>& eps, int ef, int layer,
               bool skip_t, Scratch* sc = nullptr) {
    Scratch& s = sc ? *sc : _scr;
    s.ensure(n); s.reset();
    std::priority_queue<std::pair<float, idx_t>> resmax;     // max-heap (farthest top)
    std::priority_queue<std::pair<float, idx_t>,
        std::vector<std::pair<float, idx_t>>, std::greater<>> cand;  // min-heap
    for (idx_t e : eps) {
      float d = l2sqr(q, vptr(e), dim); s.dist_calls++; s.mark(e);
      cand.push({d, e});
      if (!(skip_t && tomb[e])) resmax.push({d, e});
      s.heap_ops += 2;
    }
    while (!cand.empty()) {
      auto [cd, cur] = cand.top(); cand.pop();
      if (!resmax.empty() && cd > resmax.top().first && (int)resmax.size() >= ef) break;
      if (skip_t && tomb[cur] && std::find(eps.begin(), eps.end(), cur) == eps.end()) continue;
      if (layer >= (int)nbr[cur].size()) continue;
      std::vector<idx_t> ns_copy; const std::vector<idx_t>* nsp;
      if (concurrent) { lock_node(cur); ns_copy = nbr[cur][layer]; unlock_node(cur); nsp = &ns_copy; }
      else nsp = &nbr[cur][layer];
      const auto& ns = *nsp;
      s.neighbor_checks += ns.size();
      for (idx_t nb : ns) {
        if (s.seen(nb)) continue; s.mark(nb);
        if (skip_t && tomb[nb]) continue;
        float d = l2sqr(q, vptr(nb), dim); s.dist_calls++;
        if ((int)resmax.size() < ef || d < resmax.top().first) {
          cand.push({d, nb}); s.heap_ops++;
          if (!(skip_t && tomb[nb])) { resmax.push({d, nb}); s.heap_ops++;
            if ((int)resmax.size() > ef) { resmax.pop(); s.heap_ops++; } }
        }
      }
    }
    std::vector<std::pair<float, idx_t>> out;
    while (!resmax.empty()) { out.push_back(resmax.top()); resmax.pop(); }
    std::reverse(out.begin(), out.end());  // ascending distance
    return out;
  }
  void sync_counters() { st.dist_calls = _scr.dist_calls; st.neighbor_checks = _scr.neighbor_checks; st.heap_ops = _scr.heap_ops; }

  // HNSW heuristic neighbor selection (diversifying)
  std::vector<idx_t> select_heuristic(idx_t base, std::vector<std::pair<float, idx_t>> cand, int m) {
    (void)base;
    if ((int)cand.size() <= m) { std::sort(cand.begin(), cand.end());
      std::vector<idx_t> r; for (auto& c : cand) r.push_back(c.second); return r; }
    std::sort(cand.begin(), cand.end());
    std::vector<idx_t> sel; std::vector<idx_t> pruned;
    for (auto& [db, c] : cand) {
      if ((int)sel.size() >= m) break;
      bool ok = true;
      for (idx_t s : sel) { if (l2sqr(vptr(c), vptr(s), dim) < db) { ok = false; break; } }
      st.dist_calls += sel.size();
      if (ok) sel.push_back(c); else pruned.push_back(c);
    }
    for (idx_t c : pruned) { if ((int)sel.size() >= m) break; sel.push_back(c); }
    return sel;
  }

  void connect(idx_t a, idx_t b, int layer) {
    if (layer >= (int)nbr[a].size()) return;
    lock_node(a);
    auto& l = nbr[a][layer];
    if (std::find(l.begin(), l.end(), b) == l.end()) l.push_back(b);
    unlock_node(a);
  }
  void prune(idx_t a, int layer) {
    int mmax = layer == 0 ? M0 : M;
    lock_node(a);
    if (layer >= (int)nbr[a].size() || (int)nbr[a][layer].size() <= mmax + 1 + mmax / 4) { unlock_node(a); return; }
    std::vector<idx_t> l = nbr[a][layer];   // snapshot under lock
    unlock_node(a);
    std::vector<std::pair<float, idx_t>> cand;
    const float* base = vptr(a);
    for (idx_t x : l) cand.push_back({l2sqr(base, vptr(x), dim), x});
    auto chosen = select_heuristic(a, cand, mmax);
    lock_node(a); nbr[a][layer] = chosen; unlock_node(a);
  }

  // link an already-stored node into the graph (used by insert and parallel build)
  void link_node(idx_t id, Scratch* sc = nullptr) {
    int nl = level[id];
    idx_t e = ep; int ml = maxlevel;
    if (e < 0 || e == id) return;
    std::vector<idx_t> cur{e};
    const float* q = vptr(id);
    for (int lc = ml; lc > nl; --lc) {
      auto r = search_layer(q, cur, 1, lc, false, sc);
      if (!r.empty()) cur = {r[0].second};
    }
    for (int lc = std::min(nl, ml); lc >= 0; --lc) {
      auto cand = search_layer(q, cur, efc, lc, false, sc);
      int m = lc == 0 ? M0 : M;
      auto chosen = select_heuristic(id, cand, m);
      std::vector<idx_t> valid;
      for (idx_t c : chosen) if (level[c] >= lc) valid.push_back(c);
      lock_node(id); nbr[id][lc] = valid; unlock_node(id);
      for (idx_t c : valid) { connect(c, id, lc); prune(c, lc); }
      if (!cand.empty()) { cur.clear(); for (auto& p : cand) cur.push_back(p.second); }
    }
  }

  idx_t insert(const float* v, int lvl = -1) {
    idx_t id = add_storage(v);
    int nl = lvl < 0 ? rand_level() : lvl;
    level[id] = nl; nbr[id].assign(nl + 1, {}); tomb[id] = 0;
    if (ep < 0) { ep = id; maxlevel = nl; return id; }
    link_node(id);
    if (nl > maxlevel) { maxlevel = nl; ep = id; }
    return id;
  }

  int live_deg(idx_t u) { int c = 0; if (nbr[u].empty()) return 0;
    for (idx_t x : nbr[u][0]) if (!tomb[x]) c++; return c; }
};

// ---------------------- DirtySet (degree-deficit priority) -----------------
struct DirtySet {
  enum Policy { PRIORITY, FIFO, RANDOM, EAGER } policy = PRIORITY;
  std::priority_queue<std::tuple<double, long, idx_t>> heap;  // (priority,ctr,id)
  std::unordered_map<idx_t, double> prio;   // authoritative heap key
  std::unordered_map<idx_t, int> deficit;
  std::vector<idx_t> fifo;
  std::unordered_set<idx_t> mem;
  long ctr = 0; std::mt19937 rng{7};

  void mark(idx_t id, int def, double p) {
    if (def <= 0) { if (mem.count(id)) { mem.erase(id); prio.erase(id); deficit.erase(id);} return; }
    deficit[id] = def; prio[id] = p;
    if (!mem.count(id)) { mem.insert(id); fifo.push_back(id); }
    heap.push({p, ctr++, id});
  }
  void remove(idx_t id){ mem.erase(id); prio.erase(id); deficit.erase(id); }
  int size() const { return mem.size(); }
  double avg_deficit() const { if (deficit.empty()) return 0; double s=0; for(auto&kv:deficit)s+=kv.second; return s/deficit.size(); }
  std::vector<idx_t> pop(int k) {
    std::vector<idx_t> out;
    if (policy == EAGER) k = mem.size();
    if (policy == PRIORITY) {
      while (!heap.empty() && (int)out.size() < k) {
        auto [p, c, id] = heap.top(); heap.pop();
        if (!mem.count(id)) continue;
        if (p != prio[id]) continue;
        out.push_back(id); remove(id);
      }
    } else if (policy == FIFO) {
      while (!fifo.empty() && (int)out.size() < k) { idx_t id = fifo.front();
        fifo.erase(fifo.begin()); if (mem.count(id)) { out.push_back(id); remove(id);} }
    } else {
      std::vector<idx_t> m(mem.begin(), mem.end());
      std::shuffle(m.begin(), m.end(), rng);
      for (int i = 0; i < (int)m.size() && (int)out.size() < k; ++i) { out.push_back(m[i]); remove(m[i]); }
    }
    return out;
  }
};

} // namespace streamvec
