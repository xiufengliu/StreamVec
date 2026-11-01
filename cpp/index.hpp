// StreamVecIndex + baselines (periodic rebuild, append, delta-overlay, slot-reuse).
#pragma once
#include "streamvec.hpp"
#include <functional>

namespace streamvec {

struct Params {
  int dim, M = 16, efc = 100, ef_repair = 64;
  double tomb_threshold = 0.15, deg_deficit_threshold = 0.3, prune_fraction = 0.5;
  int repair_radius = 2;       // beam-search seed hops
  double hotness_alpha = 0.0;  // >0 enables query-driven priority
  int dirty_policy = 0;        // DirtySet::Policy
  bool skip_tombstones = true; // false => route-through (append baseline)
  uint64_t seed = 42;
};

struct StreamVecIndex {
  Params p; HNSWGraph g; DirtySet dirty;
  std::unordered_map<idx_t, idx_t> ext2int;   // external id -> live internal node
  std::vector<idx_t> int2ext;                 // internal node -> external id
  std::unordered_map<idx_t, std::vector<float>> orig;  // external id -> vector (for rebuild)
  std::unordered_map<idx_t, long> visits;
  bool no_repair = false;
  bool track_dirty = true;           // false => skip DirtySet bookkeeping (consolidation baseline)
  long forced_repairs = 0, sched_repairs = 0;
  std::function<void(StreamVecIndex&)> compactor;  // if set, the trigger compacts instead of fully rebuilding
  bool reuse_slots = false;          // replace_deleted: recycle tombstoned slots
  std::vector<idx_t> freelist;
  std::mt19937 rng;

  StreamVecIndex(Params p_) : p(p_), g(p_.dim, p_.M, p_.efc, p_.seed), rng(p_.seed) {
    dirty.policy = (DirtySet::Policy)p.dirty_policy;
  }

  void set_ext(idx_t internal, idx_t ext) {
    if ((int)int2ext.size() <= internal) int2ext.resize(internal + 1, -1);
    int2ext[internal] = ext; ext2int[ext] = internal;
  }

  void insert(idx_t ext, const float* v) {
    if (ext2int.count(ext)) { update(ext, v); return; }
    idx_t in;
    if (reuse_slots && !freelist.empty()) {     // recycle a tombstoned slot
      in = freelist.back(); freelist.pop_back();
      std::memcpy(&g.vec[(size_t)in * p.dim], v, p.dim * sizeof(float));
      g.tomb[in] = 0; int nl = g.rand_level(); g.level[in] = nl; g.nbr[in].assign(nl + 1, {});
      g.link_node(in); if (nl > g.maxlevel) { g.maxlevel = nl; g.ep = in; }
    } else { in = g.insert(v); }
    set_ext(in, ext);
    orig[ext].assign(v, v + p.dim);
  }

  // Chunked parallel bulk load (external ids 0..N-1).  Within a chunk, nodes link
  // only to already-frozen nodes, so forward edges are race-free; reverse edges
  // to frozen nodes are taken under per-node locks.  Returns build seconds.
  void bulk_load_parallel(const float* data, int N, int chunk = 4000) {
    g.reserve(N + 16);
    for (int i = 0; i < N; ++i) {
      idx_t id = g.add_storage(data + (size_t)i * p.dim);
      int nl = g.rand_level(); g.level[id] = nl; g.nbr[id].assign(nl + 1, {}); g.tomb[id] = 0;
      if (nl > g.maxlevel) { g.maxlevel = nl; g.ep = id; }
      set_ext(id, i); orig[i].assign(data + (size_t)i * p.dim, data + (size_t)(i + 1) * p.dim);
    }
    g.enable_concurrency(N + 16);
    int S = std::min(N, 2000);
    for (int i = 0; i < S; ++i) if (i != g.ep) g.link_node(i);
#ifdef _OPENMP
    for (int start = S; start < N; start += chunk) {
      int end = std::min(N, start + chunk);
      #pragma omp parallel
      {
        Scratch sc;
        #pragma omp for schedule(dynamic, 64)
        for (int i = start; i < end; ++i) if (i != g.ep) g.link_node(i, &sc);
      }
    }
#else
    for (int i = S; i < N; ++i) if (i != g.ep) g.link_node(i);
#endif
    g.concurrent = false;  // fast single-thread path for subsequent streaming
  }

  void mark_neighbors_dirty(idx_t del) {
    if (g.nbr[del].empty()) return;
    std::unordered_set<idx_t> aff;
    for (auto& layer : g.nbr[del])
      for (idx_t u : layer) if (!g.tomb[u]) aff.insert(u);
    for (idx_t u : aff) {
      int def = g.M0 - g.live_deg(u);
      double pr = def + (p.hotness_alpha > 0 ? p.hotness_alpha * visits[int2ext[u]] : 0.0);
      dirty.mark(u, def, pr);
      // tombstone-pruning: force-repair badly damaged nodes
      if (!no_repair && !g.nbr[u].empty() && !g.nbr[u][0].empty()) {
        int dead = 0; for (idx_t x : g.nbr[u][0]) if (g.tomb[x]) dead++;
        if ((double)dead / g.nbr[u][0].size() >= p.prune_fraction) { repair_node(u); dirty.remove(u); forced_repairs++; }
      }
    }
  }

  void replace_entry_point() {
    idx_t best = -1; int bl = -1;
    for (int i = 0; i < g.n; ++i) if (!g.tomb[i] && g.level[i] > bl) { bl = g.level[i]; best = i; }
    g.ep = best; g.maxlevel = best < 0 ? -1 : bl;
  }

  void remove(idx_t ext) {
    auto it = ext2int.find(ext); if (it == ext2int.end()) return;
    idx_t in = it->second; if (g.tomb[in]) return;
    g.tomb[in] = 1;
    if (in == g.ep) replace_entry_point();
    dirty.remove(in);                          // a tombstoned node is never repaired
    if (reuse_slots) freelist.push_back(in);
    else if (track_dirty) mark_neighbors_dirty(in);
    ext2int.erase(it); orig.erase(ext);
  }

  void update(idx_t ext, const float* v) {
    auto it = ext2int.find(ext);
    if (it != ext2int.end()) { idx_t old = it->second; g.tomb[old] = 1; dirty.remove(old);
      if (old == g.ep) replace_entry_point(); if (track_dirty) mark_neighbors_dirty(old); ext2int.erase(it); }
    idx_t in = g.insert(v); set_ext(in, ext); orig[ext].assign(v, v + p.dim);
  }

  // localized repair of one node
  void repair_node(idx_t u) {
    if (g.tomb[u] || g.nbr[u].empty()) return;
    const float* q = g.vptr(u);
    for (int layer = g.level[u]; layer >= 0; --layer) {
      int mmax = layer == 0 ? g.M0 : g.M;
      auto& l = g.nbr[u][layer];
      std::vector<idx_t> live; for (idx_t x : l) if (!g.tomb[x]) live.push_back(x);
      if ((int)live.size() >= mmax) { g.lock_node(u); l = live; g.unlock_node(u); continue; }
      std::unordered_set<idx_t> seeds(live.begin(), live.end());
      if (p.repair_radius >= 2)
        for (idx_t ln : live) if (layer < (int)g.nbr[ln].size())
          for (idx_t nn : g.nbr[ln][layer]) if (!g.tomb[nn]) seeds.insert(nn);
      if (seeds.empty() && g.ep >= 0) seeds.insert(g.ep);
      std::vector<idx_t> seedv(seeds.begin(), seeds.end());
      auto cand = seedv.empty() ? std::vector<std::pair<float, idx_t>>{}
                                : g.search_layer(q, seedv, p.ef_repair, layer, true);
      std::vector<std::pair<float, idx_t>> pool;
      std::unordered_set<idx_t> have;
      for (idx_t x : live) { pool.push_back({l2sqr(q, g.vptr(x), p.dim), x}); have.insert(x); }
      for (auto& [d, c] : cand) if (c != u && !have.count(c)) { have.insert(c); pool.push_back({d, c}); }
      auto chosen = g.select_heuristic(u, pool, mmax);
      g.lock_node(u); g.nbr[u][layer] = chosen; g.unlock_node(u);
      for (idx_t nb : chosen) { if (nb == u) continue;
        if (layer <= g.level[nb]) { g.connect(nb, u, layer); g.prune(nb, layer); } }
    }
    g.st.repairs++;
  }

  int repair(int budget) {
    if (no_repair) return 0;
    if (budget < 0) budget = dirty.size();
    auto nodes = dirty.pop(budget);
    for (idx_t u : nodes) repair_node(u);
    sched_repairs += nodes.size();
    return nodes.size();
  }

  bool should_rebuild() {
    int nlive = (int)ext2int.size();
    if (nlive <= 0) return false;
    double tr = (double)(g.n - nlive_count()) / std::max(1, g.n);
    if (tr > p.tomb_threshold) return true;
    if (dirty.avg_deficit() > p.deg_deficit_threshold * g.M0) return true;
    return false;
  }
  // fraction of damaged live nodes (>=1 tombstoned level-0 neighbour) that are in the DirtySet
  double dirty_coverage(long* damaged_out = nullptr) {
    long dam = 0, cov = 0;
    for (int i = 0; i < g.n; ++i) { if (g.tomb[i] || g.nbr[i].empty()) continue;
      bool d = false; for (idx_t x : g.nbr[i][0]) if (g.tomb[x]) { d = true; break; }
      if (d) { dam++; if (dirty.mem.count(i)) cov++; } }
    if (damaged_out) *damaged_out = dam;
    return dam ? (double)cov / dam : 1.0;
  }
  int nlive_count() { int c = 0; for (int i = 0; i < g.n; ++i) if (!g.tomb[i]) c++; return c; }

  void rebuild() {
    std::vector<std::pair<idx_t, std::vector<float>>> live;
    for (auto& kv : ext2int) live.push_back({kv.first, orig[kv.first]});
    std::shuffle(live.begin(), live.end(), rng);
    int N = live.size();
    HNSWGraph ng(p.dim, p.M, p.efc, p.seed);
    DirtySet nd; nd.policy = dirty.policy;
    g = std::move(ng); dirty = std::move(nd);
    ext2int.clear(); int2ext.clear(); freelist.clear();
    std::unordered_map<idx_t, std::vector<float>> norig;
    g.reserve(N + 16);
    for (int i = 0; i < N; ++i) {
      idx_t id = g.add_storage(live[i].second.data());
      int nl = g.rand_level(); g.level[id] = nl; g.nbr[id].assign(nl + 1, {}); g.tomb[id] = 0;
      if (nl > g.maxlevel) { g.maxlevel = nl; g.ep = id; }
      set_ext(id, live[i].first); norig[live[i].first] = live[i].second;
    }
    g.enable_concurrency(N + 16);
    int S = std::min(N, 2000);
    for (int i = 0; i < S; ++i) if (i != g.ep) g.link_node(i);
#ifdef _OPENMP
    for (int start = S; start < N; start += 4000) { int end = std::min(N, start + 4000);
      #pragma omp parallel
      { Scratch sc;
        #pragma omp for schedule(dynamic, 64)
        for (int i = start; i < end; ++i) if (i != g.ep) g.link_node(i, &sc); } }
#else
    for (int i = S; i < N; ++i) if (i != g.ep) g.link_node(i);
#endif
    g.concurrent = false;  // back to fast single-thread path for streaming
    orig = std::move(norig);
    g.st.rebuilds++;
  }

  std::vector<idx_t> search(const float* q0, int k, int ef, Scratch* sc = nullptr) {
    std::vector<float> q(q0, q0 + p.dim);
    if (g.ep < 0) return {};
    bool skip = p.skip_tombstones;
    std::vector<idx_t> cur{g.ep};
    for (int lc = g.maxlevel; lc > 0; --lc) {
      auto r = g.search_layer(q.data(), cur, 1, lc, skip, sc);
      if (!r.empty()) cur = {r[0].second};
    }
    auto r = g.search_layer(q.data(), cur, std::max(ef, k), 0, skip, sc);
    std::vector<idx_t> out; std::unordered_set<idx_t> seen;
    for (auto& [d, in] : r) { if (g.tomb[in]) continue; idx_t e = int2ext[in];
      if (seen.count(e)) continue; seen.insert(e);
      if (p.hotness_alpha > 0) visits[e]++;
      out.push_back(e); if ((int)out.size() >= k) break; }
    return out;
  }

  Stats stats() { Stats s = g.st; s.n_live = (int)ext2int.size();
    s.n_tomb = g.n - nlive_count(); s.n_dirty = dirty.size();
    s.tombstone_ratio = (double)s.n_tomb / std::max(1, g.n); return s; }

  // live (ext -> vector) for exact ground-truth construction
  void live_set(std::vector<idx_t>& ids, std::vector<float>& mat) {
    ids.clear(); mat.clear();
    for (auto& kv : ext2int) { ids.push_back(kv.first);
      auto& v = orig[kv.first]; mat.insert(mat.end(), v.begin(), v.end()); }
  }
  size_t memory_bytes() {
    // index memory: vector slots (live + tombstoned) + adjacency lists.
    // The ext->vector map 'orig' is a harness convenience that every system
    // could replace by reading live slots, so it is not charged to any system.
    size_t b = (size_t)g.n * p.dim * 4;
    for (auto& nn : g.nbr) for (auto& l : nn) b += l.size() * 4;
    return b;
  }
};

} // namespace streamvec
