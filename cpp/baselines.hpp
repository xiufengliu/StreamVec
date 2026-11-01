// Maintenance baselines that share the StreamVec graph code:
//  * RebuildBaseline   -- periodic full rebuild, queries a (stale) snapshot.
//  * DeltaOverlay      -- base graph + small delta index, periodic merge.
//  (replace_deleted / slot-reuse and route-through append are flags on
//   StreamVecIndex; see index.hpp.)
#pragma once
#include "index.hpp"
#include <chrono>

namespace streamvec {

// Two-level base+delta overlay: new inserts land in a small delta index that is
// periodically merged into the base, preserving freshness without per-op rebuild.
struct DeltaOverlay {
  Params p; StreamVecIndex base, delta; int merge_interval, since = 0;
  std::unordered_map<idx_t, int> where;            // ext -> 0 base / 1 delta
  std::unordered_map<idx_t, std::vector<float>> ov; // ext -> vector (for exact merge)
  int merges = 0;

  static Params mk(Params q) { q.skip_tombstones = true; return q; }
  DeltaOverlay(Params p_, int R) : p(p_), base(mk(p_)), delta(mk(p_)), merge_interval(R) {
    base.no_repair = true; delta.no_repair = true;
  }
  void insert(idx_t ext, const float* v) {
    if (where.count(ext)) { update(ext, v); return; }
    delta.insert(ext, v); where[ext] = 1; ov[ext].assign(v, v + p.dim); since++; maybe_merge();
  }
  void remove(idx_t ext) {
    auto it = where.find(ext); if (it == where.end()) return;
    (it->second ? delta : base).remove(ext); ov.erase(ext); where.erase(it); since++; maybe_merge();
  }
  void update(idx_t ext, const float* v) {
    auto it = where.find(ext);
    if (it != where.end()) { (it->second ? delta : base).remove(ext); where.erase(it); }
    delta.insert(ext, v); where[ext] = 1; ov[ext].assign(v, v + p.dim); since++; maybe_merge();
  }
  void maybe_merge() { if (since >= merge_interval) { merge(); since = 0; } }
  void merge() {
    std::vector<idx_t> eids; std::vector<float> data;
    for (auto& kv : where) { eids.push_back(kv.first);
      auto& v = ov[kv.first]; data.insert(data.end(), v.begin(), v.end()); }
    int N = eids.size();
    StreamVecIndex nb(mk(p)); nb.no_repair = true;
    if (N > 0) {
      nb.bulk_load_parallel(data.data(), N);           // parallel rebuild of base
      // remap contiguous internal ext (0..N-1) back to real external ids
      std::unordered_map<idx_t, idx_t> e2i; std::vector<idx_t> i2e(N);
      for (int i = 0; i < N; ++i) { i2e[i] = eids[i]; }
      std::unordered_map<idx_t, std::vector<float>> norig;
      // bulk_load_parallel set ext ids to 0..N-1; relabel to eids
      std::unordered_map<idx_t, idx_t> ne2i; std::vector<idx_t> ni2e(nb.int2ext.size(), -1);
      for (auto& kv : nb.ext2int) { idx_t in = kv.second; idx_t real = eids[kv.first]; ne2i[real] = in; ni2e[in] = real; norig[real] = ov[real]; }
      nb.ext2int = ne2i; nb.int2ext = ni2e; nb.orig = norig;
    }
    base = std::move(nb); delta = StreamVecIndex(mk(p)); delta.no_repair = true;
    where.clear(); for (idx_t e : eids) where[e] = 0;
    merges++;
  }
  std::vector<idx_t> search(const float* q, int k, int ef, Scratch* sc = nullptr) {
    auto a = base.search(q, ef, ef, sc); auto b = delta.search(q, ef, ef, sc);
    std::unordered_set<idx_t> cand(a.begin(), a.end()); cand.insert(b.begin(), b.end());
    std::vector<std::pair<float, idx_t>> sc2;
    for (idx_t e : cand) { auto it = ov.find(e); if (it == ov.end()) continue;
      sc2.push_back({l2sqr(q, it->second.data(), p.dim), e}); }
    std::partial_sort(sc2.begin(), sc2.begin() + std::min((int)sc2.size(), k), sc2.end());
    std::vector<idx_t> out; for (int i = 0; i < std::min((int)sc2.size(), k); ++i) out.push_back(sc2[i].second);
    return out;
  }
  void live_set(std::vector<idx_t>& ids, std::vector<float>& mat) {
    ids.clear(); mat.clear();
    for (auto& kv : where) { ids.push_back(kv.first);
      auto& v = ov[kv.first]; mat.insert(mat.end(), v.begin(), v.end()); }
  }
  int n_tomb() { return base.stats().n_tomb + delta.stats().n_tomb; }
};

// Periodic full rebuild that queries the last rebuilt snapshot (staleness source).
struct RebuildBaseline {
  Params p; StreamVecIndex* idx; int R, since = 0, rebuilds = 0;
  std::unordered_map<idx_t, std::vector<float>> live;   // current truth (applied immediately)
  RebuildBaseline(Params p_, int R_) : p(p_), R(R_) { idx = new StreamVecIndex(mk(p)); build(); }
  static Params mk(Params q) { q.skip_tombstones = true; return q; }
  void build() { delete idx; idx = new StreamVecIndex(mk(p)); idx->no_repair = true;
    int N = live.size();
    if (N > 0) {
      std::vector<idx_t> eids; std::vector<float> data;
      for (auto& kv : live) { eids.push_back(kv.first);
        data.insert(data.end(), kv.second.begin(), kv.second.end()); }
      idx->bulk_load_parallel(data.data(), N);   // parallel rebuild
      std::unordered_map<idx_t, idx_t> ne2i; std::vector<idx_t> ni2e(idx->int2ext.size(), -1);
      std::unordered_map<idx_t, std::vector<float>> norig;
      for (auto& kv : idx->ext2int) { idx_t in = kv.second; idx_t real = eids[kv.first];
        ne2i[real] = in; ni2e[in] = real; norig[real] = live[real]; }
      idx->ext2int = ne2i; idx->int2ext = ni2e; idx->orig = norig;
    }
    since = 0; rebuilds++; }
  void insert(idx_t e, const float* v) { live[e].assign(v, v + p.dim); since++; maybe(); }
  void remove(idx_t e) { live.erase(e); since++; maybe(); }
  void update(idx_t e, const float* v) { live[e].assign(v, v + p.dim); since++; maybe(); }
  void maybe() { if (since >= R) build(); }
  std::vector<idx_t> search(const float* q, int k, int ef, Scratch* sc = nullptr) { return idx->search(q, k, ef, sc); }
  void live_set(std::vector<idx_t>& ids, std::vector<float>& mat) {  // current truth
    ids.clear(); mat.clear();
    for (auto& kv : live) { ids.push_back(kv.first); mat.insert(mat.end(), kv.second.begin(), kv.second.end()); }
  }
};

// FreshDiskANN/FreshVamana-style delete consolidation, applied to the same HNSW
// graph (DeleteConsolidation in Singh et al., FreshDiskANN, arXiv:2105.09613). Deleted nodes stay in
// the graph and are routed through (excluded from results) until the deleted
// fraction exceeds tau_c; then every live node with a deleted out-neighbour v
// gets the candidate set  live(N(p)) U live(N(v))  re-pruned with the same
// diversifying heuristic, and the deleted nodes are physically compacted away.
struct ConsolidateIndex {
  Params p; StreamVecIndex ix; double tau_c; int consolidations = 0;
  double consolidate_ms = 0;
  static Params mk(Params q) { q.skip_tombstones = false; return q; }
  ConsolidateIndex(Params p_, double tc) : p(p_), ix(mk(p_)), tau_c(tc) { ix.no_repair = true; ix.track_dirty = false; }

  static std::vector<idx_t> heur(const HNSWGraph& g, std::vector<std::pair<float, idx_t>> cand, int m) {
    std::sort(cand.begin(), cand.end());
    std::vector<idx_t> sel, pruned;
    for (auto& [db, c] : cand) {
      if ((int)sel.size() >= m) break;
      bool ok = true;
      for (idx_t s : sel) if (l2sqr(g.vptr(c), g.vptr(s), g.dim) < db) { ok = false; break; }
      if (ok) sel.push_back(c); else pruned.push_back(c);
    }
    for (idx_t c : pruned) { if ((int)sel.size() >= m) break; sel.push_back(c); }
    return sel;
  }
  int n_tomb_now() { int c = 0; for (int i = 0; i < ix.g.n; ++i) if (ix.g.tomb[i]) c++; return c; }
  void maybe() {
    HNSWGraph& g = ix.g; if (g.n == 0) return;
    if ((double)n_tomb_now() / g.n > tau_c) consolidate();
  }
  void consolidate() { consolidate_index(ix, p, consolidations); consolidations++; }
  static void consolidate_index(StreamVecIndex& ix, const Params& p, int consolidations) {
    double t0 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    HNSWGraph& g = ix.g; int n = g.n;
    // 1) re-prune every live node that points at a deleted node (race-free:
    //    each p writes only its own list; deleted lists are read-only)
    std::vector<std::vector<std::vector<idx_t>>> newn(n);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int u = 0; u < n; ++u) {
      if (g.tomb[u]) continue;
      bool touched = false;
      for (auto& l : g.nbr[u]) for (idx_t x : l) if (g.tomb[x]) { touched = true; break; }
      if (!touched) continue;
      newn[u] = g.nbr[u];
      for (int layer = 0; layer < (int)g.nbr[u].size(); ++layer) {
        int mmax = layer == 0 ? g.M0 : g.M;
        std::unordered_set<idx_t> C;
        for (idx_t x : g.nbr[u][layer]) {
          if (!g.tomb[x]) C.insert(x);
          else if (layer < (int)g.nbr[x].size())
            for (idx_t y : g.nbr[x][layer]) if (!g.tomb[y] && y != u) C.insert(y);
        }
        std::vector<std::pair<float, idx_t>> cand;
        for (idx_t c : C) cand.push_back({l2sqr(g.vptr(u), g.vptr(c), g.dim), c});
        newn[u][layer] = heur(g, cand, mmax);
      }
    }
    for (int u = 0; u < n; ++u) if (!newn[u].empty()) g.nbr[u] = std::move(newn[u]);
    // 2) compact: drop deleted nodes, remap ids
    std::vector<idx_t> remap(n, -1); int nn = 0;
    for (int u = 0; u < n; ++u) if (!g.tomb[u]) remap[u] = nn++;
    HNSWGraph ng(p.dim, p.M, p.efc, p.seed + consolidations + 1);
    ng.reserve(nn + 16);
    std::vector<idx_t> ni2e(nn, -1);
    for (int u = 0; u < n; ++u) if (remap[u] >= 0) {
      idx_t id = ng.add_storage(g.vptr(u)); ng.level[id] = g.level[u]; ng.tomb[id] = 0;
      ng.nbr[id].resize(g.nbr[u].size());
      for (int l = 0; l < (int)g.nbr[u].size(); ++l)
        for (idx_t x : g.nbr[u][l]) if (remap[x] >= 0) ng.nbr[id][l].push_back(remap[x]);
      if (u < (int)ix.int2ext.size()) ni2e[id] = ix.int2ext[u];
    }
    int bl = -1; idx_t best = -1;
    for (int i = 0; i < nn; ++i) if (ng.level[i] > bl) { bl = ng.level[i]; best = i; }
    ng.ep = best; ng.maxlevel = bl;
    std::unordered_map<idx_t, idx_t> ne2i;
    for (auto& kv : ix.ext2int) if (remap[kv.second] >= 0) ne2i[kv.first] = remap[kv.second];
    ix.g = std::move(ng); ix.int2ext = ni2e; ix.ext2int = ne2i; ix.freelist.clear();
    DirtySet nd; nd.policy = ix.dirty.policy; ix.dirty = std::move(nd);
    (void)t0;
  }
  void insert(idx_t e, const float* v) { ix.insert(e, v); }
  void remove(idx_t e) { ix.remove(e); maybe(); }
  void update(idx_t e, const float* v) { ix.update(e, v); maybe(); }
  std::vector<idx_t> search(const float* q, int k, int ef, Scratch* sc = nullptr) { return ix.search(q, k, ef, sc); }
  void live_set(std::vector<idx_t>& a, std::vector<float>& b) { ix.live_set(a, b); }
};

} // namespace streamvec
