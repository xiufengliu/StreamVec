// Industry-practice baselines requested by reviewers:
//  * RebuildBaseline   -- periodic full rebuild, queries a (stale) snapshot.
//  * DeltaOverlay      -- base graph + small delta index, periodic merge (Q1).
//  (replace_deleted / slot-reuse and route-through append are flags on
//   StreamVecIndex; see index.hpp.)
#pragma once
#include "index.hpp"

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

} // namespace streamvec
