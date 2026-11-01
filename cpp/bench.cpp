// StreamVec C++ experiment harness. Modes selected by argv[1].
#include "index.hpp"
#include "baselines.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
using namespace streamvec;
using Clock = std::chrono::high_resolution_clock;
static double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

struct Data { int n, dim; std::vector<float> x; const float* row(int i) const { return &x[(size_t)i * dim]; } };
Data load_fbin(const std::string& path) {
  std::ifstream f(path, std::ios::binary); Data d; f.read((char*)&d.n, 4); f.read((char*)&d.dim, 4);
  d.x.resize((size_t)d.n * d.dim); f.read((char*)d.x.data(), (size_t)d.n * d.dim * 4); return d;
}

// brute-force exact kNN over a live matrix (ids+mat), parallel over queries
std::vector<std::vector<idx_t>> brute_gt(const std::vector<idx_t>& ids, const std::vector<float>& mat,
                                         const Data& Q, const std::vector<int>& qi, int dim, int k) {
  int nq = qi.size(); int nl = ids.size();
  std::vector<std::vector<idx_t>> gt(nq);
  #pragma omp parallel for schedule(dynamic)
  for (int j = 0; j < nq; ++j) {
    const float* q = Q.row(qi[j]);
    std::vector<std::pair<float, int>> d(nl);
    for (int i = 0; i < nl; ++i) d[i] = {l2sqr(q, &mat[(size_t)i * dim], dim), i};
    int kk = std::min(k, nl);
    std::partial_sort(d.begin(), d.begin() + kk, d.end());
    gt[j].resize(kk); for (int t = 0; t < kk; ++t) gt[j][t] = ids[d[t].second];
  }
  return gt;
}
double recall_at_k(const std::vector<std::vector<idx_t>>& pred, const std::vector<std::vector<idx_t>>& gt, int k) {
  long hit = 0, tot = 0;
  for (size_t j = 0; j < gt.size(); ++j) {
    std::unordered_set<idx_t> g(gt[j].begin(), gt[j].begin() + std::min((int)gt[j].size(), k));
    for (int t = 0; t < std::min((int)pred[j].size(), k); ++t) if (g.count(pred[j][t])) hit++;
    tot += g.size();
  }
  return tot ? (double)hit / tot : 0;
}

// ---- uniform system adapter over StreamVec / baselines ----
struct Sys {
  std::string name;
  std::function<void(int, int, const float*)> apply;   // (op:0ins1del2upd, ext, vec)
  std::function<std::vector<idx_t>(const float*, int, int, Scratch*)> search;
  std::function<void(std::vector<idx_t>&, std::vector<float>&)> live_set;
  std::function<void(long)> maintain;                  // per-op hook
  std::function<int()> ntomb;
  std::function<size_t()> mem;
  std::shared_ptr<void> hold;                          // keep underlying object alive
  std::function<std::string()> diag = []() { return std::string("-1,-1,-1,-1"); };  // forced,sched,coverage,damaged
};

Params base_params(int dim, int M = 16, int efc = 100) { Params p; p.dim = dim; p.M = M; p.efc = efc; return p; }

Sys make_streamvec(int dim, int M, int efc, int budget, long interval, int efr,
                   double tomb_th, int policy = 0, bool no_repair = false, bool reuse = false,
                   bool route_through = false, double hot = 0) {
  Params p = base_params(dim, M, efc); p.ef_repair = efr; p.tomb_threshold = tomb_th;
  p.dirty_policy = policy; p.hotness_alpha = hot; p.skip_tombstones = !route_through;
  auto ix = std::make_shared<StreamVecIndex>(p);
  ix->no_repair = no_repair; ix->reuse_slots = reuse;
  Sys s; s.hold = ix;
  s.name = reuse ? "ReplaceDeleted" : route_through ? "Append"
           : no_repair ? "StreamVec-NoRepair" : hot > 0 ? "StreamVec-Hot" : "StreamVec";
  auto* r = ix.get();
  s.apply = [r](int op, int e, const float* v) { if (op == 0) r->insert(e, v); else if (op == 1) r->remove(e); else r->update(e, v); };
  s.search = [r](const float* q, int k, int ef, Scratch* sc) { return r->search(q, k, ef, sc); };
  s.live_set = [r](std::vector<idx_t>& a, std::vector<float>& b) { r->live_set(a, b); };
  s.maintain = [r, budget, interval, no_repair](long n) {
    if (no_repair) return;
    if (interval > 0 && n % interval == 0) { r->repair(budget);
      if (r->should_rebuild()) { if (r->compactor) r->compactor(*r); else r->rebuild(); } }
  };
  s.ntomb = [r]() { return r->stats().n_tomb; };
  s.mem = [r]() { return r->memory_bytes(); };
  s.diag = [r]() { long dam = 0; double c = r->dirty_coverage(&dam); char b[96];
    snprintf(b, sizeof(b), "%ld,%ld,%.4f,%ld", r->forced_repairs, r->sched_repairs, c, dam); return std::string(b); };
  return s;
}
Sys make_rebuild(int dim, int M, int efc, int R) {
  auto ix = std::make_shared<RebuildBaseline>(base_params(dim, M, efc), R);
  Sys s; s.hold = ix; s.name = "Rebuild-R" + std::to_string(R); auto* r = ix.get();
  s.apply = [r](int op, int e, const float* v) { if (op == 0) r->insert(e, v); else if (op == 1) r->remove(e); else r->update(e, v); };
  s.search = [r](const float* q, int k, int ef, Scratch* sc) { return r->search(q, k, ef, sc); };
  s.live_set = [r](std::vector<idx_t>& a, std::vector<float>& b) { r->live_set(a, b); };
  s.maintain = [](long) {}; s.ntomb = [r]() { return r->idx->stats().n_tomb; }; s.mem = [r]() { return r->idx->memory_bytes(); };
  return s;
}
Sys make_delta(int dim, int M, int efc, int R) {
  auto ix = std::make_shared<DeltaOverlay>(base_params(dim, M, efc), R);
  Sys s; s.hold = ix; s.name = "DeltaOverlay"; auto* r = ix.get();
  s.apply = [r](int op, int e, const float* v) { if (op == 0) r->insert(e, v); else if (op == 1) r->remove(e); else r->update(e, v); };
  s.search = [r](const float* q, int k, int ef, Scratch* sc) { return r->search(q, k, ef, sc); };
  s.live_set = [r](std::vector<idx_t>& a, std::vector<float>& b) { r->live_set(a, b); };
  s.maintain = [](long) {}; s.ntomb = [r]() { return r->n_tomb(); }; s.mem = [r]() { return r->base.memory_bytes() + r->delta.memory_bytes(); };
  return s;
}

Sys make_consolidate(int dim, int M, int efc, double tc) {
  auto ix = std::make_shared<ConsolidateIndex>(base_params(dim, M, efc), tc);
  char nm[64]; snprintf(nm, sizeof(nm), "Consolidate-%.2f", tc);
  Sys s; s.hold = ix; s.name = nm; auto* r = ix.get();
  s.apply = [r](int op, int e, const float* v) { if (op == 0) r->insert(e, v); else if (op == 1) r->remove(e); else r->update(e, v); };
  s.search = [r](const float* q, int k, int ef, Scratch* sc) { return r->search(q, k, ef, sc); };
  s.live_set = [r](std::vector<idx_t>& a, std::vector<float>& b) { r->live_set(a, b); };
  s.maintain = [](long) {}; s.ntomb = [r]() { return r->n_tomb_now(); }; s.mem = [r]() { return r->ix.memory_bytes(); };
  return s;
}
static int envi(const char* k, int d) { const char* v = getenv(k); return v ? atoi(v) : d; }
static double envd(const char* k, double d) { const char* v = getenv(k); return v ? atof(v) : d; }
static void apply_env(Sys& s) {   // optional overrides: forced-repair fraction P, deficit trigger
  auto* r = std::static_pointer_cast<StreamVecIndex>(s.hold).get();
  if (getenv("SV_P")) r->p.prune_fraction = atof(getenv("SV_P"));
  if (getenv("SV_DEFTH")) r->p.deg_deficit_threshold = atof(getenv("SV_DEFTH"));
}
Sys with_compactor(Sys s, const char* nm) {
  auto* r = std::static_pointer_cast<StreamVecIndex>(s.hold).get();
  auto cnt = std::make_shared<int>(0);
  r->compactor = [cnt](StreamVecIndex& ix) { ConsolidateIndex::consolidate_index(ix, ix.p, (*cnt)++); };
  s.name = nm; return s;
}
Sys make_named(const std::string& sysname, int dim, int nops) {
  int M = envi("SV_M", 16), efc = envi("SV_EFC", 100), b = envi("SV_B", 64), iv = envi("SV_IV", 256);
  double th = envd("SV_TH", 0.15);
  if (sysname == "svc") { auto s = with_compactor(make_streamvec(dim, M, efc, b, iv, 64, th), "StreamVec-C"); apply_env(s); return s; }
  if (sysname == "svc0") { auto s = with_compactor(make_streamvec(dim, M, efc, 0, iv, 64, th), "Skip-C");
    auto* r = std::static_pointer_cast<StreamVecIndex>(s.hold).get(); r->p.prune_fraction = 2.0; r->track_dirty = false; return s; }
  if (sysname == "cons") return make_consolidate(dim, M, efc, th);
  if (sysname == "sv") return make_streamvec(dim, M, efc, b, iv, 64, th);
  int pol = envi("SV_POL", 0);
  if (sysname == "streamvec") { auto s = make_streamvec(dim, M, efc, b, iv, 64, th, pol); apply_env(s);
    const char* pn[] = {"StreamVec", "StreamVec-FIFO", "StreamVec-Random", "StreamVec-Eager"}; s.name = pn[pol]; return s; }
  if (sysname == "streamvec_notrig") return make_streamvec(dim, M, efc, b, iv, 64, 0.99, pol);
  if (sysname == "norepair") return make_streamvec(dim, M, efc, 0, 0, 64, 0.99, 0, true);
  if (sysname == "append") return make_streamvec(dim, M, efc, 0, 0, 64, 0.99, 0, true, false, true);
  if (sysname == "reuse") return make_streamvec(dim, M, efc, 0, 0, 64, 0.99, 0, false, true);
  if (sysname == "rebuild") return make_rebuild(dim, M, efc, envi("SV_R", nops / 10));
  if (sysname == "delta") return make_delta(dim, M, efc, envi("SV_R", nops / 10));
  if (sysname == "consolidate") return make_consolidate(dim, M, efc, th);
  fprintf(stderr, "unknown system %s\n", sysname.c_str()); exit(2);
}

// mixed insert/delete/update stream over base ids [0,Nbase)
struct Op { int type; int ext; int vrow; };
std::vector<Op> gen_stream(int n_seed, int n_ops, int Nbase, double ir, double dr, double ur, uint64_t seed) {
  std::mt19937 g(seed); std::uniform_real_distribution<double> u(0, 1);
  std::vector<int> live; for (int i = 0; i < n_seed; ++i) live.push_back(i);
  int nxt = n_seed; std::vector<Op> ops;
  for (int t = 0; t < n_ops; ++t) {
    double r = u(g);
    if (r < ir || (int)live.size() < 100) { ops.push_back({0, nxt, nxt % Nbase}); live.push_back(nxt); nxt++; }
    else if (r < ir + dr && (int)live.size() > 100) { int j = g() % live.size(); ops.push_back({1, live[j], -1}); live[j] = live.back(); live.pop_back(); }
    else { int j = g() % live.size(); ops.push_back({2, live[j], (int)(g() % Nbase)}); }
  }
  return ops;
}

double recall_now(Sys& s, const Data& Q, int nq, int dim, int k, int ef) {
  std::vector<idx_t> ids; std::vector<float> mat; s.live_set(ids, mat);
  std::vector<int> qi; for (int j = 0; j < nq; ++j) qi.push_back(j);
  auto gt = brute_gt(ids, mat, Q, qi, dim, k);
  std::vector<std::vector<idx_t>> pred(nq);
  for (int j = 0; j < nq; ++j) pred[j] = s.search(Q.row(j), k, ef, nullptr);
  return recall_at_k(pred, gt, k);
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "validate";
  Data B = load_fbin(getenv("SV_BASE") ? getenv("SV_BASE") : "data/bin/sift_base.fbin");
  Data Q = load_fbin(getenv("SV_QUERY") ? getenv("SV_QUERY") : "data/bin/sift_query.fbin");
  int dim = B.dim;
  fprintf(stderr, "loaded base=%d query=%d dim=%d\n", B.n, Q.n, dim);

  if (mode == "valpar") {
    int N = argc > 2 ? atoi(argv[2]) : 200000;
    int nq = argc > 3 ? atoi(argv[3]) : 1000;
    Params p; p.dim = dim; p.M = 16; p.efc = 100;
    StreamVecIndex idx(p);
    double t0 = now_ms();
    idx.bulk_load_parallel(B.x.data(), N);
    double tb = now_ms() - t0;
    std::vector<idx_t> ids; std::vector<float> mat; idx.live_set(ids, mat);
    std::vector<int> qi; for (int j = 0; j < nq; ++j) qi.push_back(j);
    auto gt = brute_gt(ids, mat, Q, qi, dim, 10);
    std::vector<std::vector<idx_t>> pred(nq);
    // parallel query throughput (read-only; per-thread scratch)
    double t1 = now_ms();
    #pragma omp parallel
    { Scratch sc;
      #pragma omp for schedule(dynamic)
      for (int j = 0; j < nq; ++j) pred[j] = idx.search(Q.row(qi[j]), 10, 64, &sc); }
    double tq = now_ms() - t1;
    printf("PARBUILD N=%d build=%.1fs (%.0f ins/s) recall@10=%.4f parQ=%.0f q/s mem=%.0fMB\n",
           N, tb / 1000, N / (tb / 1000.0), recall_at_k(pred, gt, 10), nq / (tq / 1000.0),
           idx.memory_bytes() / 1e6);
    return 0;
  }

  auto append_csv = [](const std::string& path, const std::string& header, const std::string& row) {
    bool exist = std::ifstream(path).good();
    std::ofstream f(path, std::ios::app);
    if (!exist) f << header << "\n";
    f << row << "\n";
  };
  std::string RES = getenv("SV_RES") ? getenv("SV_RES") : "results_cpp/";

  // ---- e2e2: multi-seed end-to-end with time-averaged recall during the stream ----
  //   argv: e2e2 <system> <N> <nops> <seed>
  if (mode == "e2e2") {
    std::string sysname = argc > 2 ? argv[2] : "streamvec";
    int N = argc > 3 ? atoi(argv[3]) : 60000;
    int nops = argc > 4 ? atoi(argv[4]) : 60000;
    uint64_t seed = argc > 5 ? atoll(argv[5]) : 42;
    int nq = 1000, nq_s = 300, nsamp = 12;
    Sys s = make_named(sysname, dim, nops);
    int nseed = N / 2;
    for (int i = 0; i < nseed; ++i) s.apply(0, i, B.row(i));
    auto ops = gen_stream(nseed, nops, B.n, 0.6, 0.2, 0.2, seed);
    // sample points offset so that they never coincide with a rebuild/merge boundary
    long step = nops / nsamp; std::vector<double> samp;
    double tstream = 0;
    for (long t = 0; t < (long)ops.size(); ++t) { Op& o = ops[t];
      double a = now_ms();
      s.apply(o.type, o.ext, o.vrow >= 0 ? B.row(o.vrow) : nullptr); s.maintain(t + 1);
      tstream += now_ms() - a;
      if ((t + 1 + step / 2) % step == 0) samp.push_back(recall_now(s, Q, nq_s, dim, 10, 40));
    }
    double tavg = 0; for (double r : samp) tavg += r; tavg /= std::max<size_t>(1, samp.size());
    double utput = ops.size() / (tstream / 1000.0);
    for (int ef : {10, 40, 100}) {
      std::vector<idx_t> ids; std::vector<float> mat; s.live_set(ids, mat);
      std::vector<int> qi; for (int j = 0; j < nq; ++j) qi.push_back(j);
      auto gt = brute_gt(ids, mat, Q, qi, dim, 10);
      std::vector<std::vector<idx_t>> pred(nq);
      double t1 = now_ms();
      #pragma omp parallel
      { Scratch sc;
        #pragma omp for schedule(dynamic)
        for (int j = 0; j < nq; ++j) pred[j] = s.search(Q.row(j), 10, ef, &sc); }
      double tq = now_ms() - t1;
      char row[320]; snprintf(row, sizeof(row), "%s,%llu,%d,%d,%d,%.4f,%.4f,%.0f,%.0f,%.1f,%d",
        s.name.c_str(), (unsigned long long)seed, N, nops, ef, recall_at_k(pred, gt, 10), tavg,
        nq / (tq / 1000.0), utput, s.mem() / 1e6, s.ntomb());
      append_csv(RES + "e2e_v2.csv", "system,seed,N,nops,ef,recall_end,tavg_recall_ef40,query_qps,update_tput,memory_mb,n_tomb_end", row);
    }
    printf("e2e2 %s seed=%llu update_tput=%.0f tavg=%.4f\n", s.name.c_str(), (unsigned long long)seed, utput, tavg);
    return 0;
  }

  // ---- e2e3: e2e2 + workload mix (SV_MIX=i,d,u), time-averaged query QPS, repair diagnostics ----
  if (mode == "e2e3") {
    std::string sysname = argc > 2 ? argv[2] : "streamvec";
    int N = argc > 3 ? atoi(argv[3]) : 60000;
    int nops = argc > 4 ? atoi(argv[4]) : 60000;
    uint64_t seed = argc > 5 ? atoll(argv[5]) : 42;
    double ir = 0.6, dr = 0.2, ur = 0.2;
    if (getenv("SV_MIX")) sscanf(getenv("SV_MIX"), "%lf,%lf,%lf", &ir, &dr, &ur);
    int nq_s = 300, nsamp = 12;
    Sys s = make_named(sysname, dim, nops);
    int nseed = N / 2;
    for (int i = 0; i < nseed; ++i) s.apply(0, i, B.row(i));
    auto ops = gen_stream(nseed, nops, B.n, ir, dr, ur, seed);
    long step = nops / nsamp; double tstream = 0, rsum = 0, qsum = 0, csum = 0; int ns = 0;
    for (long t = 0; t < (long)ops.size(); ++t) { Op& o = ops[t];
      double a = now_ms();
      s.apply(o.type, o.ext, o.vrow >= 0 ? B.row(o.vrow) : nullptr); s.maintain(t + 1);
      tstream += now_ms() - a;
      if ((t + 1 + step / 2) % step == 0) {
        std::vector<idx_t> ids; std::vector<float> mat; s.live_set(ids, mat);
        std::vector<int> qi; for (int j = 0; j < nq_s; ++j) qi.push_back(j);
        auto gt = brute_gt(ids, mat, Q, qi, dim, 10);
        std::vector<std::vector<idx_t>> pred(nq_s); Scratch sc;
        double q0 = now_ms(); for (int j = 0; j < nq_s; ++j) pred[j] = s.search(Q.row(j), 10, 40, &sc);
        double qt = now_ms() - q0;
        rsum += recall_at_k(pred, gt, 10); qsum += nq_s / (qt / 1000.0);
        std::string dg = s.diag(); double cov = atof(dg.substr(dg.find(',', dg.find(',') + 1) + 1).c_str()); csum += cov; ns++;
      }
    }
    char row[400]; snprintf(row, sizeof(row), "%s,%llu,%d,%d,%.2f/%.2f/%.2f,%.4f,%.0f,%.0f,%.1f,%s,%.4f",
      s.name.c_str(), (unsigned long long)seed, N, nops, ir, dr, ur, rsum / ns, qsum / ns,
      ops.size() / (tstream / 1000.0), s.mem() / 1e6, s.diag().c_str(), csum / ns);
    append_csv(RES + "e2e3.csv", "system,seed,N,nops,mix,tavg_recall,tavg_qps_1thr,update_tput,memory_mb,forced,sched,coverage_end,damaged_end,coverage_mean", row);
    printf("e2e3 %s seed=%llu mix=%.2f/%.2f/%.2f tavg=%.4f qps=%.0f tput=%.0f\n", s.name.c_str(), (unsigned long long)seed, ir, dr, ur, rsum / ns, qsum / ns, ops.size() / (tstream / 1000.0));
    return 0;
  }

  // ---- fresh2: freshness series for any system (seeded) ----
  //   argv: fresh2 <system> <R or 0> <seed>
  if (mode == "fresh2") {
    std::string kind = argc > 2 ? argv[2] : "streamvec";
    int R = argc > 3 ? atoi(argv[3]) : 4000;
    uint64_t seed = argc > 4 ? atoll(argv[4]) : 11;
    int N = 50000, nops = 80000, nq = 500, ef = 40;
    Sys s = (kind == "rebuild") ? make_rebuild(dim, 16, 100, R) : make_named(kind, dim, nops);
    int nseed = N / 2;
    for (int i = 0; i < nseed; ++i) s.apply(0, i, B.row(i));
    auto ops = gen_stream(nseed, nops, B.n, 0.5, 0.25, 0.25, seed);
    std::vector<double> series; double tstream = 0;
    for (long t = 0; t < (long)ops.size(); ++t) { Op& o = ops[t];
      double a = now_ms();
      s.apply(o.type, o.ext, o.vrow >= 0 ? B.row(o.vrow) : nullptr); s.maintain(t + 1);
      tstream += now_ms() - a;
      if ((t + 1) % 2000 == 0) series.push_back(recall_now(s, Q, nq, dim, 10, ef));
    }
    double avg = 0, mn = 1; for (double r : series) { avg += r; mn = std::min(mn, r); } avg /= series.size();
    std::string ser; for (double r : series) { char b[16]; snprintf(b, sizeof(b), "%.4f;", r); ser += b; }
    char row[2048]; snprintf(row, sizeof(row), "%s,%d,%llu,%.4f,%.4f,%.0f,%s", s.name.c_str(), R,
      (unsigned long long)seed, avg, mn, nops / (tstream / 1000.0), ser.c_str());
    append_csv(RES + "fresh_v2.csv", "system,R,seed,time_avg_recall,min_recall,update_tput,series", row);
    printf("fresh2 %s R=%d seed=%llu avg=%.4f min=%.4f\n", s.name.c_str(), R, (unsigned long long)seed, avg, mn);
    return 0;
  }

  // ---- e2e: real update throughput + recall/QPS for one system ----
  if (mode == "e2e") {
    std::string sysname = argc > 2 ? argv[2] : "streamvec";
    int N = argc > 3 ? atoi(argv[3]) : 100000;
    int nops = argc > 4 ? atoi(argv[4]) : 100000;
    int nq = 1000;
    Sys s;
    if (sysname == "streamvec") s = make_streamvec(dim, 16, 100, 64, 256, 64, 0.15);
    else if (sysname == "append") s = make_streamvec(dim, 16, 100, 0, 0, 64, 0.99, 0, true, false, true);
    else if (sysname == "reuse") s = make_streamvec(dim, 16, 100, 0, 0, 64, 0.99, 0, false, true);
    else if (sysname == "rebuild") s = make_rebuild(dim, 16, 100, nops / 10);
    else if (sysname == "delta") s = make_delta(dim, 16, 100, nops / 10);
    int nseed = N / 2;
    for (int i = 0; i < nseed; ++i) s.apply(0, i, B.row(i));
    auto ops = gen_stream(nseed, nops, B.n, 0.6, 0.2, 0.2, 42);
    double t0 = now_ms();
    for (long t = 0; t < (long)ops.size(); ++t) { Op& o = ops[t];
      s.apply(o.type, o.ext, o.vrow >= 0 ? B.row(o.vrow) : nullptr); s.maintain(t + 1); }
    double tstream = now_ms() - t0;
    double utput = ops.size() / (tstream / 1000.0);
    for (int ef : {10, 40, 100}) {
      std::vector<idx_t> ids; std::vector<float> mat; s.live_set(ids, mat);
      std::vector<int> qi; for (int j = 0; j < nq; ++j) qi.push_back(j);
      auto gt = brute_gt(ids, mat, Q, qi, dim, 10);
      std::vector<std::vector<idx_t>> pred(nq); double tq;
      { double t1 = now_ms();
        #pragma omp parallel
        { Scratch sc; std::vector<std::vector<idx_t>> lp(nq);
          #pragma omp for schedule(dynamic) nowait
          for (int j = 0; j < nq; ++j) pred[j] = s.search(Q.row(j), 10, ef, &sc); }
        tq = now_ms() - t1; }
      double rec = recall_at_k(pred, gt, 10);
      double qps = nq / (tq / 1000.0);
      char row[256]; snprintf(row, sizeof(row), "%s,%d,%d,%.4f,%.0f,%.0f,%.1f",
        s.name.c_str(), N, ef, rec, qps, utput, s.mem() / 1e6);
      append_csv(RES + "e2e.csv", "system,N,ef,recall_at_10,query_qps,update_tput,memory_mb", row);
    }
    printf("e2e %s N=%d update_tput=%.0f ops/s\n", s.name.c_str(), N, utput);
    return 0;
  }

  // ---- fresh: time-averaged recall, StreamVec (always fresh) vs periodic rebuild ----
  if (mode == "fresh") {
    std::string kind = argc > 2 ? argv[2] : "streamvec";
    int R = argc > 3 ? atoi(argv[3]) : 4000;
    int N = 50000, nops = 80000, nq = 500, ef = 40;
    Sys s = (kind == "streamvec") ? make_streamvec(dim, 16, 100, 64, 256, 64, 0.99)
                                   : make_rebuild(dim, 16, 100, R);
    int nseed = N / 2;
    for (int i = 0; i < nseed; ++i) s.apply(0, i, B.row(i));
    auto ops = gen_stream(nseed, nops, B.n, 0.5, 0.25, 0.25, 11);
    std::vector<double> series;
    for (long t = 0; t < (long)ops.size(); ++t) { Op& o = ops[t];
      s.apply(o.type, o.ext, o.vrow >= 0 ? B.row(o.vrow) : nullptr); s.maintain(t + 1);
      if ((t + 1) % 4000 == 0) series.push_back(recall_now(s, Q, nq, dim, 10, ef));
    }
    double avg = 0, mn = 1; for (double r : series) { avg += r; mn = std::min(mn, r); } avg /= series.size();
    char row[128]; snprintf(row, sizeof(row), "%s,%d,%.4f,%.4f", kind == "streamvec" ? "StreamVec" : ("Rebuild-R" + std::to_string(R)).c_str(), R, avg, mn);
    append_csv(RES + "fresh.csv", "system,R,time_avg_recall,min_recall", row);
    printf("fresh %s R=%d avg=%.4f min=%.4f\n", kind.c_str(), R, avg, mn);
    return 0;
  }

  // ---- churn: recall + query cost vs tombstone ratio ----
  if (mode == "churn") {
    std::string kind = argc > 2 ? argv[2] : "streamvec";
    int N = argc > 3 ? atoi(argv[3]) : 100000;
    int nq = 1000, ef = 40;
    Sys s;
    if (kind == "streamvec") s = make_streamvec(dim, 8, 80, 32, 1, 48, 0.99);  // interval handled below
    else if (kind == "norepair") s = make_streamvec(dim, 8, 80, 0, 0, 48, 0.99, 0, true);
    else if (kind == "consolidate") s = make_consolidate(dim, 8, 80, 0.15);
    else s = make_streamvec(dim, 8, 80, 0, 0, 48, 0.99, 0, true, false, true);  // append
    StreamVecIndex* ix = (kind == "consolidate") ? &std::static_pointer_cast<ConsolidateIndex>(s.hold)->ix
                                                 : std::static_pointer_cast<StreamVecIndex>(s.hold).get();
    ConsolidateIndex* cx = (kind == "consolidate") ? std::static_pointer_cast<ConsolidateIndex>(s.hold).get() : nullptr;
    ix->bulk_load_parallel(B.x.data(), N);
    std::mt19937 g(42); std::vector<int> order(N); for (int i = 0; i < N; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), g);
    int del = 0;
    for (double lvl : {0.0, 0.1, 0.2, 0.3, 0.4}) {
      int tgt = (int)(lvl * N);
      while (del < tgt) { if (cx) cx->remove(order[del]); else ix->remove(order[del]); del++;
        if (kind == "streamvec" && del % 256 == 0) ix->repair(32); }
      if (kind == "streamvec") ix->repair(32);
      std::vector<idx_t> ids; std::vector<float> mat; s.live_set(ids, mat);
      std::vector<int> qi; for (int j = 0; j < nq; ++j) qi.push_back(j);
      auto gt = brute_gt(ids, mat, Q, qi, dim, 10);
      std::vector<std::vector<idx_t>> pred(nq);
      double t1 = now_ms();
      #pragma omp parallel
      { Scratch sc;
        #pragma omp for schedule(dynamic)
        for (int j = 0; j < nq; ++j) pred[j] = s.search(Q.row(j), 10, ef, &sc); }
      double tq = now_ms() - t1;
      char row[160]; snprintf(row, sizeof(row), "%s,%d,%.2f,%.4f,%.0f", s.name.c_str(), N, lvl,
        recall_at_k(pred, gt, 10), nq / (tq / 1000.0));
      append_csv(RES + "churn.csv", "system,N,tombstone_ratio,recall_at_10,query_qps", row);
      if (cx) fprintf(stderr, "consolidations=%d\n", cx->consolidations);
    }
    printf("churn %s done\n", s.name.c_str());
    return 0;
  }

  // ---- scale: build/insert/query/memory vs N ----
  if (mode == "scale") {
    int N = argc > 2 ? atoi(argv[2]) : 100000;
    int nq = 1000;
    Params p = base_params(dim, 16, 100);
    auto ix = std::make_shared<StreamVecIndex>(p);
    double t0 = now_ms(); ix->bulk_load_parallel(B.x.data(), N); double tb = now_ms() - t0;
    // single-op insert latency (append a few new ids beyond N)
    double tins = 0; int ni = 200;
    for (int i = 0; i < ni; ++i) { double a = now_ms(); ix->insert(N + i, B.row((N + i) % B.n)); tins += now_ms() - a; }
    // query latency (single-thread p50) and parallel QPS
    std::vector<double> lat(nq);
    for (int j = 0; j < nq; ++j) { double a = now_ms(); ix->search(Q.row(j), 10, 64, nullptr); lat[j] = now_ms() - a; }
    std::sort(lat.begin(), lat.end());
    std::vector<std::vector<idx_t>> pr(nq);
    double t1 = now_ms();
    #pragma omp parallel
    { Scratch sc; std::vector<std::vector<idx_t>>();
      #pragma omp for schedule(dynamic)
      for (int j = 0; j < nq; ++j) pr[j] = ix->search(Q.row(j), 10, 64, &sc); }
    double tq = now_ms() - t1;
    char row[200]; snprintf(row, sizeof(row), "%d,%.0f,%.4f,%.4f,%.0f,%.0f",
      N, N / (tb / 1000.0), tins / ni, lat[nq / 2], nq / (tq / 1000.0), ix->memory_bytes() / 1e6);
    append_csv(RES + "scale.csv", "N,build_ins_per_s,insert_ms,query_p50_ms,par_qps,memory_mb", row);
    printf("scale N=%d build=%.0f ins/s insert=%.3fms p50=%.3fms parQPS=%.0f\n",
           N, N / (tb / 1000.0), tins / ni, lat[nq / 2], nq / (tq / 1000.0));
    return 0;
  }

  // ---- concur: query QPS + tail latency under concurrent updates+repair ----
  if (mode == "concur") {
    int nqt = argc > 2 ? atoi(argv[2]) : 3;     // query threads
    int N = argc > 3 ? atoi(argv[3]) : 100000;
    double secs = 3.0;
    Params p = base_params(dim, 16, 100);
    auto ix = std::make_shared<StreamVecIndex>(p);
    ix->bulk_load_parallel(B.x.data(), N);
    ix->g.concurrent = true;                     // enable locked neighbor reads/writes
    std::atomic<bool> stop{false}; std::atomic<long> qtotal{0};
    std::vector<std::vector<double>> lat(nqt);
    // single writer: delete + repair (exercises concurrent neighbor edits / reverse-edge pruning)
    bool writer_on = !(getenv("SV_NOWRITER"));
    std::atomic<long> wdel{0};
    std::thread updater([&]() {
      if (!writer_on) return;
      std::mt19937 g(1); std::vector<int> order(N); for (int i = 0; i < N; ++i) order[i] = i;
      std::shuffle(order.begin(), order.end(), g); int d = 0;
      while (!stop.load()) {
        for (int b = 0; b < 64 && d < (int)(0.3 * N); ++b) { ix->remove(order[d++]); wdel++; }
        ix->repair(32);
      }
    });
    std::vector<std::thread> qts;
    double t0 = now_ms();
    for (int th = 0; th < nqt; ++th) qts.emplace_back([&, th]() {
      Scratch sc; std::mt19937 g(100 + th); long c = 0;
      while (!stop.load()) { int j = g() % Q.n; double a = now_ms();
        ix->search(Q.row(j), 10, 64, &sc); lat[th].push_back(now_ms() - a); c++; }
      qtotal += c;
    });
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    stop = true; updater.join(); for (auto& t : qts) t.join();
    double dt = (now_ms() - t0) / 1000.0;
    std::vector<double> all; for (auto& v : lat) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    double p50 = all[all.size() / 2], p99 = all[(size_t)(all.size() * 0.99)];
    char row[200]; snprintf(row, sizeof(row), "%d,%d,%.0f,%.3f,%.3f,%d,%ld,%.2f", nqt, N,
      qtotal.load() / dt, p50, p99, writer_on ? 1 : 0, wdel.load(), dt);
    append_csv(RES + "concur2.csv", "query_threads,N,query_qps,p50_ms,p99_ms,writer,writer_deletes,secs", row);
    printf("concur threads=%d QPS=%.0f p50=%.3fms p99=%.3fms\n", nqt, qtotal.load() / dt, p50, p99);
    return 0;
  }

  if (mode == "validate") {
    int N = argc > 2 ? atoi(argv[2]) : 50000;
    int nq = argc > 3 ? atoi(argv[3]) : 500;
    Params p; p.dim = dim; p.M = 16; p.efc = 100;
    StreamVecIndex idx(p); idx.g.reserve(N + 16);
    double t0 = now_ms();
    for (int i = 0; i < N; ++i) idx.insert(i, B.row(i));
    double tb = now_ms() - t0;
    std::vector<idx_t> ids; std::vector<float> mat; idx.live_set(ids, mat);
    std::vector<int> qi; for (int j = 0; j < nq; ++j) qi.push_back(j);
    auto gt = brute_gt(ids, mat, Q, qi, dim, 10);
    std::vector<std::vector<idx_t>> pred(nq);
    double t1 = now_ms();
    for (int j = 0; j < nq; ++j) pred[j] = idx.search(Q.row(j), 10, 64);
    double tq = now_ms() - t1;
    printf("N=%d build=%.0fms (%.0f ins/s) recall@10=%.4f query=%.0f q/s\n",
           N, tb, N / (tb / 1000.0), recall_at_k(pred, gt, 10), nq / (tq / 1000.0));
  }
  return 0;
}
