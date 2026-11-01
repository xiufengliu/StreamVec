// Trace-equivalence check: tombstone-skipping search on G versus the unmodified
// search (skip_t=false) on G[L] (tombstoned list entries deleted, same order;
// tombstoned nodes' lists cleared => unreachable from live seeds).
#include "../../cpp/index.hpp"
#include <cstdio>
#include <string>
using namespace streamvec;

struct Trace { std::vector<std::vector<std::pair<float,idx_t>>> layers; long dc=0, nc=0, ho=0; };

// exact copy of StreamVecIndex::search's descent, parameterised by graph and skip flag
Trace descend(HNSWGraph& g, const float* q, int ef, bool skip, Scratch& sc) {
  Trace t; long d0=sc.dist_calls, n0=sc.neighbor_checks, h0=sc.heap_ops;
  std::vector<idx_t> cur{g.ep};
  for (int lc = g.maxlevel; lc > 0; --lc) {
    auto r = g.search_layer(q, cur, 1, lc, skip, &sc); t.layers.push_back(r);
    if (!r.empty()) cur = {r[0].second};
  }
  auto r = g.search_layer(q, cur, ef, 0, skip, &sc); t.layers.push_back(r);
  t.dc = sc.dist_calls-d0; t.nc = sc.neighbor_checks-n0; t.ho = sc.heap_ops-h0; return t;
}

void induced_live(const HNSWGraph& g, HNSWGraph& h) {
  h.vec=g.vec; h.n=g.n; h.cap=g.cap; h.level=g.level; h.nbr=g.nbr; h.tomb=g.tomb;
  h.ep=g.ep; h.maxlevel=g.maxlevel; h.concurrent=false;
  for (int u = 0; u < h.n; ++u) for (auto& l : h.nbr[u]) {
    if (g.tomb[u]) { l.clear(); continue; }
    std::vector<idx_t> f; for (idx_t x : l) if (!g.tomb[x]) f.push_back(x); l = f;
  }
}

// Count expansions and sum of tombstoned entries over expanded nodes, by replaying
// the skip-search (instrumented copy of search_layer's loop; used only for the bound).
struct Exp { long E=0, tomb_entries=0, maxlist=0; };
void count_exp(HNSWGraph& g, const float* q, const std::vector<idx_t>& eps, int ef, int layer, Exp& X) {
  std::vector<char> seen(g.n,0);
  std::priority_queue<std::pair<float,idx_t>> resmax;
  std::priority_queue<std::pair<float,idx_t>, std::vector<std::pair<float,idx_t>>, std::greater<>> cand;
  for (idx_t e: eps){ float d=l2sqr(q,g.vptr(e),g.dim); seen[e]=1; cand.push({d,e}); if(!g.tomb[e]) resmax.push({d,e}); }
  while(!cand.empty()){ auto [cd,cur]=cand.top(); cand.pop();
    if(!resmax.empty() && cd>resmax.top().first && (int)resmax.size()>=ef) break;
    if(g.tomb[cur] && std::find(eps.begin(),eps.end(),cur)==eps.end()) continue;
    if(layer>=(int)g.nbr[cur].size()) continue;
    auto& ns=g.nbr[cur][layer]; X.E++; X.maxlist=std::max<long>(X.maxlist,ns.size());
    for(idx_t nb:ns) if(g.tomb[nb]) X.tomb_entries++;
    for(idx_t nb:ns){ if(seen[nb]) continue; seen[nb]=1; if(g.tomb[nb]) continue;
      float d=l2sqr(q,g.vptr(nb),g.dim);
      if((int)resmax.size()<ef || d<resmax.top().first){ cand.push({d,nb}); resmax.push({d,nb}); if((int)resmax.size()>ef) resmax.pop(); } } }
}
long overhead_bound_layer(int layer, int M){ int mmax = layer==0?2*M:M; return mmax+1+mmax/4; }

int main() {
  const int N=3000, D=16, M=8, EFC=100, NQ=300; std::mt19937 rng(123);
  std::normal_distribution<float> nd(0,1);
  auto randv=[&](std::vector<float>& v){ v.resize(D); for(auto& x:v) x=nd(rng); };
  std::vector<double> fracs{0.1,0.2,0.3,0.4,0.5,0.6}; std::vector<int> efs{10,50,100};
  std::string js="{\n  \"check\": \"trace_equivalence\",\n  \"setup\": {\"N\":3000,\"dim\":16,\"M\":8,\"M0\":16,\"efc\":100,\"queries_per_cell\":300,\"ef\":[10,50,100],\"Lmax_layer0\":21,\"Lmax_upper\":11},\n  \"cells\": [\n";
  long total_q=0, mism_res=0, mism_dc=0, mism_ho=0, mism_nc_identity=0, bound_viol=0, maxlist_global=0;
  long rt_tomb_in_result=0, rt_q=0; double rt_dc_ratio_sum=0;
  bool first=true;
  for (int scen=0; scen<2; ++scen)
  for (double f: fracs) {
    Params p; p.dim=D; p.M=M; p.efc=EFC; p.seed=7; StreamVecIndex ix(p); ix.no_repair = (scen==0);
    std::vector<float> v; for(int i=0;i<N;++i){ randv(v); ix.insert(i,v.data()); }
    std::vector<int> ids(N); for(int i=0;i<N;++i) ids[i]=i; std::shuffle(ids.begin(),ids.end(),rng);
    int ndel=(int)(f*N); int nextext=N;
    if (scen==0) { for(int i=0;i<ndel;++i) ix.remove(ids[i]); }
    else { // mixed stream: deletes, updates, inserts, with budgeted repair, no rebuild
      std::uniform_real_distribution<double> u(0,1);
      for(int i=0;i<ndel;++i){ double r=u(rng);
        if(r<0.5) ix.remove(ids[i]); else if(r<0.8){ randv(v); ix.update(ids[i],v.data()); }
        else { randv(v); ix.insert(nextext++,v.data()); ix.remove(ids[i]); }
        if(i%50==0) ix.repair(8); }
    }
    HNSWGraph& G=ix.g; HNSWGraph H(D,M,EFC,1); induced_live(G,H);
    long ml=0; for(int u=0;u<G.n;++u) for(int l=0;l<(int)G.nbr[u].size();++l){ long s=G.nbr[u][l].size();
      if(s>overhead_bound_layer(l,M)) bound_viol++; ml=std::max(ml,s);} maxlist_global=std::max(maxlist_global,ml);
    long ntomb=0; for(int u=0;u<G.n;++u) ntomb+=G.tomb[u];
    // tombstoned entries in lists of nodes inserted after tombstoning began (scen 1)
    long new_nodes=0,new_with_tomb=0; for(int u=N;u<G.n;++u) if(!G.tomb[u]){ new_nodes++; bool t=false; for(auto& l:G.nbr[u]) for(idx_t x:l) if(G.tomb[x]) t=true; new_with_tomb+=t; }
    for (int ef: efs) {
      long cell_mis=0; double ovh_sum=0, bound_sum=0, dcA=0, ncA=0, ncB=0, E_sum=0; long cell_bv=0;
      for(int qi=0;qi<NQ;++qi){ randv(v); Scratch sa, sb, sr;
        Trace A=descend(G,v.data(),ef,true,sa), B=descend(H,v.data(),ef,false,sb);
        bool same = A.layers==B.layers; total_q++;
        if(!same){mism_res++;cell_mis++;} if(A.dc!=B.dc){mism_dc++;cell_mis++;} if(A.ho!=B.ho) mism_ho++;
        // per-expansion accounting on G (skip): count expansions per layer
        long E=0, te=0, bnd=0; std::vector<idx_t> cur{G.ep};
        for(int lc=G.maxlevel; lc>=0; --lc){ Exp X; count_exp(G,v.data(),cur,lc>0?1:ef,lc,X);
          E+=X.E; te+=X.tomb_entries; bnd+=X.E*overhead_bound_layer(lc,M);
          auto r=G.search_layer(v.data(),cur,lc>0?1:ef,lc,true,&sr); if(lc>0 && !r.empty()) cur={r[0].second}; }
        if(A.nc-B.nc != te) mism_nc_identity++;
        if(A.nc > bnd){ cell_bv++; bound_viol++; }
        ovh_sum+=A.nc-B.nc; bound_sum+=bnd; dcA+=A.dc; ncA+=A.nc; ncB+=B.nc; E_sum+=E;
        // route-through contrast: skip=false on G itself
        Scratch s2; Trace R=descend(G,v.data(),ef,false,s2); rt_q++; rt_dc_ratio_sum += (double)R.dc/A.dc;
        for(auto& pr: R.layers.back()) if(G.tomb[pr.second]) rt_tomb_in_result++;
      }
      char buf[900]; snprintf(buf,sizeof buf,
        "%s    {\"scenario\":\"%s\",\"tomb_frac_target\":%.1f,\"n_slots\":%d,\"n_tomb\":%ld,\"ef\":%d,\"queries\":%d,\"mismatches\":%ld,\"mean_dist_calls\":%.2f,\"mean_expansions\":%.2f,\"mean_neighbor_checks_G\":%.2f,\"mean_neighbor_checks_GL\":%.2f,\"mean_overhead\":%.2f,\"mean_bound_Lmax_times_E\":%.2f,\"overhead_over_bound\":%.4f,\"queries_violating_bound\":%ld,\"max_list_len\":%ld,\"new_live_nodes\":%ld,\"new_live_nodes_with_tomb_entry\":%ld}",
        first?"":",\n", scen==0?"delete_only_no_repair":"mixed_ins_del_upd_with_repair", f, G.n, ntomb, ef, NQ, cell_mis, dcA/NQ, E_sum/NQ, ncA/NQ, ncB/NQ, ovh_sum/NQ, bound_sum/NQ, ovh_sum/bound_sum, cell_bv, ml, new_nodes, new_with_tomb);
      js+=buf; first=false;
      fprintf(stderr,"scen %d f %.1f ef %d mis %ld ovh %.1f bound %.1f\n",scen,f,ef,cell_mis,ovh_sum/NQ,bound_sum/NQ);
    }
  }
  bool pass = mism_res==0 && mism_dc==0 && mism_ho==0 && mism_nc_identity==0 && bound_viol==0;
  char buf[900]; snprintf(buf,sizeof buf,"\n  ],\n  \"totals\": {\"queries\":%ld,\"result_list_mismatches\":%ld,\"dist_calls_mismatches\":%ld,\"heap_ops_mismatches\":%ld,\"overhead_identity_mismatches\":%ld,\"list_or_query_bound_violations\":%ld,\"max_list_len_any_layer\":%ld,\"route_through_mean_distcalls_ratio_vs_skip\":%.4f,\"route_through_tombstones_in_layer0_results\":%ld},\n  \"verdict\": \"%s\"\n}\n",
    total_q,mism_res,mism_dc,mism_ho,mism_nc_identity,bound_viol,maxlist_global,rt_dc_ratio_sum/rt_q,rt_tomb_in_result,pass?"pass":"fail");
  js+=buf; printf("%s",js.c_str()); return pass?0:1;
}
