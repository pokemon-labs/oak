// src/net.hpp's forward pass on the GPU (2026-11-28, for pkmn.ai's M6 mini):
// the same math as Net::act, float32, every layer on Metal (the kernels:
// src/net_metal.metal). The CPU keeps what is integer work: the tables'
// row indices (with Net::idx's checks), each token's list of terms, the
// cache rows gathered in and the new keys and values scattered back.
//
// Embeddings are sums of table rows and scaled weight columns, so each
// token is a list of (offset, coefficient) terms into one parameter
// buffer, which holds the weights and these tables folded at load:
// - the weight columns as rows (each small linear transposed);
// - moveset's gelu(move_pp(move, pp)) for every move and pp in [0, 64]
//   (a batch with a pp outside that runs on the CPU);
// - active_extra's status part per status byte;
// - the logit key of every choice byte (key is linear: key(p + e) =
//   W p + W e + b, so the request tokens' keys are one GEMM and the choice
//   embeddings' are a table).
// Results match the CPU path to float rounding, not bit for bit.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "net.hpp"
#include "net_metal_src.h"  // kMetalSource: src/net_metal.metal (CMake)

namespace ataraxos::net {

namespace {

constexpr uint32_t ACT = 0x80000000u;  // a term in the activation buffer, not the parameters
constexpr int PP_MAX = 64;             // the moveset table's PPs: [0, PP_MAX]
constexpr int TG = 128;                // threads a threadgroup (4 SIMD-groups)
constexpr int MAX_SEQ = 128;           // attention's keys (net_metal.metal: 4 a lane)

// net_metal.metal's argument structs
struct GemmArgs {
  uint32_t M, N, K, lda, has_bias, has_rows, gelu, accumulate;
};
struct NormArgs {
  uint32_t rows, d, has_src, has_dst;
  float eps;
};
struct EmbedArgs {
  uint32_t tokens, d;
};
struct SlotArgs {
  uint32_t lines, fd, hp_w, hp_b;
  uint32_t tab[10];
};
struct AttnArgs {
  uint32_t d, heads, has_mask;
  float scale;
};
struct CausalArgs {
  uint32_t rows, d, heads, R;
  float scale;
};
struct HeadArgs {
  uint32_t rows, d, T, choices, req_tokens, value_w, value_b, key_b, ke;
  float sqrt_d;
};

// a linear in the parameters: weight [out, in], bias, transposed weight [in, out]
struct Lin {
  uint32_t w = 0, b = 0, wt = 0;
  int out = 0, in = 0;
  bool bias = false;
};
struct Nrm {
  uint32_t w = 0, b = 0;
};
struct Lay {
  Nrm n1, n2;
  Lin qkv, out, lin1, lin2;
};

// a shared (CPU and GPU) buffer that only grows
struct Buf {
  id<MTLBuffer> b = nil;
  size_t cap = 0;
  template <class T>
  T* get(id<MTLDevice> dev, size_t n) {
    const size_t bytes = std::max<size_t>(n * sizeof(T), 64);
    if (bytes > cap) {
      const size_t c = std::max(bytes, cap + cap / 2);
      b = [dev newBufferWithLength:c options:MTLResourceStorageModeShared];
      if (!b) throw std::runtime_error("net (Metal): out of memory");
      cap = c;
    }
    return static_cast<T*>([b contents]);
  }
};

// one row's tokens' terms: the row owns [base, base + cap) of the term arrays
struct Emit {
  uint32_t* off;
  float* coef;
  uint32_t *ts, *tc;  // every token's first term and count
  uint32_t base, cap, n = 0, tok = 0, t0 = 0;
  void begin(uint32_t t) {
    tok = t;
    t0 = base + n;
  }
  void add(uint32_t o, float c) {
    if (n >= cap) throw std::logic_error("net (Metal): a row's terms overflow");
    off[base + n] = o;
    coef[base + n] = c;
    n++;
  }
  void row(uint32_t table, int r, int d) { add(table + uint32_t(r) * uint32_t(d), 1.f); }
  void col(const Lin& L, int k, float c) {  // c * W[:, k]
    if (c != 0.f) add(L.wt + uint32_t(k) * uint32_t(L.out), c);
  }
  void end() {
    ts[tok] = t0;
    tc[tok] = base + n - t0;
  }
};

}  // namespace

struct MetalForward {
  Net& net;
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> p_gemm = nil, p_norm = nil, p_embed = nil, p_slots = nil, p_add = nil, p_attend = nil,
                              p_causal = nil, p_heads = nil;
  id<MTLBuffer> params = nil;
  std::mutex mu;

  // the parameters' offsets (floats)
  std::vector<float> P;  // built at init, then uploaded and freed
  uint32_t tab[10] = {}, hp_w = 0, hp_b = 0;
  Lin line1, line2, header, move_scalars, mon_scalars, value_head, query, key;
  Lin active_scalars, mon_state, active_extra, side_extra, foe_threat, move_feat, switch_feat;
  Nrm line_norm, update_norm, out_norm, norm;
  std::vector<Lay> update_layers, history_layers, causal_layers;
  uint32_t line_pos = 0, summary = 0, request_kind = 0, req_type = 0, species = 0, status = 0, move = 0, pos = 0;
  uint32_t value_token = 0, h_type = 0, type_emb = 0, disabled = 0, state_type = 0, moveset_tab = 0, active_status = 0;
  uint32_t choice_key = 0;

  // scratch, per call
  Buf lidx, lhp, e, m, lines, ustart, umask, uts, utc, uoff, ucoef, xu, h, qkv, att, f, hist, posoff, valid, kc, vc, nk,
      nv, dstart, dts, dtc, doff, dcoef, xd, nrows, qrows, krows, qv, kr, tgt, chb, nn, lo, vo;

  explicit MetalForward(Net& n) : net(n) {}

  // ---- parameters ----

  uint32_t put(const float* p, size_t n) {
    const size_t o = (P.size() + 15) & ~size_t(15);  // 64-byte aligned
    if (o + n >= (size_t(1) << 31)) throw std::runtime_error("net (Metal): parameters past 2^31 floats");
    P.resize(o + n, 0.f);
    if (n) std::memcpy(P.data() + o, p, n * 4);
    return uint32_t(o);
  }
  uint32_t put(const std::vector<float>& v) { return put(v.data(), v.size()); }
  Lin lin(const Linear& L, bool transposed) {
    Lin r;
    r.out = L.out;
    r.in = L.in;
    r.w = put(L.w, size_t(L.out) * L.in);
    if (L.b) {
      r.b = put(L.b, L.out);
      r.bias = true;
    }
    if (transposed) {
      std::vector<float> t(size_t(L.in) * L.out);
      for (int j = 0; j < L.out; j++)
        for (int k = 0; k < L.in; k++) t[size_t(k) * L.out + j] = L.w[size_t(j) * L.in + k];
      r.wt = put(t);
    }
    return r;
  }
  Nrm nrm(const Norm& n) { return {put(n.w, net.c_.d), put(n.b, net.c_.d)}; }
  Lay lay(const Layer& L) { return {nrm(L.n1), nrm(L.n2), lin(L.qkv, false), lin(L.out, false), lin(L.lin1, false),
                                    lin(L.lin2, false)}; }

  // kernel `name`'s pipeline, or nil with `why`
  id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, const char* name, std::string& why) {
    if (!why.empty()) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
      why = std::string("no kernel ") + name;
      return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) {
      why = std::string("kernel ") + name + ": " + (err ? err.localizedDescription.UTF8String : "?");
      return nil;
    }
    if (p.threadExecutionWidth != 32 || p.maxTotalThreadsPerThreadgroup < NSUInteger(TG)) {
      why = std::string("kernel ") + name + ": SIMD width " + std::to_string(p.threadExecutionWidth) +
            ", max threads " + std::to_string(p.maxTotalThreadsPerThreadgroup);
      return nil;
    }
    return p;
  }

  std::string init() {
    @autoreleasepool {
      const Config& c = net.c_;
      const int d = c.d;
      if (d % 4 || c.ff % 4 || c.line_ff % 4 || (SLOTS * c.field_d) % 4) return "dimensions not multiples of 4";
      if (d / c.heads > 64) return "head size over 64";
      if (2 + REQUEST_TOKENS + c.max_lines > MAX_SEQ || 2 + REQUEST_TOKENS + STATE_TOKENS > MAX_SEQ ||
          c.window > MAX_SEQ)
        return "sequences over 128";
      dev = MTLCreateSystemDefaultDevice();
      if (!dev) return "no Metal device";
      queue = [dev newCommandQueue];
      if (!queue) return "no command queue";
      MTLCompileOptions* opt = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      opt.fastMathEnabled = NO;  // the CPU's float semantics (exp, sqrt, division)
#pragma clang diagnostic pop
      NSError* err = nil;
      id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource] options:opt
                                               error:&err];
      if (!lib) return std::string("Metal compile: ") + (err ? err.localizedDescription.UTF8String : "?");
      std::string why;
      p_gemm = pipeline(lib, "gemm", why);
      p_norm = pipeline(lib, "layer_norm", why);
      p_embed = pipeline(lib, "embed", why);
      p_slots = pipeline(lib, "line_slots", why);
      p_add = pipeline(lib, "add_rows", why);
      p_attend = pipeline(lib, "attend", why);
      p_causal = pipeline(lib, "causal_attend", why);
      p_heads = pipeline(lib, "heads", why);
      if (!why.empty()) return why;
      build();
      params = [dev newBufferWithBytes:P.data() length:P.size() * 4 options:MTLResourceStorageModeShared];
      P.clear();
      P.shrink_to_fit();
      if (!params) return "out of memory for the parameters";
      return "";
    }
  }

  void build() {
    Net& n = net;
    const Config& c = n.c_;
    const int d = c.d, fd = c.field_d;
    // 1. lines
    for (int s = 0; s < SLOTS; s++)
      if (s != 8) tab[s] = put(n.slot_tab_[s]);
    hp_w = put(n.hp_w_);
    hp_b = put(n.hp_b_);
    (void)fd;
    line1 = lin(n.line1_, false);
    line2 = lin(n.line2_, false);
    line_norm = nrm(n.line_norm_);
    line_pos = put(n.line_pos_, size_t(c.max_lines) * d);
    // 2. the update encoder
    summary = put(n.summary_token_, d);
    request_kind = put(n.request_kind_, size_t(3) * d);
    req_type = put(n.req_type_, size_t(REQUEST_TOKENS) * d);
    species = put(n.species_, size_t(SPECIES_N) * d);
    status = put(n.status_, size_t(BYTE) * d);
    move = put(n.move_, size_t(MOVES_N) * d);
    header = lin(n.header_, true);
    move_scalars = lin(n.move_scalars_, true);
    mon_scalars = lin(n.mon_scalars_, true);
    for (const Layer& L : n.update_layers_) update_layers.push_back(lay(L));
    update_norm = nrm(n.update_norm_);
    // 3. the causal history
    pos = put(n.pos_, size_t(c.max_pos) * d);
    for (const Layer& L : n.causal_layers_) causal_layers.push_back(lay(L));
    out_norm = nrm(n.out_norm_);
    // 4. the decision block
    value_token = put(n.value_token_, d);
    h_type = put(n.h_type_, d);
    for (const Layer& L : n.history_layers_) history_layers.push_back(lay(L));
    norm = nrm(n.norm_);
    value_head = lin(n.value_head_, false);
    query = lin(n.query_, false);
    key = lin(n.key_, false);
    {  // each choice byte's key: W choice[e] (key's bias added in the heads)
      std::vector<float> t(size_t(256) * d);
      gemm(n.choice_, 256, d, n.key_.w, d, nullptr, t.data());
      choice_key = put(t);
    }
    if (c.state) {
      type_emb = put(n.type_emb_, size_t(TYPES) * d);
      active_scalars = lin(n.active_scalars_, true);
      mon_state = lin(n.mon_state_, true);
      disabled = put(n.disabled_tab_);
      state_type = put(n.state_type_, size_t(STATE_TOKENS) * d);
      // moveset's term per (move, pp): net.hpp's moveset, the same float ops
      std::vector<float> t(size_t(MOVES_N) * (PP_MAX + 1) * d);
      parallel_for(MOVES_N, [&](int64_t mv) {
        const float* tm = n.move_pp_tab_.data() + size_t(mv) * d;
        for (int pp = 0; pp <= PP_MAX; pp++) {
          const float p = int16_t(pp) / 64.f;
          float* o = t.data() + (size_t(mv) * (PP_MAX + 1) + pp) * d;
          for (int j = 0; j < d; j++) o[j] = gelu(tm[j] + n.move_pp_.w[size_t(j) * (d + 1) + d] * p);
        }
      });
      moveset_tab = put(t);
    }
    if (c.features) {
      active_extra = lin(n.active_extra_, true);  // columns d, d + 1 used as rows
      side_extra = lin(n.side_extra_, true);
      foe_threat = lin(n.foe_threat_, true);
      move_feat = lin(n.move_feat_, true);
      switch_feat = lin(n.switch_feat_, true);
      // active_extra's status part per status byte: W[:, :d] status[s]
      std::vector<float> wl(size_t(d) * d), t(size_t(BYTE) * d);
      for (int j = 0; j < d; j++) std::memcpy(&wl[size_t(j) * d], n.active_extra_.w + size_t(j) * (d + 2), d * 4);
      gemm(n.status_, BYTE, d, wl.data(), d, nullptr, t.data());
      active_status = put(t);
    }
  }

  // ---- terms (net.hpp's embed_request, request_features, embed_state, active_features) ----

  // the most terms a row's tokens take
  int request_terms(bool features) const {
    return (HEADER + 2) + ACTIVE_MOVES * (3 + 3 + (features ? MOVE_FEATS + 1 : 0)) +
           SIDE * (8 + 3 + 4 + 1 + (features ? SWITCH_FEATS + 1 : 0));
  }
  int state_terms(bool features) const {
    const int active = ACTIVE_SCALARS + 1 + 4 + 4 + 1 + (features ? 6 + 3 + 5 + 1 : 0);
    return 2 * active + (features ? 4 * (MOVE_FEATS + 1) : 0) + 12 * (8 + 3 + 4 + 1);
  }

  // request tokens t0.. of q; with `s` (the state, cs3's features) request_features' terms too
  void emit_request(Emit& E, uint32_t t0, const int16_t* q, const int16_t* s) {
    Net& n = net;
    const int d = n.c_.d;
    E.begin(t0);
    for (int j = 0; j < HEADER; j++) E.col(header, j, std::clamp(float(q[j]), 0.f, 8.f));
    E.add(header.b, 1.f);
    E.row(req_type, 0, d);
    E.end();
    const int16_t* f = s ? s + BASE_STATE_FIELDS : nullptr;
    for (int i = 0; i < ACTIVE_MOVES; i++) {
      const int16_t* mv = q + HEADER + i * MOVE_FIELDS;
      const float pp = mv[1], maxpp = mv[2];
      const float sc[3] = {pp / std::max(maxpp, 1.f), maxpp / 64.f, float(mv[3])};
      E.begin(t0 + 1 + i);
      for (int k = 0; k < 3; k++) E.col(move_scalars, k, sc[k]);
      E.add(move_scalars.b, 1.f);
      E.row(move, n.idx(mv[0], MOVES_N), d);
      E.row(req_type, 1 + i, d);
      if (s) {
        const int16_t* ids = s + 1 + 13;
        const int rid = q[HEADER + i * MOVE_FIELDS];
        float v[MOVE_FEATS] = {};
        int hits = 0;
        for (int k = 0; k < 4; k++) {
          if (rid > 0 && rid == ids[k]) {
            hits++;
            for (int j = 0; j < MOVE_FEATS; j++) v[j] += f[k * MOVE_FEATS + j] * MOVE_SCALE[j];
          }
        }
        for (int j = 0; j < MOVE_FEATS; j++) v[j] /= std::max(hits, 1);
        for (int j = 0; j < MOVE_FEATS; j++) E.col(move_feat, j, v[j]);
        E.add(move_feat.b, 1.f);
      }
      E.end();
    }
    for (int i = 0; i < SIDE; i++) {
      const int16_t* p = q + HEADER + ACTIVE_MOVES * MOVE_FIELDS + i * POKEMON_FIELDS;
      const float sc[8] = {p[1] / 100.f, p[2] / std::max(float(p[3]), 1.f), float(p[5]), p[0] == 0 ? 1.f : 0.f,
                           p[6] / 512.f, p[7] / 512.f, p[8] / 512.f, p[9] / 512.f};
      E.begin(t0 + 1 + ACTIVE_MOVES + i);
      for (int k = 0; k < 8; k++) E.col(mon_scalars, k, sc[k]);
      E.add(mon_scalars.b, 1.f);
      E.row(species, n.idx(p[0], SPECIES_N), d);
      E.row(status, p[4] & 0xFF, d);
      for (int k = 10; k < 14; k++) E.row(move, n.idx(p[k], MOVES_N), d);
      E.row(req_type, 1 + ACTIVE_MOVES + i, d);
      if (s) {
        const int16_t* party = s + 1 + 2 * ACTIVE_FIELDS;
        const int rsp = q[HEADER + ACTIVE_MOVES * MOVE_FIELDS + i * POKEMON_FIELDS];
        float v[SWITCH_FEATS] = {};
        int hits = 0;
        for (int k = 0; k < 6; k++) {
          if (rsp > 0 && rsp == party[k * MON_FIELDS + 3]) {
            hits++;
            for (int j = 0; j < SWITCH_FEATS; j++)
              v[j] += f[8 * MOVE_FEATS + k * SWITCH_FEATS + j] * SWITCH_SCALE[j];
          }
        }
        for (int j = 0; j < SWITCH_FEATS; j++) v[j] /= std::max(hits, 1);
        for (int j = 0; j < SWITCH_FEATS; j++) E.col(switch_feat, j, v[j]);
        E.add(switch_feat.b, 1.f);
      }
      E.end();
    }
  }

  // false: a PP outside the table (the batch goes to the CPU)
  bool emit_moveset(Emit& E, const int16_t* ids, const int16_t* pp) {
    const int d = net.c_.d;
    for (int k = 0; k < 4; k++) {
      if (ids[k] <= 0) continue;
      const int mv = net.idx(ids[k], MOVES_N);
      if (pp[k] < 0 || pp[k] > PP_MAX) return false;
      E.add(moveset_tab + (uint32_t(mv) * (PP_MAX + 1) + uint32_t(pp[k])) * uint32_t(d), 1.f);
    }
    return true;
  }

  bool emit_state(Emit& E, uint32_t t0, const int16_t* s) {
    Net& n = net;
    const Config& c = n.c_;
    const int d = c.d;
    bool ok = true;
    const float turn = s[0] / 100.f;
    const int16_t* mons = s + 1 + 2 * ACTIVE_FIELDS;
    for (int side = 0; side < 2; side++) {
      const int16_t* a = s + 1 + side * ACTIVE_FIELDS;
      float sc[ACTIVE_SCALARS];
      int k = 0;
      for (int j = 3; j < 9; j++) sc[k++] = a[j] / 6.f;
      for (int j = 9; j < 13; j++) sc[k++] = a[j] / 512.f;
      for (int j = 22; j < 26; j++) sc[k++] = a[j] / 7.f;
      sc[k++] = a[26] / 256.f;
      sc[k++] = a[27] / 16.f;
      sc[k++] = a[28] / 512.f;
      for (int bit = 0; bit < ACTIVE_FLAGS; bit++) sc[k++] = float((int(a[29]) >> bit) & 1);
      sc[k++] = turn;
      E.begin(t0 + side);
      for (int j = 0; j < ACTIVE_SCALARS; j++) E.col(active_scalars, j, sc[j]);
      E.add(active_scalars.b, 1.f);
      E.row(species, n.idx(a[0], SPECIES_N), d);
      E.row(type_emb, std::clamp<int>(a[1], 0, TYPES - 1), d);
      E.row(type_emb, std::clamp<int>(a[2], 0, TYPES - 1), d);
      E.row(disabled, n.idx(a[21], MOVES_N), d);
      ok &= emit_moveset(E, a + 13, a + 17);
      if (c.features) {  // active_features' this side
        float row_d = 0.f, row_d1 = 0.f, sides[5] = {0, 0, 0, 0, 0};
        bool sleep = false, frozen = false;
        for (int m = 0; m < 6; m++) {
          const int16_t* p = mons + (side * 6 + m) * MON_FIELDS;
          const bool exists = p[0] > 0, seen = p[1] > 0;
          const float active = p[2];
          const int st = p[7] & 0xFF;
          const float hp = p[5], max_hp = p[6], frac = hp / std::max(max_hp, 1.f);
          if (active != 0.f) {
            E.add(active_status + uint32_t(st) * uint32_t(d), active);
            row_d += frac * active;
            row_d1 += p[8] / 7.f * active;
          }
          const bool alive = exists && (!seen || hp > 0);
          sides[0] += alive;
          sides[1] += seen ? frac : float(exists);
          sides[2] += exists && !seen;
          sleep |= (st & SLP) && !(st & REST_BIT) && hp > 0;
          frozen |= st == FRZ && hp > 0;
        }
        sides[0] /= 6;
        sides[1] /= 6;
        sides[2] /= 6;
        sides[3] = sleep;
        sides[4] = frozen;
        E.col(active_extra, d, row_d);
        E.col(active_extra, d + 1, row_d1);
        E.add(active_extra.b, 1.f);
        for (int j = 0; j < 5; j++) E.col(side_extra, j, sides[j]);
        E.add(side_extra.b, 1.f);
        if (side == 1) {  // the foe's known moves' threat
          const int16_t* f = s + BASE_STATE_FIELDS;
          const int16_t* foe_ids = s + 1 + ACTIVE_FIELDS + 13;
          for (int m = 0; m < 4; m++) {
            if (foe_ids[m] <= 0) continue;
            for (int j = 0; j < MOVE_FEATS; j++) E.col(foe_threat, j, f[(4 + m) * MOVE_FEATS + j] * MOVE_SCALE[j]);
            E.add(foe_threat.b, 1.f);
          }
        }
      }
      E.row(state_type, side, d);
      E.end();
    }
    for (int i = 0; i < 12; i++) {
      const int16_t* p = mons + i * MON_FIELDS;
      const float hp = p[5], max_hp = p[6];
      const float sc[8] = {float(p[0]),  float(p[1]),     float(p[2]), p[4] / 100.f, hp / std::max(max_hp, 1.f),
                           max_hp > 0 ? 1.f : 0.f, max_hp / 512.f, p[8] / 7.f};
      E.begin(t0 + 2 + i);
      for (int k = 0; k < 8; k++) E.col(mon_state, k, sc[k]);
      E.add(mon_state.b, 1.f);
      E.row(species, n.idx(p[3], SPECIES_N), d);
      E.row(status, p[7] & 0xFF, d);
      ok &= emit_moveset(E, p + 9, p + 13);
      E.row(state_type, 2 + i, d);
      E.end();
    }
    return ok;
  }

  // ---- dispatches ----

  static void groups(id<MTLComputeCommandEncoder> enc, size_t n) {
    [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
  }
  template <class A>
  static void args(id<MTLComputeCommandEncoder> enc, const A& a, int i) {
    [enc setBytes:&a length:sizeof(A) atIndex:i];
  }

  // y[M, W.out] (+)= f(x[rows][M, K] W^T + b)
  void gemm_gpu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, uint32_t lda, id<MTLBuffer> rows, uint32_t M,
                const Lin& W, bool bias, id<MTLBuffer> y, bool gelu, bool accumulate) {
    if (M == 0) return;
    [enc setComputePipelineState:p_gemm];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:params offset:size_t(W.w) * 4 atIndex:1];
    [enc setBuffer:params offset:size_t(bias && W.bias ? W.b : 0) * 4 atIndex:2];
    [enc setBuffer:y offset:0 atIndex:3];
    [enc setBuffer:(rows ? rows : params) offset:0 atIndex:4];
    const GemmArgs a{M, uint32_t(W.out), uint32_t(W.in), lda, uint32_t(bias && W.bias), uint32_t(rows != nil),
                     uint32_t(gelu), uint32_t(accumulate)};
    args(enc, a, 5);
    [enc dispatchThreadgroups:MTLSizeMake((W.out + 63) / 64, (M + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
  }

  void norm_gpu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, const Nrm& n,
                id<MTLBuffer> src, id<MTLBuffer> dst) {
    if (rows == 0) return;
    [enc setComputePipelineState:p_norm];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:params offset:size_t(n.w) * 4 atIndex:2];
    [enc setBuffer:params offset:size_t(n.b) * 4 atIndex:3];
    [enc setBuffer:(src ? src : params) offset:0 atIndex:4];
    [enc setBuffer:(dst ? dst : params) offset:0 atIndex:5];
    const NormArgs a{rows, uint32_t(net.c_.d), uint32_t(src != nil), uint32_t(dst != nil), LN_EPS};
    args(enc, a, 6);
    groups(enc, (rows + 3) / 4);
  }

  void embed_gpu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> act, Buf& ts, Buf& tc, Buf& off, Buf& coef,
                 id<MTLBuffer> out, uint32_t tokens) {
    if (tokens == 0) return;
    [enc setComputePipelineState:p_embed];
    [enc setBuffer:params offset:0 atIndex:0];
    [enc setBuffer:act offset:0 atIndex:1];
    [enc setBuffer:ts.b offset:0 atIndex:2];
    [enc setBuffer:tc.b offset:0 atIndex:3];
    [enc setBuffer:off.b offset:0 atIndex:4];
    [enc setBuffer:coef.b offset:0 atIndex:5];
    [enc setBuffer:out offset:0 atIndex:6];
    const EmbedArgs a{tokens, uint32_t(net.c_.d)};
    args(enc, a, 7);
    groups(enc, (tokens + 3) / 4);
  }

  // net.hpp's encode: pre-LN layers over x [N, d]'s sequences
  void encode_gpu(id<MTLComputeCommandEncoder> enc, const std::vector<Lay>& layers, id<MTLBuffer> x, uint32_t N,
                  id<MTLBuffer> start, uint32_t seqs, id<MTLBuffer> mask) {
    const Config& c = net.c_;
    const uint32_t d = c.d;
    for (const Lay& L : layers) {
      norm_gpu(enc, x, h.b, N, L.n1, nil, nil);
      gemm_gpu(enc, h.b, d, nil, N, L.qkv, true, qkv.b, false, false);
      [enc setComputePipelineState:p_attend];
      [enc setBuffer:qkv.b offset:0 atIndex:0];
      [enc setBuffer:start offset:0 atIndex:1];
      [enc setBuffer:(mask ? mask : params) offset:0 atIndex:2];
      [enc setBuffer:att.b offset:0 atIndex:3];
      const AttnArgs a{d, uint32_t(c.heads), uint32_t(mask != nil), 1.f / std::sqrt(float(d / c.heads))};
      args(enc, a, 4);
      groups(enc, size_t(seqs) * c.heads);
      gemm_gpu(enc, att.b, d, nil, N, L.out, true, x, false, true);
      norm_gpu(enc, x, h.b, N, L.n2, nil, nil);
      gemm_gpu(enc, h.b, d, nil, N, L.lin1, true, f.b, true, false);
      gemm_gpu(enc, f.b, uint32_t(c.ff), nil, N, L.lin2, true, x, false, true);
    }
  }

  // ---- one batch ----

  // "" when done (outputs and caches written), "cpu" when the batch must
  // go to the CPU (nothing written), else the GPU's error
  std::string act(const Batch& b, float* K, float* V, int64_t S, float* logits, float* value) {
    std::lock_guard<std::mutex> lock(mu);
    {  // the CPU's part (it may throw: no autorelease pool open)
      Net& n = net;
      const Config& c = n.c_;
      const int d = c.d, fd = c.field_d, D = int(b.D), R = n.R(), L = c.causal_depth;
      const bool decide = logits != nullptr;
      if (decide && c.state && !b.state) throw std::invalid_argument("net: the net reads the state");
      std::atomic<bool> to_cpu{false};

      // 1. lines: each update's last max_lines (net.hpp's summarize)
      std::vector<int64_t> first(D), keep(D), line0(D + 1, 0);
      for (int i = 0; i < D; i++) {
        const int64_t len = b.offsets[i + 1] - b.offsets[i];
        if (len < 0) throw std::out_of_range("net: offsets");
        keep[i] = std::min<int64_t>(len, c.max_lines);
        first[i] = b.offsets[i + 1] - keep[i];
        line0[i + 1] = line0[i] + keep[i];
      }
      const uint32_t NL = uint32_t(line0[D]);
      int32_t* lv = lidx.get<int32_t>(dev, size_t(NL) * SLOTS);
      float* lh = lhp.get<float>(dev, size_t(NL) * 2);
      parallel_for(D, [&](int64_t i) {
        for (int64_t l = 0; l < keep[i]; l++) {
          const int16_t* t = b.tokens + (first[i] + l) * COLUMNS;
          int32_t* v = lv + (line0[i] + l) * SLOTS;
          v[0] = n.idx(t[KIND], ARG_TYPES);
          v[1] = n.idx(t[SUB], BYTE);
          v[2] = n.idx(t[ACTOR], IDENTS);
          v[3] = n.idx(t[OTHER], IDENTS);
          v[4] = n.idx(t[MOVE], MOVES_N);
          v[5] = n.idx(t[MOVE2], MOVES_N);
          v[6] = n.idx(t[SPECIES], SPECIES_N);
          v[7] = std::clamp<int>(t[NUM], 0, BYTE - 1);
          v[8] = 0;
          v[9] = t[STATUS] & 0xFF;
          const float max_hp = t[MAX_HP];
          lh[(line0[i] + l) * 2] = t[HP] / std::max(max_hp, 1.f);
          lh[(line0[i] + l) * 2 + 1] = max_hp > 0 ? 1.f : 0.f;
        }
      });
      float* e_ = e.get<float>(dev, size_t(NL) * SLOTS * fd);
      float* m_ = m.get<float>(dev, size_t(NL) * c.line_ff);
      float* lines_ = lines.get<float>(dev, size_t(NL) * d);
      (void)e_, (void)m_, (void)lines_;

      // 2. the update encoder's tokens: summary, request kind, request, lines
      constexpr int PRE = 2 + REQUEST_TOKENS;
      uint32_t* us = ustart.get<uint32_t>(dev, size_t(D) + 1);
      us[0] = 0;
      for (int i = 0; i < D; i++) us[i + 1] = us[i] + uint32_t(PRE + keep[i]);
      const uint32_t Nu = us[D];
      const uint32_t UROW = uint32_t(2 + request_terms(false) + 2 * c.max_lines);
      uint8_t* um = umask.get<uint8_t>(dev, Nu);
      uint32_t* uts_ = uts.get<uint32_t>(dev, Nu);
      uint32_t* utc_ = utc.get<uint32_t>(dev, Nu);
      uint32_t* uoff_ = uoff.get<uint32_t>(dev, size_t(D) * UROW);
      float* ucoef_ = ucoef.get<float>(dev, size_t(D) * UROW);
      parallel_for(D, [&](int64_t i) {
        Emit E{uoff_, ucoef_, uts_, utc_, uint32_t(i) * UROW, UROW};
        const uint32_t t0 = us[i];
        const int16_t* q = b.request + i * REQUEST_FIELDS;
        E.begin(t0);
        E.add(summary, 1.f);
        E.end();
        E.begin(t0 + 1);
        E.row(request_kind, n.idx(q[0], 3), d);
        E.end();
        emit_request(E, t0 + 2, q, nullptr);
        for (int p = 0; p < PRE; p++) um[t0 + p] = 1;
        for (int64_t l = 0; l < keep[i]; l++) {
          E.begin(t0 + PRE + uint32_t(l));
          E.add(ACT | uint32_t((line0[i] + l) * d), 1.f);
          E.row(line_pos, int(l), d);
          E.end();
          um[t0 + PRE + l] = b.tokens[(first[i] + l) * COLUMNS + KIND] != 0;
        }
      });

      // 3. the causal step: positions, the rows' valid cache slots in
      uint32_t* po = posoff.get<uint32_t>(dev, D);
      uint32_t* va = valid.get<uint32_t>(dev, D);
      const size_t per = size_t(D) * R * d;  // a layer's gathered cache
      float* kc_ = kc.get<float>(dev, size_t(L) * per);
      float* vc_ = vc.get<float>(dev, size_t(L) * per);
      float* nk_ = nk.get<float>(dev, size_t(L) * D * d);
      float* nv_ = nv.get<float>(dev, size_t(L) * D * d);
      float* hist_ = hist.get<float>(dev, size_t(D) * d);
      (void)hist_;
      parallel_for(D, [&](int64_t i) {
        po[i] = pos + uint32_t(std::min<int64_t>(b.t[i], c.max_pos - 1) * d);
        const int64_t v = std::min<int64_t>(b.t[i], R);
        va[i] = uint32_t(v);
        for (int l = 0; l < L; l++) {
          const size_t src = ((size_t(l) * S + b.keys[i]) * R) * d, dst = size_t(l) * per + size_t(i) * R * d;
          std::memcpy(kc_ + dst, K + src, size_t(v) * d * 4);
          std::memcpy(vc_ + dst, V + src, size_t(v) * d * 4);
        }
      });

      // 4. the decision block's tokens: value, history, request, state
      const int T = 2 + REQUEST_TOKENS + (c.state ? STATE_TOKENS : 0);
      const uint32_t Nd = decide ? uint32_t(D * T) : 0;
      if (decide) {
        const uint32_t DROW = uint32_t(3 + request_terms(c.features) + (c.state ? state_terms(c.features) : 0));
        uint32_t* ds = dstart.get<uint32_t>(dev, size_t(D) + 1);
        for (int i = 0; i <= D; i++) ds[i] = uint32_t(i * T);
        uint32_t* dts_ = dts.get<uint32_t>(dev, Nd);
        uint32_t* dtc_ = dtc.get<uint32_t>(dev, Nd);
        uint32_t* doff_ = doff.get<uint32_t>(dev, size_t(D) * DROW);
        float* dcoef_ = dcoef.get<float>(dev, size_t(D) * DROW);
        uint32_t* nr = nrows.get<uint32_t>(dev, size_t(D) * (1 + REQUEST_TOKENS));
        uint32_t* qr = qrows.get<uint32_t>(dev, D);
        uint32_t* kr_ = krows.get<uint32_t>(dev, size_t(D) * REQUEST_TOKENS);
        int32_t* tg_ = tgt.get<int32_t>(dev, size_t(D) * CHOICES);
        uint32_t* ch_ = chb.get<uint32_t>(dev, size_t(D) * CHOICES);
        uint32_t* nn_ = nn.get<uint32_t>(dev, D);
        parallel_for(D, [&](int64_t i) {
          Emit E{doff_, dcoef_, dts_, dtc_, uint32_t(i) * DROW, DROW};
          const uint32_t t0 = uint32_t(i * T);
          E.begin(t0);
          E.add(value_token, 1.f);
          E.end();
          E.begin(t0 + 1);
          E.add(ACT | uint32_t(i * d), 1.f);
          E.add(h_type, 1.f);
          E.end();
          const int16_t* q = b.request + i * REQUEST_FIELDS;
          const int16_t* s = c.state ? b.state + i * STATE_FIELDS : nullptr;
          emit_request(E, t0 + 2, q, c.features ? s : nullptr);
          if (s && !emit_state(E, t0 + 2 + REQUEST_TOKENS, s)) to_cpu = true;
          nr[i * (1 + REQUEST_TOKENS)] = t0;
          for (int r = 0; r < REQUEST_TOKENS; r++) {
            nr[i * (1 + REQUEST_TOKENS) + 1 + r] = t0 + 2 + r;
            kr_[i * REQUEST_TOKENS + r] = t0 + 2 + r;
          }
          qr[i] = t0 + 2 + 1 + ACTIVE_MOVES;  // the request's first side slot (mons[0])
          const uint8_t* ch = b.choices + i * CHOICES;
          const int nc = b.n[i];
          nn_[i] = uint32_t(nc);
          for (int k = 0; k < CHOICES; k++) {
            tg_[i * CHOICES + k] = -1;
            ch_[i * CHOICES + k] = ch[k];
            if (k >= nc) continue;
            const int kind = ch[k] & 3, slot = ch[k] >> 2;
            const int target = kind == 2 ? ACTIVE_MOVES + slot : slot;
            if (target != 0) tg_[i * CHOICES + k] = n.idx(target, REQUEST_TOKENS);
          }
        });
        xd.get<float>(dev, size_t(Nd) * d);
        qv.get<float>(dev, size_t(D) * d);
        kr.get<float>(dev, size_t(D) * REQUEST_TOKENS * d);
        lo.get<float>(dev, size_t(D) * CHOICES);
        vo.get<float>(dev, size_t(D) * 3);
      }
      if (to_cpu) return "cpu";
      const uint32_t Nmax = std::max(Nu, std::max(Nd, uint32_t(D)));
      xu.get<float>(dev, size_t(Nu) * d);
      h.get<float>(dev, size_t(Nmax) * d);
      qkv.get<float>(dev, size_t(Nmax) * 3 * d);
      att.get<float>(dev, size_t(Nmax) * d);
      f.get<float>(dev, size_t(Nmax) * c.ff);

      // ---- the GPU: every buffer is allocated by now ----
      std::string failed;
      @autoreleasepool {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];  // serial: each dispatch sees the last
      // 1. line vectors
      if (NL) {
        [enc setComputePipelineState:p_slots];
        [enc setBuffer:params offset:0 atIndex:0];
        [enc setBuffer:lidx.b offset:0 atIndex:1];
        [enc setBuffer:lhp.b offset:0 atIndex:2];
        [enc setBuffer:e.b offset:0 atIndex:3];
        SlotArgs sa{NL, uint32_t(fd), hp_w, hp_b, {}};
        std::copy(tab, tab + SLOTS, sa.tab);
        args(enc, sa, 4);
        groups(enc, (size_t(NL) * SLOTS * fd + TG - 1) / TG);
        gemm_gpu(enc, e.b, uint32_t(SLOTS * fd), nil, NL, line1, true, m.b, true, false);
        gemm_gpu(enc, m.b, uint32_t(c.line_ff), nil, NL, line2, true, lines.b, false, false);
        norm_gpu(enc, lines.b, lines.b, NL, line_norm, nil, nil);
      }
      // 2. the update encoder, its summary token normed into hist
      embed_gpu(enc, lines.b, uts, utc, uoff, ucoef, xu.b, Nu);
      encode_gpu(enc, update_layers, xu.b, Nu, ustart.b, uint32_t(D), umask.b);
      norm_gpu(enc, xu.b, hist.b, uint32_t(D), update_norm, ustart.b, nil);
      // 3. the causal step
      {
        [enc setComputePipelineState:p_add];
        [enc setBuffer:hist.b offset:0 atIndex:0];
        [enc setBuffer:params offset:0 atIndex:1];
        [enc setBuffer:posoff.b offset:0 atIndex:2];
        const uint32_t nd[2] = {uint32_t(D), uint32_t(d)};
        [enc setBytes:nd length:sizeof(nd) atIndex:3];
        groups(enc, (size_t(D) * d + TG - 1) / TG);
      }
      for (int l = 0; l < L; l++) {
        const Lay& Ly = causal_layers[l];
        norm_gpu(enc, hist.b, h.b, uint32_t(D), Ly.n1, nil, nil);
        gemm_gpu(enc, h.b, uint32_t(d), nil, uint32_t(D), Ly.qkv, true, qkv.b, false, false);
        [enc setComputePipelineState:p_causal];
        [enc setBuffer:qkv.b offset:0 atIndex:0];
        [enc setBuffer:kc.b offset:size_t(l) * per * 4 atIndex:1];
        [enc setBuffer:vc.b offset:size_t(l) * per * 4 atIndex:2];
        [enc setBuffer:valid.b offset:0 atIndex:3];
        [enc setBuffer:att.b offset:0 atIndex:4];
        [enc setBuffer:nk.b offset:size_t(l) * D * d * 4 atIndex:5];
        [enc setBuffer:nv.b offset:size_t(l) * D * d * 4 atIndex:6];
        const CausalArgs ca{uint32_t(D), uint32_t(d), uint32_t(c.heads), uint32_t(R),
                            1.f / std::sqrt(float(d / c.heads))};
        args(enc, ca, 7);
        groups(enc, (size_t(D) * c.heads + 3) / 4);
        gemm_gpu(enc, att.b, uint32_t(d), nil, uint32_t(D), Ly.out, true, hist.b, false, true);
        norm_gpu(enc, hist.b, h.b, uint32_t(D), Ly.n2, nil, nil);
        gemm_gpu(enc, h.b, uint32_t(d), nil, uint32_t(D), Ly.lin1, true, f.b, true, false);
        gemm_gpu(enc, f.b, uint32_t(c.ff), nil, uint32_t(D), Ly.lin2, true, hist.b, false, true);
      }
      norm_gpu(enc, hist.b, hist.b, uint32_t(D), out_norm, nil, nil);
      // 4. the decision block
      if (decide) {
        embed_gpu(enc, hist.b, dts, dtc, doff, dcoef, xd.b, Nd);
        encode_gpu(enc, history_layers, xd.b, Nd, dstart.b, uint32_t(D), nil);
        norm_gpu(enc, xd.b, xd.b, uint32_t(D) * (1 + REQUEST_TOKENS), norm, nrows.b, nrows.b);
        gemm_gpu(enc, xd.b, uint32_t(d), qrows.b, uint32_t(D), query, true, qv.b, false, false);
        gemm_gpu(enc, xd.b, uint32_t(d), krows.b, uint32_t(D) * REQUEST_TOKENS, key, false, kr.b, false, false);
        [enc setComputePipelineState:p_heads];
        [enc setBuffer:params offset:0 atIndex:0];
        [enc setBuffer:xd.b offset:0 atIndex:1];
        [enc setBuffer:qv.b offset:0 atIndex:2];
        [enc setBuffer:kr.b offset:0 atIndex:3];
        [enc setBuffer:tgt.b offset:0 atIndex:4];
        [enc setBuffer:chb.b offset:0 atIndex:5];
        [enc setBuffer:nn.b offset:0 atIndex:6];
        [enc setBuffer:lo.b offset:0 atIndex:7];
        [enc setBuffer:vo.b offset:0 atIndex:8];
        const HeadArgs ha{uint32_t(D),     uint32_t(d),   uint32_t(T), uint32_t(CHOICES), uint32_t(REQUEST_TOKENS),
                          value_head.w,    value_head.b,  key.b,       choice_key,        std::sqrt(float(d))};
        args(enc, ha, 9);
        groups(enc, (size_t(D) * (CHOICES + 3) + 3) / 4);
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        NSError* err = cb.error;
        failed = std::string("command buffer: ") + (err ? err.localizedDescription.UTF8String : "failed");
      } else {
        n.gpu_seconds_ += cb.GPUEndTime - cb.GPUStartTime;
      }
      }  // @autoreleasepool
      if (!failed.empty()) return failed;

      // the new keys and values at slot t % R, the outputs
      parallel_for(D, [&](int64_t i) {
        const int64_t slot = b.t[i] % R;
        for (int l = 0; l < L; l++) {
          const size_t dst = ((size_t(l) * S + b.keys[i]) * R + slot) * d, src = (size_t(l) * D + i) * d;
          std::memcpy(K + dst, nk_ + src, size_t(d) * 4);
          std::memcpy(V + dst, nv_ + src, size_t(d) * 4);
        }
      });
      if (decide) {
        std::memcpy(logits, [lo.b contents], size_t(D) * CHOICES * 4);
        std::memcpy(value, [vo.b contents], size_t(D) * 3 * 4);
      }
      return "";
    }
  }
};

std::string Net::set_gpu(bool on) {
  if (!on) {
    gpu_.reset();
    return "";
  }
  if (gpu_) return "";
  try {
    auto g = std::make_shared<MetalForward>(*this);
    const std::string why = g->init();
    if (!why.empty()) return why;
    gpu_ = g;
    return "";
  } catch (const std::exception& e) {
    return std::string("net (Metal): ") + e.what();
  }
}

bool Net::metal_act(const Batch& b, float* K, float* V, int64_t S, float* logits, float* value) {
  const std::string why = gpu_->act(b, K, V, S, logits, value);
  if (why.empty()) return true;
  if (why != "cpu") {  // the GPU failed: off for good, this batch and the rest on the CPU
    std::fprintf(stderr, "net: the GPU failed (%s), the CPU from now on\n", why.c_str());
    gpu_.reset();
  }
  return false;
}

}  // namespace ataraxos::net
