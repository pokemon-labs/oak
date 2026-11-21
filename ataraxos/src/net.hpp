// The causal net's forward pass without torch (2026-11-21, for pkmn.ai's
// arena: the bundle can't carry torch). Mirrors ataraxos/nets/causal_net.py
// (CausalPolicy) and policy.py at inference, float32, as CausalActor runs it
// on the CPU: an update's summary (LineEncoder, UpdateEncoder), one position
// of the causal history against each layer's cached keys and values, and
// the decision block's heads (logits and value; the Q head isn't run: the
// actor and the search don't ask for it).
//
// Weights: ataraxos/arena/export.py's file (its docstring), memory-mapped.
// Lookups that are linear in an embedding are folded into tables at load
// (a line slot's projection of each table row, move_pp's and
// disabled_move's of each move), so results match torch to float rounding,
// not bit for bit.
//
// Rows are batched: every linear layer is one GEMM over all the batch's
// tokens (Accelerate's on Apple, so the AMX units run it; Eigen's
// elsewhere), attention and the per-row embedding work are split over
// threads (GCD on Apple, OpenMP elsewhere when built with it).

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#else
#include <Eigen/Core>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ataraxos::net {

// src/tokens.hpp's and src/state_tokens.hpp's layouts (policy.py's and
// causal_net.py's constants)
constexpr int COLUMNS = 11, SLOTS = 10;
enum Column { KIND, SUB, ACTOR, OTHER, MOVE, MOVE2, SPECIES, NUM, HP, MAX_HP, STATUS };
constexpr int ARG_TYPES = 43, SPECIES_N = 152, MOVES_N = 166, IDENTS = 17, BYTE = 256, TYPES = 16;
constexpr int HEADER = 7, ACTIVE_MOVES = 4, MOVE_FIELDS = 4, SIDE = 6, POKEMON_FIELDS = 14;
constexpr int REQUEST_FIELDS = HEADER + ACTIVE_MOVES * MOVE_FIELDS + SIDE * POKEMON_FIELDS;
constexpr int REQUEST_TOKENS = 1 + ACTIVE_MOVES + SIDE;
constexpr int CHOICES = 9;
constexpr int ACTIVE_FLAGS = 16;
constexpr int ACTIVE_FIELDS = 1 + 2 + 6 + 4 + 8 + 1 + 4 + 3 + 1;
constexpr int MON_FIELDS = 3 + 5 + 1 + 8;
constexpr int BASE_STATE_FIELDS = 1 + 2 * ACTIVE_FIELDS + 12 * MON_FIELDS;
constexpr int MOVE_FEATS = 9, SWITCH_FEATS = 5;
constexpr int FEATURE_FIELDS = 8 * MOVE_FEATS + 6 * SWITCH_FEATS;
constexpr int STATE_FIELDS = BASE_STATE_FIELDS + FEATURE_FIELDS;
constexpr int STATE_TOKENS = 2 + 12;
constexpr int ACTIVE_SCALARS = 6 + 4 + 4 + 3 + ACTIVE_FLAGS + 1;
constexpr float MOVE_SCALE[MOVE_FEATS] = {1 / 1000.f, 1 / 1000.f, 1 / 1000.f, 1 / 256.f, 1 / 256.f,
                                          1 / 256.f,  1 / 256.f,  1 / 4.f,    1.f};
constexpr float SWITCH_SCALE[SWITCH_FEATS] = {1 / 1000.f, 1 / 256.f, 1 / 1000.f, 1 / 256.f, 1.f};
constexpr int SLP = 0b111, FRZ = 0b00100000, REST_BIT = 0b10000000;
constexpr float LN_EPS = 1e-5f;

struct Config {
  int d = 0, heads = 0, window = 0, ff = 0, field_d = 0, line_ff = 0, max_lines = 0, max_pos = 0;
  int depth = 0, update_depth = 0, causal_depth = 0;
  bool state = false, features = false, q_head = false;
};

// ---- threads and math -------------------------------------------------------

inline int& threads_setting() {
  static int n = 1;
  return n;
}

// f(i) for i in [0, n), split over the threads (each takes a contiguous range)
template <class F>
void parallel_for(int64_t n, const F& f) {
  const int T = int(std::max<int64_t>(1, std::min<int64_t>(threads_setting(), n)));
  if (T <= 1) {
    for (int64_t i = 0; i < n; i++) f(i);
    return;
  }
#if defined(__APPLE__)
  const F* fp = &f;
  dispatch_apply(size_t(T), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
    const int64_t a = n * int64_t(w) / T, b = n * int64_t(w + 1) / T;
    for (int64_t i = a; i < b; i++) (*fp)(i);
  });
#elif defined(_OPENMP)
#pragma omp parallel for num_threads(T) schedule(static)
  for (int64_t i = 0; i < n; i++) f(i);
#else
  for (int64_t i = 0; i < n; i++) f(i);
#endif
}

// y[M, N] = x[M, K] W[N, K]^T (+ b[N])
inline void gemm(const float* x, int64_t M, int64_t K, const float* W, int64_t N, const float* b, float* y) {
  if (M == 0) return;
#ifdef __APPLE__
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(M), int(N), int(K), 1.f, x, int(K), W, int(K), 0.f, y,
              int(N));
#else
  using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const Mat> X(x, M, K), Wm(W, N, K);
  Eigen::Map<Mat> Y(y, M, N);
  Y.noalias() = X * Wm.transpose();
#endif
  if (b) {
    parallel_for(M, [&](int64_t i) {
      float* r = y + i * N;
      for (int64_t j = 0; j < N; j++) r[j] += b[j];
    });
  }
}

// y[N] (+)= W[N, K] x[K] (+ b): one row, no threads
inline void gemv(const float* W, int N, int K, const float* x, const float* b, float* y, bool accumulate = false) {
  for (int j = 0; j < N; j++) {
    const float* w = W + int64_t(j) * K;
    float s = b ? b[j] : 0.f;
    for (int k = 0; k < K; k++) s += w[k] * x[k];
    y[j] = accumulate ? y[j] + s : s;
  }
}

inline float gelu(float x) { return 0.5f * x * (1.f + std::erf(x * float(M_SQRT1_2))); }

inline void layer_norm_row(const float* x, int d, const float* w, const float* b, float* y) {
  double mean = 0, var = 0;
  for (int j = 0; j < d; j++) mean += x[j];
  mean /= d;
  for (int j = 0; j < d; j++) var += (x[j] - mean) * (x[j] - mean);
  var /= d;
  const float inv = float(1.0 / std::sqrt(var + LN_EPS));
  for (int j = 0; j < d; j++) y[j] = float(x[j] - mean) * inv * w[j] + b[j];
}

// ---- weights ----------------------------------------------------------------

struct Tensor {
  const float* p = nullptr;
  std::vector<int64_t> shape;
  int64_t size() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

struct Linear {
  const float* w = nullptr;  // [out, in]
  const float* b = nullptr;  // [out] or null
  int out = 0, in = 0;
};

struct Norm {
  const float* w = nullptr;
  const float* b = nullptr;
};

// nn.TransformerEncoderLayer (norm_first, GELU) or CausalLayer: the same
// pieces (in_proj = qkv, out_proj = out, lin1/lin2 = ff.0/ff.2)
struct Layer {
  Norm n1, n2;
  Linear qkv, out, lin1, lin2;
};

class Weights {
 public:
  explicit Weights(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("net: can't open " + path);
    struct stat st{};
    ::fstat(fd, &st);
    size_ = size_t(st.st_size);
    base_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base_ == MAP_FAILED) throw std::runtime_error("net: can't map " + path);
    const char* text = static_cast<const char*>(base_);
    const size_t head = std::min<size_t>(size_, HEADER_BYTES);
    std::string header(text, strnlen(text, head));
    std::istringstream in(header);
    std::string line;
    if (!std::getline(in, line) || line != "ATXNET1") throw std::runtime_error("net: " + path + " isn't a net file");
    bool ended = false;
    while (std::getline(in, line)) {
      std::istringstream ls(line);
      std::string kind, name;
      ls >> kind;
      if (kind == "end") {
        ended = true;
        break;
      }
      ls >> name;
      if (kind == "cfg") {
        std::string v;
        ls >> v;
        cfg_[name] = v;
      } else if (kind == "t") {
        int64_t off = 0, s = 0;
        ls >> off;
        Tensor t;
        while (ls >> s) t.shape.push_back(s);
        if (HEADER_BYTES + size_t(off) + size_t(t.size()) * 4 > size_) throw std::runtime_error("net: truncated " + name);
        t.p = reinterpret_cast<const float*>(text + HEADER_BYTES + off);
        tensors_[name] = t;
      }
    }
    if (!ended) throw std::runtime_error("net: header without end");
  }
  ~Weights() {
    if (base_ && base_ != MAP_FAILED) ::munmap(base_, size_);
  }
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;

  int cfg(const std::string& k) const {
    auto it = cfg_.find(k);
    if (it == cfg_.end()) throw std::runtime_error("net: no config " + k);
    return std::stoi(it->second);
  }
  bool has(const std::string& name) const { return tensors_.count(name) > 0; }
  const float* get(const std::string& name, std::vector<int64_t> shape) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("net: no tensor " + name);
    if (it->second.shape != shape) {
      std::string s;
      for (auto x : it->second.shape) s += " " + std::to_string(x);
      throw std::runtime_error("net: " + name + " has shape" + s);
    }
    return it->second.p;
  }
  Linear linear(const std::string& name, int out, int in, bool bias = true) const {
    return {get(name + ".weight", {out, in}), bias ? get(name + ".bias", {out}) : nullptr, out, in};
  }
  Norm norm(const std::string& name, int d) const { return {get(name + ".weight", {d}), get(name + ".bias", {d})}; }

  static constexpr size_t HEADER_BYTES = 1 << 16;  // export.py's HEADER

 private:
  void* base_ = nullptr;
  size_t size_ = 0;
  std::map<std::string, std::string> cfg_;
  std::map<std::string, Tensor> tensors_;
};

// ---- the net ----------------------------------------------------------------

// One batch: D decisions (rows), their update's lines (tokens [T, COLUMNS] by
// offsets [D + 1]), request [D, REQUEST_FIELDS], state [D, STATE_FIELDS]
// (null without `state`), choices [D, CHOICES], n [D]; each row's cache key
// and position t (its decision index in the game).
struct Batch {
  int64_t D = 0;
  const int16_t* tokens = nullptr;
  const int64_t* offsets = nullptr;
  const int16_t* request = nullptr;
  const int16_t* state = nullptr;
  const uint8_t* choices = nullptr;
  const uint8_t* n = nullptr;
  const int64_t* keys = nullptr;
  const int64_t* t = nullptr;
};

class Net {
 public:
  explicit Net(const std::string& path) : w_(path) {
    Config& c = c_;
    c.d = w_.cfg("d");
    c.heads = w_.cfg("heads");
    c.window = w_.cfg("window");
    c.ff = w_.cfg("ff");
    c.field_d = w_.cfg("field_d");
    c.line_ff = w_.cfg("line_ff");
    c.max_lines = w_.cfg("max_lines");
    c.max_pos = w_.cfg("max_pos");
    c.depth = w_.cfg("depth");
    c.update_depth = w_.cfg("update_depth");
    c.causal_depth = w_.cfg("causal_depth");
    c.state = w_.cfg("state");
    c.features = w_.cfg("features");
    c.q_head = w_.cfg("q_head");
    if (c.d % c.heads) throw std::runtime_error("net: d % heads");
    if (c.features && !c.state) throw std::runtime_error("net: features without state");
    load();
  }

  const Config& config() const { return c_; }
  int R() const { return c_.window - 1; }

  // Runs the batch: summaries, one causal position each (reading K, V [L,
  // S, R, d] at the rows' keys, then writing the new keys and values at
  // slot t % R), and with `logits` the decision block: logits [D, CHOICES]
  // (-inf past n) and value logits [D, 3]. Without, a prefill.
  void act(const Batch& b, float* K, float* V, int64_t S, float* logits, float* value) {
    const int d = c_.d, D = int(b.D);
    if (D == 0) return;
    for (int64_t i = 0; i < D; i++) {
      if (b.keys[i] < 0 || b.keys[i] >= S) throw std::out_of_range("net: cache key");
      if (b.t[i] < 0) throw std::out_of_range("net: t");
    }
    bad_ = false;
    std::vector<float> h(size_t(D) * d);
    summarize(b, h.data());
    causal(b, h.data(), K, V, S);
    if (logits) decide(b, h.data(), logits, value);
    if (bad_) throw std::out_of_range("net: an index past its table (tokens, request or state)");
  }

 private:
  Weights w_;
  Config c_;
  std::atomic<bool> bad_{false};

  // tables
  const float *kind_, *sub_, *ident_, *other_, *move_, *species_, *num_, *status_;
  Linear hp_, header_, move_scalars_, mon_scalars_, value_head_, query_, key_;
  const float *req_type_, *value_token_, *line_pos_, *summary_token_, *request_kind_, *choice_, *pos_, *h_type_;
  Norm norm_, update_norm_, out_norm_, line_norm_;
  Linear line1_, line2_;
  std::vector<Layer> update_layers_, history_layers_, causal_layers_;
  // state
  const float *type_emb_ = nullptr, *state_type_ = nullptr;
  Linear active_scalars_, mon_state_, move_pp_;
  // features
  Linear active_extra_, side_extra_, foe_threat_, move_feat_, switch_feat_;
  // folded tables: each line slot's projection per table row [rows, field_d],
  // hp's slot as a [field_d, 2] linear; move_pp's move part per move [MOVES_N,
  // d] (bias in); disabled_move per move [MOVES_N, d]
  std::vector<float> slot_tab_[SLOTS];
  std::vector<float> hp_w_, hp_b_;
  std::vector<float> move_pp_tab_, disabled_tab_;

  Layer layer(const std::string& p, bool causal) const {
    const int d = c_.d, ff = c_.ff;
    Layer L;
    if (causal) {
      L.n1 = w_.norm(p + ".ln1", d);
      L.n2 = w_.norm(p + ".ln2", d);
      L.qkv = w_.linear(p + ".qkv", 3 * d, d);
      L.out = w_.linear(p + ".out", d, d);
      L.lin1 = w_.linear(p + ".ff.0", ff, d);
      L.lin2 = w_.linear(p + ".ff.2", d, ff);
    } else {
      L.n1 = w_.norm(p + ".norm1", d);
      L.n2 = w_.norm(p + ".norm2", d);
      L.qkv = {w_.get(p + ".self_attn.in_proj_weight", {3 * d, d}), w_.get(p + ".self_attn.in_proj_bias", {3 * d}),
               3 * d, d};
      L.out = w_.linear(p + ".self_attn.out_proj", d, d);
      L.lin1 = w_.linear(p + ".linear1", ff, d);
      L.lin2 = w_.linear(p + ".linear2", d, ff);
    }
    return L;
  }

  void load() {
    const int d = c_.d, fd = c_.field_d;
    kind_ = w_.get("kind.weight", {ARG_TYPES, d});
    sub_ = w_.get("sub.weight", {BYTE, d});
    ident_ = w_.get("ident.weight", {IDENTS, d});
    other_ = w_.get("other.weight", {IDENTS, d});
    move_ = w_.get("move.weight", {MOVES_N, d});
    species_ = w_.get("species.weight", {SPECIES_N, d});
    num_ = w_.get("num.weight", {BYTE, d});
    status_ = w_.get("status.weight", {BYTE, d});
    hp_ = w_.linear("hp", d, 2);
    req_type_ = w_.get("req_type", {REQUEST_TOKENS, d});
    header_ = w_.linear("header", d, HEADER);
    move_scalars_ = w_.linear("move_scalars", d, 3);
    mon_scalars_ = w_.linear("mon_scalars", d, 8);
    value_token_ = w_.get("value_token", {d});
    norm_ = w_.norm("norm", d);
    value_head_ = w_.linear("value_head", 3, d);
    query_ = w_.linear("query", d, d);
    key_ = w_.linear("key", d, d);
    line1_ = w_.linear("line_encoder.mlp.0", c_.line_ff, SLOTS * fd);
    line2_ = w_.linear("line_encoder.mlp.2", d, c_.line_ff);
    line_norm_ = w_.norm("line_encoder.mlp.3", d);
    line_pos_ = w_.get("line_pos", {c_.max_lines, d});
    summary_token_ = w_.get("update_encoder.summary_token", {d});
    update_norm_ = w_.norm("update_encoder.norm", d);
    request_kind_ = w_.get("request_kind.weight", {3, d});
    choice_ = w_.get("choice.weight", {256, d});
    pos_ = w_.get("pos", {c_.max_pos, d});
    out_norm_ = w_.norm("out_norm", d);
    h_type_ = w_.get("h_type", {d});
    for (int i = 0; i < c_.update_depth; i++)
      update_layers_.push_back(layer("update_encoder.encoder.layers." + std::to_string(i), false));
    for (int i = 0; i < c_.depth; i++)
      history_layers_.push_back(layer("history_encoder.layers." + std::to_string(i), false));
    for (int i = 0; i < c_.causal_depth; i++) causal_layers_.push_back(layer("layers." + std::to_string(i), true));
    if (c_.state) {
      type_emb_ = w_.get("type_emb.weight", {TYPES, d});
      active_scalars_ = w_.linear("active_scalars", d, ACTIVE_SCALARS);
      Linear disabled = w_.linear("disabled_move", d, d, false);
      move_pp_ = w_.linear("move_pp.0", d, d + 1);
      mon_state_ = w_.linear("mon_state", d, 8);
      state_type_ = w_.get("state_type", {STATE_TOKENS, d});
      // move_pp's move part: W[:, :d] move[id] + b; the PP column stays
      std::vector<float> wm(size_t(d) * d);
      for (int j = 0; j < d; j++) std::memcpy(&wm[size_t(j) * d], move_pp_.w + size_t(j) * (d + 1), d * 4);
      move_pp_tab_.resize(size_t(MOVES_N) * d);
      gemm(move_, MOVES_N, d, wm.data(), d, move_pp_.b, move_pp_tab_.data());
      disabled_tab_.resize(size_t(MOVES_N) * d);
      gemm(move_, MOVES_N, d, disabled.w, d, nullptr, disabled_tab_.data());
    }
    if (c_.features) {
      active_extra_ = w_.linear("active_extra", d, d + 2);
      side_extra_ = w_.linear("side_extra", d, 5);
      foe_threat_ = w_.linear("foe_threat", d, MOVE_FEATS);
      move_feat_ = w_.linear("move_feat", d, MOVE_FEATS);
      switch_feat_ = w_.linear("switch_feat", d, SWITCH_FEATS);
    }
    // the line slots' tables
    const float* tabs[SLOTS] = {kind_, sub_, ident_, other_, move_, move_, species_, num_, nullptr, status_};
    const int rows[SLOTS] = {ARG_TYPES, BYTE, IDENTS, IDENTS, MOVES_N, MOVES_N, SPECIES_N, BYTE, 0, BYTE};
    for (int s = 0; s < SLOTS; s++) {
      const float* W = w_.get("line_encoder.slot." + std::to_string(s) + ".weight", {fd, d});
      if (s == 8) {  // hp: slot(hp(x)) = (W hp.W) x + W hp.b
        hp_w_.assign(size_t(fd) * 2, 0.f);
        hp_b_.assign(fd, 0.f);
        for (int f = 0; f < fd; f++) {
          for (int j = 0; j < d; j++) {
            hp_w_[f * 2] += W[size_t(f) * d + j] * hp_.w[j * 2];
            hp_w_[f * 2 + 1] += W[size_t(f) * d + j] * hp_.w[j * 2 + 1];
            hp_b_[f] += W[size_t(f) * d + j] * hp_.b[j];
          }
        }
        continue;
      }
      slot_tab_[s].resize(size_t(rows[s]) * fd);
      gemm(tabs[s], rows[s], d, W, fd, nullptr, slot_tab_[s].data());
    }
  }

  // a table row, or row 0 and the batch marked bad when out of range
  int idx(int v, int rows) {
    if (v < 0 || v >= rows) {
      bad_ = true;
      return 0;
    }
    return v;
  }

  // ---- shared blocks ----

  // multi-head attention of queries q (stride qs) over keys k, values v
  // (stride kvs; Tk of them, `allowed` null or a 0/1 mask) -> out (stride os)
  void attend(const float* q, int64_t qs, int Tq, const float* k, const float* v, int64_t kvs, int Tk,
              const uint8_t* allowed, float* out, int64_t os, std::vector<float>& sc) const {
    const int H = c_.heads, hd = c_.d / H;
    const float scale = 1.f / std::sqrt(float(hd));
    sc.resize(Tk);
    for (int i = 0; i < Tq; i++) {
      for (int h = 0; h < H; h++) {
        const float* qi = q + i * qs + h * hd;
        float mx = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < Tk; j++) {
          if (allowed && !allowed[j]) {
            sc[j] = -std::numeric_limits<float>::infinity();
            continue;
          }
          const float* kj = k + j * kvs + h * hd;
          float s = 0;
          for (int e = 0; e < hd; e++) s += qi[e] * kj[e];
          sc[j] = s * scale;
          mx = std::max(mx, sc[j]);
        }
        float z = 0;
        for (int j = 0; j < Tk; j++) {
          sc[j] = (allowed && !allowed[j]) ? 0.f : std::exp(sc[j] - mx);
          z += sc[j];
        }
        float* o = out + i * os + h * hd;
        std::fill(o, o + hd, 0.f);
        for (int j = 0; j < Tk; j++) {
          if (sc[j] == 0.f) continue;
          const float p = sc[j] / z;
          const float* vj = v + j * kvs + h * hd;
          for (int e = 0; e < hd; e++) o[e] += p * vj[e];
        }
      }
    }
  }

  void norm_rows(const float* x, int64_t N, const Norm& n, float* y) const {
    const int d = c_.d;
    parallel_for(N, [&](int64_t i) { layer_norm_row(x + i * d, d, n.w, n.b, y + i * d); });
  }

  // x += lin2(gelu(lin1(norm2(x)))) over N rows
  void feed_forward(const Layer& L, float* x, int64_t N, std::vector<float>& h, std::vector<float>& f) const {
    const int d = c_.d, ff = c_.ff;
    h.resize(size_t(N) * d);
    f.resize(size_t(N) * ff);
    norm_rows(x, N, L.n2, h.data());
    gemm(h.data(), N, d, L.lin1.w, ff, L.lin1.b, f.data());
    parallel_for(N, [&](int64_t i) {
      float* r = f.data() + i * ff;
      for (int j = 0; j < ff; j++) r[j] = gelu(r[j]);
    });
    gemm(f.data(), N, ff, L.lin2.w, d, L.lin2.b, h.data());
    parallel_for(N, [&](int64_t i) {
      for (int j = 0; j < d; j++) x[i * d + j] += h[i * d + j];
    });
  }

  // pre-LN encoder layers over sequences [start[s], start[s + 1]) of x
  // [N, d], each attending within itself; `allowed` [N] null or the key mask
  void encode(const std::vector<Layer>& layers, float* x, const std::vector<int64_t>& start,
              const uint8_t* allowed) const {
    const int d = c_.d;
    const int64_t N = start.back(), B = int64_t(start.size()) - 1;
    std::vector<float> h(size_t(N) * d), qkv(size_t(N) * 3 * d), a(size_t(N) * d), f;
    for (const Layer& L : layers) {
      norm_rows(x, N, L.n1, h.data());
      gemm(h.data(), N, d, L.qkv.w, 3 * d, L.qkv.b, qkv.data());
      parallel_for(B, [&](int64_t s) {
        thread_local std::vector<float> sc;
        const int64_t s0 = start[s], T = start[s + 1] - s0;
        const float* base = qkv.data() + s0 * 3 * d;
        attend(base, 3 * d, int(T), base + d, base + 2 * d, 3 * d, int(T), allowed ? allowed + s0 : nullptr,
               a.data() + s0 * d, d, sc);
      });
      gemm(a.data(), N, d, L.out.w, d, L.out.b, h.data());
      parallel_for(N, [&](int64_t i) {
        for (int j = 0; j < d; j++) x[i * d + j] += h[i * d + j];
      });
      feed_forward(L, x, N, h, f);
    }
  }

  // embed_request: q [REQUEST_FIELDS] -> out [REQUEST_TOKENS, d], req_type in
  void embed_request(const int16_t* q, float* out) {
    const int d = c_.d;
    float hd[HEADER];
    for (int j = 0; j < HEADER; j++) hd[j] = std::clamp(float(q[j]), 0.f, 8.f);
    gemv(header_.w, d, HEADER, hd, header_.b, out);
    for (int i = 0; i < ACTIVE_MOVES; i++) {
      const int16_t* m = q + HEADER + i * MOVE_FIELDS;
      const float pp = m[1], maxpp = m[2];
      const float sc[3] = {pp / std::max(maxpp, 1.f), maxpp / 64.f, float(m[3])};
      float* o = out + (1 + i) * d;
      gemv(move_scalars_.w, d, 3, sc, move_scalars_.b, o);
      const float* e = move_ + size_t(idx(m[0], MOVES_N)) * d;
      for (int j = 0; j < d; j++) o[j] += e[j];
    }
    for (int i = 0; i < SIDE; i++) {
      const int16_t* s = q + HEADER + ACTIVE_MOVES * MOVE_FIELDS + i * POKEMON_FIELDS;
      const float sc[8] = {s[1] / 100.f, s[2] / std::max(float(s[3]), 1.f), float(s[5]), s[0] == 0 ? 1.f : 0.f,
                           s[6] / 512.f, s[7] / 512.f, s[8] / 512.f, s[9] / 512.f};
      float* o = out + (1 + ACTIVE_MOVES + i) * d;
      gemv(mon_scalars_.w, d, 8, sc, mon_scalars_.b, o);
      const float* sp = species_ + size_t(idx(s[0], SPECIES_N)) * d;
      const float* st = status_ + size_t(s[4] & 0xFF) * d;
      for (int j = 0; j < d; j++) o[j] += sp[j] + st[j];
      for (int k = 10; k < 14; k++) {
        const float* e = move_ + size_t(idx(s[k], MOVES_N)) * d;
        for (int j = 0; j < d; j++) o[j] += e[j];
      }
    }
    for (int64_t j = 0; j < int64_t(REQUEST_TOKENS) * d; j++) out[j] += req_type_[j];
  }

  // ---- 1. summaries ----

  void summarize(const Batch& b, float* out) {
    const int d = c_.d, fd = c_.field_d, D = int(b.D);
    // kept lines: each update's last max_lines (pad_chunks)
    std::vector<int64_t> first(D), keep(D), line0(D + 1, 0);
    for (int i = 0; i < D; i++) {
      const int64_t len = b.offsets[i + 1] - b.offsets[i];
      if (len < 0) throw std::out_of_range("net: offsets");
      keep[i] = std::min<int64_t>(len, c_.max_lines);
      first[i] = b.offsets[i + 1] - keep[i];
      line0[i + 1] = line0[i] + keep[i];
    }
    const int64_t NL = line0[D];
    // line vectors: the slots' projections, then the MLP
    std::vector<float> e(size_t(NL) * SLOTS * fd), m(size_t(NL) * c_.line_ff), lines(size_t(NL) * d);
    parallel_for(D, [&](int64_t i) {
      for (int64_t l = 0; l < keep[i]; l++) {
        const int16_t* t = b.tokens + (first[i] + l) * COLUMNS;
        float* o = e.data() + (line0[i] + l) * SLOTS * fd;
        const int v[SLOTS] = {idx(t[KIND], ARG_TYPES), idx(t[SUB], BYTE), idx(t[ACTOR], IDENTS),
                              idx(t[OTHER], IDENTS), idx(t[MOVE], MOVES_N), idx(t[MOVE2], MOVES_N),
                              idx(t[SPECIES], SPECIES_N), std::clamp<int>(t[NUM], 0, BYTE - 1), 0,
                              t[STATUS] & 0xFF};
        for (int s = 0; s < SLOTS; s++) {
          float* os = o + s * fd;
          if (s == 8) {
            const float max_hp = t[MAX_HP];
            const float x0 = t[HP] / std::max(max_hp, 1.f), x1 = max_hp > 0 ? 1.f : 0.f;
            for (int f = 0; f < fd; f++) os[f] = hp_w_[f * 2] * x0 + hp_w_[f * 2 + 1] * x1 + hp_b_[f];
          } else {
            std::memcpy(os, slot_tab_[s].data() + size_t(v[s]) * fd, fd * 4);
          }
        }
      }
    });
    gemm(e.data(), NL, SLOTS * fd, line1_.w, c_.line_ff, line1_.b, m.data());
    parallel_for(NL, [&](int64_t i) {
      float* r = m.data() + i * c_.line_ff;
      for (int j = 0; j < c_.line_ff; j++) r[j] = gelu(r[j]);
    });
    gemm(m.data(), NL, c_.line_ff, line2_.w, d, line2_.b, lines.data());
    norm_rows(lines.data(), NL, line_norm_, lines.data());
    // the update encoder's sequences: summary, request kind, request, lines
    constexpr int PRE = 2 + REQUEST_TOKENS;
    std::vector<int64_t> start(D + 1, 0);
    for (int i = 0; i < D; i++) start[i + 1] = start[i] + PRE + keep[i];
    std::vector<float> x(size_t(start[D]) * d);
    std::vector<uint8_t> allowed(start[D], 1);
    parallel_for(D, [&](int64_t i) {
      float* o = x.data() + start[i] * d;
      std::memcpy(o, summary_token_, d * 4);
      const int16_t* q = b.request + i * REQUEST_FIELDS;
      std::memcpy(o + d, request_kind_ + size_t(idx(q[0], 3)) * d, d * 4);
      embed_request(q, o + 2 * d);
      for (int64_t l = 0; l < keep[i]; l++) {
        float* ol = o + (PRE + l) * d;
        const float* src = lines.data() + (line0[i] + l) * d;
        for (int j = 0; j < d; j++) ol[j] = src[j] + line_pos_[l * d + j];
        allowed[start[i] + PRE + l] = b.tokens[(first[i] + l) * COLUMNS + KIND] != 0;
      }
    });
    encode(update_layers_, x.data(), start, allowed.data());
    parallel_for(D, [&](int64_t i) {
      layer_norm_row(x.data() + start[i] * d, d, update_norm_.w, update_norm_.b, out + i * d);
    });
  }

  // ---- 2. the causal history, one position ----

  void causal(const Batch& b, float* x, float* K, float* V, int64_t S) {
    const int d = c_.d, D = int(b.D), R = this->R();
    for (int i = 0; i < D; i++) {
      const float* p = pos_ + std::min<int64_t>(b.t[i], c_.max_pos - 1) * d;
      for (int j = 0; j < d; j++) x[size_t(i) * d + j] += p[j];
    }
    std::vector<float> h(size_t(D) * d), qkv(size_t(D) * 3 * d), a(size_t(D) * d), f;
    for (size_t l = 0; l < causal_layers_.size(); l++) {
      const Layer& L = causal_layers_[l];
      norm_rows(x, D, L.n1, h.data());
      gemm(h.data(), D, d, L.qkv.w, 3 * d, L.qkv.b, qkv.data());
      parallel_for(D, [&](int64_t i) {
        thread_local std::vector<float> kv, sc;
        const int valid = int(std::min<int64_t>(b.t[i], R));
        float* kc = K + ((int64_t(l) * S + b.keys[i]) * R) * d;
        float* vc = V + ((int64_t(l) * S + b.keys[i]) * R) * d;
        const float* row = qkv.data() + i * 3 * d;
        // keys then values, the cache's valid slots and this position's
        kv.resize(size_t(2) * (valid + 1) * d);
        float* kk = kv.data();
        float* vv = kv.data() + size_t(valid + 1) * d;
        std::memcpy(kk, kc, size_t(valid) * d * 4);
        std::memcpy(vv, vc, size_t(valid) * d * 4);
        std::memcpy(kk + size_t(valid) * d, row + d, d * 4);
        std::memcpy(vv + size_t(valid) * d, row + 2 * d, d * 4);
        attend(row, 3 * d, 1, kk, vv, d, valid + 1, nullptr, a.data() + i * d, d, sc);
        const int64_t slot = b.t[i] % R;
        std::memcpy(kc + slot * d, row + d, d * 4);
        std::memcpy(vc + slot * d, row + 2 * d, d * 4);
      });
      gemm(a.data(), D, d, L.out.w, d, L.out.b, h.data());
      for (int64_t j = 0; j < int64_t(D) * d; j++) x[j] += h[j];
      feed_forward(L, x, D, h, f);
    }
    norm_rows(x, D, out_norm_, x);
  }

  // ---- 3. the decision block ----

  // moveset: the moves with their PP as a set (an empty slot adds nothing)
  void moveset(const int16_t* ids, const int16_t* pp, float* out) {
    const int d = c_.d;
    for (int k = 0; k < 4; k++) {
      if (ids[k] <= 0) continue;
      const float* t = move_pp_tab_.data() + size_t(idx(ids[k], MOVES_N)) * d;
      const float p = pp[k] / 64.f;
      for (int j = 0; j < d; j++) out[j] += gelu(t[j] + move_pp_.w[size_t(j) * (d + 1) + d] * p);
    }
  }

  void embed_state(const int16_t* s, float* out) {
    const int d = c_.d;
    std::fill(out, out + size_t(STATE_TOKENS) * d, 0.f);
    const float turn = s[0] / 100.f;
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
      float* o = out + side * d;
      gemv(active_scalars_.w, d, ACTIVE_SCALARS, sc, active_scalars_.b, o);
      const float* sp = species_ + size_t(idx(a[0], SPECIES_N)) * d;
      const float* t1 = type_emb_ + size_t(std::clamp<int>(a[1], 0, TYPES - 1)) * d;
      const float* t2 = type_emb_ + size_t(std::clamp<int>(a[2], 0, TYPES - 1)) * d;
      const float* dis = disabled_tab_.data() + size_t(idx(a[21], MOVES_N)) * d;
      for (int j = 0; j < d; j++) o[j] += sp[j] + t1[j] + t2[j] + dis[j];
      moveset(a + 13, a + 17, o);
    }
    for (int i = 0; i < 12; i++) {
      const int16_t* m = s + 1 + 2 * ACTIVE_FIELDS + i * MON_FIELDS;
      const float hp = m[5], max_hp = m[6];
      const float sc[8] = {float(m[0]),  float(m[1]),     float(m[2]), m[4] / 100.f, hp / std::max(max_hp, 1.f),
                           max_hp > 0 ? 1.f : 0.f, max_hp / 512.f, m[8] / 7.f};
      float* o = out + (2 + i) * d;
      gemv(mon_state_.w, d, 8, sc, mon_state_.b, o);
      const float* sp = species_ + size_t(idx(m[3], SPECIES_N)) * d;
      const float* st = status_ + size_t(m[7] & 0xFF) * d;
      for (int j = 0; j < d; j++) o[j] += sp[j] + st[j];
      moveset(m + 9, m + 13, o);
    }
    if (c_.features) active_features(s, out);
    for (int64_t j = 0; j < int64_t(STATE_TOKENS) * d; j++) out[j] += state_type_[j];
  }

  // cs3's additions to the two active tokens (out[0..1])
  void active_features(const int16_t* s, float* out) {
    const int d = c_.d;
    const int16_t* mons = s + 1 + 2 * ACTIVE_FIELDS;
    std::vector<float> row(d + 2), x(d);
    for (int side = 0; side < 2; side++) {
      std::fill(row.begin(), row.end(), 0.f);
      float sides[5] = {0, 0, 0, 0, 0};
      bool sleep = false, frozen = false;
      for (int k = 0; k < 6; k++) {
        const int16_t* m = mons + (side * 6 + k) * MON_FIELDS;
        const bool exists = m[0] > 0, seen = m[1] > 0;
        const float active = m[2];
        const int status = m[7] & 0xFF;
        const float hp = m[5], max_hp = m[6], frac = hp / std::max(max_hp, 1.f);
        if (active != 0.f) {
          const float* st = status_ + size_t(status) * d;
          for (int j = 0; j < d; j++) row[j] += st[j] * active;
          row[d] += frac * active;
          row[d + 1] += m[8] / 7.f * active;
        }
        const bool alive = exists && (!seen || hp > 0);
        sides[0] += alive;
        sides[1] += seen ? frac : float(exists);
        sides[2] += exists && !seen;
        sleep |= (status & SLP) && !(status & REST_BIT) && hp > 0;
        frozen |= status == FRZ && hp > 0;
      }
      sides[0] /= 6;
      sides[1] /= 6;
      sides[2] /= 6;
      sides[3] = sleep;
      sides[4] = frozen;
      float* o = out + side * d;
      gemv(active_extra_.w, d, d + 2, row.data(), active_extra_.b, o, true);
      gemv(side_extra_.w, d, 5, sides, side_extra_.b, o, true);
    }
    // the foe's known moves' threat, on the foe's active token
    const int16_t* f = s + BASE_STATE_FIELDS;
    const int16_t* foe_ids = s + 1 + ACTIVE_FIELDS + 13;
    for (int k = 0; k < 4; k++) {
      if (foe_ids[k] <= 0) continue;
      float v[MOVE_FEATS];
      for (int j = 0; j < MOVE_FEATS; j++) v[j] = f[(4 + k) * MOVE_FEATS + j] * MOVE_SCALE[j];
      gemv(foe_threat_.w, d, MOVE_FEATS, v, foe_threat_.b, out + d, true);
    }
  }

  // cs3's additions to the request tokens (req [REQUEST_TOKENS, d])
  void request_features(const int16_t* q, const int16_t* s, float* req) {
    const int d = c_.d;
    const int16_t* f = s + BASE_STATE_FIELDS;
    const int16_t* ids = s + 1 + 13;  // our active's moves, the features' order
    for (int i = 0; i < ACTIVE_MOVES; i++) {
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
      gemv(move_feat_.w, d, MOVE_FEATS, v, move_feat_.b, req + (1 + i) * d, true);
    }
    const int16_t* party = s + 1 + 2 * ACTIVE_FIELDS;
    for (int i = 0; i < SIDE; i++) {
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
      gemv(switch_feat_.w, d, SWITCH_FEATS, v, switch_feat_.b, req + (1 + ACTIVE_MOVES + i) * d, true);
    }
  }

  void decide(const Batch& b, const float* hist, float* logits, float* value) {
    const int d = c_.d, D = int(b.D);
    const int T = 2 + REQUEST_TOKENS + (c_.state ? STATE_TOKENS : 0);
    if (c_.state && !b.state) throw std::invalid_argument("net: the net reads the state");
    std::vector<int64_t> start(D + 1);
    for (int i = 0; i <= D; i++) start[i] = int64_t(i) * T;
    std::vector<float> x(size_t(D) * T * d);
    parallel_for(D, [&](int64_t i) {
      float* o = x.data() + i * T * d;
      std::memcpy(o, value_token_, d * 4);
      for (int j = 0; j < d; j++) o[d + j] = hist[i * d + j] + h_type_[j];
      const int16_t* q = b.request + i * REQUEST_FIELDS;
      embed_request(q, o + 2 * d);
      if (c_.state) {
        const int16_t* s = b.state + i * STATE_FIELDS;
        if (c_.features) request_features(q, s, o + 2 * d);
        embed_state(s, o + (2 + REQUEST_TOKENS) * d);
      }
    });
    encode(history_layers_, x.data(), start, nullptr);
    parallel_for(D, [&](int64_t i) {
      float* o = x.data() + i * T * d;
      layer_norm_row(o, d, norm_.w, norm_.b, o);  // value token
      for (int t = 2; t < 2 + REQUEST_TOKENS; t++) layer_norm_row(o + t * d, d, norm_.w, norm_.b, o + t * d);
      gemv(value_head_.w, 3, d, o, value_head_.b, value + i * 3);
      // the logit of choice c: query(mons[0]) . key(target + E[c]) / sqrt(d)
      std::vector<float> qv(d), tc(d), kk(d);
      const float* req = o + 2 * d;
      gemv(query_.w, d, d, req + (1 + ACTIVE_MOVES) * d, query_.b, qv.data());
      const uint8_t* ch = b.choices + i * CHOICES;
      const int n = b.n[i];
      for (int c = 0; c < CHOICES; c++) {
        if (c >= n) {
          logits[i * CHOICES + c] = -std::numeric_limits<float>::infinity();
          continue;
        }
        const int kind = ch[c] & 3, slot = ch[c] >> 2;
        const int target = kind == 2 ? ACTIVE_MOVES + slot : slot;  // into (nothing, moves, side)
        const float* e = choice_ + size_t(ch[c]) * d;
        if (target == 0) {
          std::memcpy(tc.data(), e, d * 4);
        } else {
          const float* p = req + size_t(idx(target, REQUEST_TOKENS)) * d;
          for (int j = 0; j < d; j++) tc[j] = p[j] + e[j];
        }
        gemv(key_.w, d, d, tc.data(), key_.b, kk.data());
        float s = 0;
        for (int j = 0; j < d; j++) s += qv[j] * kk[j];
        logits[i * CHOICES + c] = s / std::sqrt(float(d));
      }
    });
  }
};

}  // namespace ataraxos::net
