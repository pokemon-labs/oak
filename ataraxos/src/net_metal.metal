// The causal net's forward pass on Apple GPUs (2026-11-28): src/net.hpp's
// math as Metal compute kernels, float32 throughout (the CPU path's
// precision), for src/net_metal.mm. CMake embeds this file in
// ataraxos_net as a string; it is compiled at load time (newLibraryWithSource,
// fast math off), so the bundle carries no .metallib.
//
// Kernels assume 32-wide SIMD-groups (Apple GPUs; net_metal.mm checks) and
// 128-thread threadgroups. Layouts are net.hpp's: rows of d floats,
// qkv rows [q | k | v], linears' weights [out, in].

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint SG = 4;  // SIMD-groups a threadgroup

// erf by Abramowitz & Stegun 7.1.26 (|error| <= 1.5e-7; MSL has no erf)
inline float erf_as(float x) {
  const float ax = fabs(x);
  const float t = 1.f / (1.f + 0.3275911f * ax);
  const float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t +
                      0.254829592f) * t;
  const float y = 1.f - poly * precise::exp(-ax * ax);
  return x < 0.f ? -y : y;
}

inline float gelu(float x) { return 0.5f * x * (1.f + erf_as(x * 0.70710678118654752f)); }

// ---- GEMM: y[M, N] (+)= f(x[M, K] W[N, K]^T + b) ----------------------------
// x's row m is x + (rows ? rows[m] : m) * lda. K and lda are multiples of 4.

struct GemmArgs {
  uint M, N, K, lda;
  uint has_bias, has_rows, gelu, accumulate;
};

constant constexpr uint BM = 64, BN = 64, BK = 32, LDS = BK + 4;  // LDS: the tiles' padded row

kernel void gemm(device const float* x [[buffer(0)]], device const float* W [[buffer(1)]],
                 device const float* bias [[buffer(2)]], device float* y [[buffer(3)]],
                 device const uint* rows [[buffer(4)]], constant GemmArgs& a [[buffer(5)]],
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float smem[(BM + BN) * LDS];  // A and B tiles; then the C tile [BM][BN]
  threadgroup float* As = smem;
  threadgroup float* Bs = smem + BM * LDS;
  const uint m0 = tg.y * BM, n0 = tg.x * BN;
  const uint sm = sg >> 1, sn = sg & 1;  // this SIMD-group's 32x32 quarter
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);
  for (uint k0 = 0; k0 < a.K; k0 += BK) {
    for (uint e = tid; e < BM * BK / 4; e += SG * 32) {
      const uint r = e / (BK / 4), c = (e % (BK / 4)) * 4, k = k0 + c;
      float4 va = 0.f, vb = 0.f;
      const uint m = m0 + r, n = n0 + r;
      if (m < a.M && k < a.K) va = *(device const float4*)(x + (a.has_rows ? rows[m] : m) * a.lda + k);
      if (n < a.N && k < a.K) vb = *(device const float4*)(W + n * a.K + k);
      *(threadgroup float4*)(As + r * LDS + c) = va;
      *(threadgroup float4*)(Bs + r * LDS + c) = vb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 af[4], bf[4];
      for (uint i = 0; i < 4; i++) simdgroup_load(af[i], As + (sm * 32 + i * 8) * LDS + kk, LDS);
      for (uint j = 0; j < 4; j++) simdgroup_load(bf[j], Bs + (sn * 32 + j * 8) * LDS + kk, LDS, ulong2(0, 0), true);
      for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* Cs = smem;
  for (uint i = 0; i < 4; i++)
    for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], Cs + (sm * 32 + i * 8) * BN + sn * 32 + j * 8, BN);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < BM * BN; e += SG * 32) {
    const uint m = m0 + e / BN, n = n0 + e % BN;
    if (m >= a.M || n >= a.N) continue;
    float v = Cs[e];
    if (a.has_bias) v += bias[n];
    if (a.gelu) v = gelu(v);
    device float* o = y + m * a.N + n;
    *o = a.accumulate ? *o + v : v;
  }
}

// ---- layer norm: y[dst[r]] = norm(x[src[r]]), one SIMD-group a row --------

struct NormArgs {
  uint rows, d, has_src, has_dst;
  float eps;
};

kernel void layer_norm(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       device const float* w [[buffer(2)]], device const float* b [[buffer(3)]],
                       device const uint* src [[buffer(4)]], device const uint* dst [[buffer(5)]],
                       constant NormArgs& a [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg * SG + sg;
  if (r >= a.rows) return;
  device const float* xi = x + (a.has_src ? src[r] : r) * a.d;
  device float* yi = y + (a.has_dst ? dst[r] : r) * a.d;
  float s = 0.f;
  for (uint j = lane; j < a.d; j += 32) s += xi[j];
  const float mean = simd_sum(s) / float(a.d);
  float v = 0.f;
  for (uint j = lane; j < a.d; j += 32) {
    const float t = xi[j] - mean;
    v += t * t;
  }
  const float inv = 1.f / precise::sqrt(simd_sum(v) / float(a.d) + a.eps);
  for (uint j = lane; j < a.d; j += 32) yi[j] = (xi[j] - mean) * inv * w[j] + b[j];
}

// ---- token embeddings: out[t] = sum of coef * row over t's terms ----------
// A term's offset is into P (the parameters) or, with bit 31, into A (an
// activation buffer: the line vectors, or the history's outputs).

struct EmbedArgs {
  uint tokens, d;
};

kernel void embed(device const float* P [[buffer(0)]], device const float* A [[buffer(1)]],
                  device const uint* start [[buffer(2)]], device const uint* count [[buffer(3)]],
                  device const uint* off [[buffer(4)]], device const float* coef [[buffer(5)]],
                  device float* out [[buffer(6)]], constant EmbedArgs& a [[buffer(7)]],
                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  const uint t = tg * SG + sg;
  if (t >= a.tokens) return;
  const uint s0 = start[t], n = count[t];
  for (uint j = lane; j < a.d; j += 32) {
    float acc = 0.f;
    for (uint q = 0; q < n; q++) {
      const uint o = off[s0 + q];
      const float v = (o & 0x80000000u) ? A[(o & 0x7FFFFFFFu) + j] : P[o + j];
      acc += coef[s0 + q] * v;
    }
    out[t * a.d + j] = acc;
  }
}

// ---- the line encoder's input: each slot's folded table row ---------------

struct SlotArgs {
  uint lines, fd, hp_w, hp_b;
  uint tab[10];  // each slot's table (slot 8, hp, is the linear hp_w, hp_b)
};

kernel void line_slots(device const float* P [[buffer(0)]], device const int* v [[buffer(1)]],
                       device const float* hp [[buffer(2)]], device float* e [[buffer(3)]],
                       constant SlotArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const uint W = 10 * a.fd;
  if (gid >= a.lines * W) return;
  const uint l = gid / W, c = gid % W, s = c / a.fd, f = c % a.fd;
  if (s == 8)
    e[gid] = P[a.hp_w + f * 2] * hp[l * 2] + P[a.hp_w + f * 2 + 1] * hp[l * 2 + 1] + P[a.hp_b + f];
  else
    e[gid] = P[a.tab[s] + uint(v[l * 10 + s]) * a.fd + f];
}

// x[r] += P[off[r]] (the causal positions)
kernel void add_rows(device float* x [[buffer(0)]], device const float* P [[buffer(1)]],
                     device const uint* off [[buffer(2)]], constant uint2& nd [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
  if (gid >= nd.x * nd.y) return;
  x[gid] += P[off[gid / nd.y] + gid % nd.y];
}

// ---- attention --------------------------------------------------------------
// Keys sit on lanes (key j on lane j % 32, up to 4 a lane: 128 keys), the
// output's elements too (e and e + 32: head size <= 64).

struct AttnArgs {
  uint d, heads, has_mask;
  float scale;
};

// self-attention within sequences [start[s], start[s + 1]) of qkv [N, 3d];
// a threadgroup per (sequence, head), its SIMD-groups take the queries
kernel void attend(device const float* qkv [[buffer(0)]], device const uint* start [[buffer(1)]],
                   device const uchar* mask [[buffer(2)]], device float* out [[buffer(3)]],
                   constant AttnArgs& a [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint seq = tg / a.heads, h = tg % a.heads, hd = a.d / a.heads, D3 = 3 * a.d;
  const uint s0 = start[seq], T = start[seq + 1] - s0;
  for (uint i = sg; i < T; i += SG) {
    device const float* q = qkv + (s0 + i) * D3 + h * hd;
    float sc[4];
    float mx = -INFINITY;
    for (uint u = 0; u < 4; u++) {
      const uint j = lane + 32 * u;
      sc[u] = -INFINITY;
      if (j < T && (!a.has_mask || mask[s0 + j])) {
        device const float* k = qkv + (s0 + j) * D3 + a.d + h * hd;
        float dot = 0.f;
        for (uint e = 0; e < hd; e++) dot += q[e] * k[e];
        sc[u] = dot * a.scale;
      }
      mx = max(mx, sc[u]);
    }
    mx = simd_max(mx);
    float z = 0.f;
    for (uint u = 0; u < 4; u++) {
      sc[u] = sc[u] == -INFINITY ? 0.f : precise::exp(sc[u] - mx);
      z += sc[u];
    }
    z = simd_sum(z);
    float o0 = 0.f, o1 = 0.f;
    for (uint j = 0; j < T; j++) {
      const float mine = j < 32 ? sc[0] : j < 64 ? sc[1] : j < 96 ? sc[2] : sc[3];
      const float p = simd_shuffle(mine, ushort(j & 31));
      if (p == 0.f) continue;
      device const float* v = qkv + (s0 + j) * D3 + 2 * a.d + h * hd;
      if (lane < hd) o0 += p * v[lane];
      if (lane + 32 < hd) o1 += p * v[lane + 32];
    }
    device float* o = out + (s0 + i) * a.d + h * hd;
    if (lane < hd) o[lane] = o0 / z;
    if (lane + 32 < hd) o[lane + 32] = o1 / z;
  }
}

struct CausalArgs {
  uint rows, d, heads, R;
  float scale;
};

// one causal position a row: its query against the cache's valid[i] keys
// (Kc, Vc [rows, R, d], this layer's) and its own; writes its own key and
// value to nk, nv [rows, d]. A SIMD-group per (row, head).
kernel void causal_attend(device const float* qkv [[buffer(0)]], device const float* Kc [[buffer(1)]],
                          device const float* Vc [[buffer(2)]], device const uint* valid [[buffer(3)]],
                          device float* out [[buffer(4)]], device float* nk [[buffer(5)]],
                          device float* nv [[buffer(6)]], constant CausalArgs& a [[buffer(7)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint job = tg * SG + sg;
  if (job >= a.rows * a.heads) return;
  const uint i = job / a.heads, h = job % a.heads, hd = a.d / a.heads;
  const uint V = valid[i], T = V + 1;
  device const float* row = qkv + i * 3 * a.d;
  device const float* q = row + h * hd;
  device const float* kn = row + a.d + h * hd;
  device const float* vn = row + 2 * a.d + h * hd;
  float sc[4];
  float mx = -INFINITY;
  for (uint u = 0; u < 4; u++) {
    const uint j = lane + 32 * u;
    sc[u] = -INFINITY;
    if (j < T) {
      device const float* k = j < V ? Kc + (i * a.R + j) * a.d + h * hd : kn;
      float dot = 0.f;
      for (uint e = 0; e < hd; e++) dot += q[e] * k[e];
      sc[u] = dot * a.scale;
    }
    mx = max(mx, sc[u]);
  }
  mx = simd_max(mx);
  float z = 0.f;
  for (uint u = 0; u < 4; u++) {
    sc[u] = sc[u] == -INFINITY ? 0.f : precise::exp(sc[u] - mx);
    z += sc[u];
  }
  z = simd_sum(z);
  float o0 = 0.f, o1 = 0.f;
  for (uint j = 0; j < T; j++) {
    const float mine = j < 32 ? sc[0] : j < 64 ? sc[1] : j < 96 ? sc[2] : sc[3];
    const float p = simd_shuffle(mine, ushort(j & 31));
    if (p == 0.f) continue;
    device const float* v = j < V ? Vc + (i * a.R + j) * a.d + h * hd : vn;
    if (lane < hd) o0 += p * v[lane];
    if (lane + 32 < hd) o1 += p * v[lane + 32];
  }
  device float* o = out + i * a.d + h * hd;
  if (lane < hd) {
    o[lane] = o0 / z;
    nk[i * a.d + h * hd + lane] = kn[lane];
    nv[i * a.d + h * hd + lane] = vn[lane];
  }
  if (lane + 32 < hd) {
    o[lane + 32] = o1 / z;
    nk[i * a.d + h * hd + lane + 32] = kn[lane + 32];
    nv[i * a.d + h * hd + lane + 32] = vn[lane + 32];
  }
}

// ---- the decision block's heads ---------------------------------------------
// Per row (x: its T tokens, normed): value logits from token 0, and choice
// c's logit query . (Kr[target] + KE[choice byte] + key bias) / sqrt(d),
// Kr the request tokens' keys (no bias), KE the choice table's.

struct HeadArgs {
  uint rows, d, T, choices, req_tokens;
  uint value_w, value_b, key_b, ke;
  float sqrt_d;
};

kernel void heads(device const float* P [[buffer(0)]], device const float* x [[buffer(1)]],
                  device const float* qv [[buffer(2)]], device const float* Kr [[buffer(3)]],
                  device const int* tgt [[buffer(4)]], device const uint* ch [[buffer(5)]],
                  device const uint* n [[buffer(6)]], device float* logits [[buffer(7)]],
                  device float* value [[buffer(8)]], constant HeadArgs& a [[buffer(9)]],
                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
  const uint per = a.choices + 3, job = tg * SG + sg;
  if (job >= a.rows * per) return;
  const uint i = job / per, c = job % per;
  if (c >= a.choices) {
    const uint o = c - a.choices;
    device const float* xi = x + i * a.T * a.d;
    float s = 0.f;
    for (uint j = lane; j < a.d; j += 32) s += P[a.value_w + o * a.d + j] * xi[j];
    s = simd_sum(s);
    if (lane == 0) value[i * 3 + o] = s + P[a.value_b + o];
    return;
  }
  if (c >= n[i]) {
    if (lane == 0) logits[i * a.choices + c] = -INFINITY;
    return;
  }
  const int t = tgt[i * a.choices + c];
  const uint e = ch[i * a.choices + c];
  float s = 0.f;
  for (uint j = lane; j < a.d; j += 32) {
    float k = P[a.ke + e * a.d + j] + P[a.key_b + j];
    if (t >= 0) k += Kr[(i * a.req_tokens + uint(t)) * a.d + j];
    s += qv[i * a.d + j] * k;
  }
  s = simd_sum(s);
  if (lane == 0) logits[i * a.choices + c] = s / a.sqrt_d;
}
