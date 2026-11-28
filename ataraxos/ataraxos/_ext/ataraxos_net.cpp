// Python binding for src/net.hpp (2026-11-21): the causal net's forward pass
// without torch, for the pkmn.ai bundle (ataraxos/arena/). Needs neither
// libpkmn nor torch: Eigen (oak's submodule) for the GEMMs off Apple,
// Accelerate on Apple.
//
// set_threads(n): the threads the net's loops (and Eigen's GEMMs) use.
// Net(path): export.py's weight file.
//   config -> dict.
//   act(tokens i16 [T, COLUMNS], offsets i64 [D + 1], request i16 [D,
//     REQUEST_FIELDS], state i16 [D, STATE_FIELDS] or None, choices u8 [D,
//     CHOICES], n u8 [D], keys i64 [D], t i64 [D], K, V f32 [layers, slots,
//     window - 1, d] (written in place), decide=True; the state only with decide) -> (logits f32 [D,
//     CHOICES], -inf past n; value logits f32 [D, 3]), or None with
//     decide=False (a prefill: the summaries and the cache writes only).
//   set_gpu(on) -> "" or why not (2026-11-28): act on the GPU (Metal, macOS
//     builds; src/net_metal.mm) or the CPU (the default). gpu -> bool.
//   timing -> dict: act's calls, rows, seconds and the GPU's own seconds.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

#include "net.hpp"

namespace py = pybind11;
using namespace ataraxos::net;

namespace {

template <class T>
using In = py::array_t<T, py::array::c_style | py::array::forcecast>;

void expect(const py::array& a, std::vector<py::ssize_t> shape, const char* name) {
  if (a.ndim() != py::ssize_t(shape.size())) throw std::invalid_argument(std::string("act: ") + name + "'s rank");
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] >= 0 && a.shape(i) != shape[i])
      throw std::invalid_argument(std::string("act: ") + name + "'s shape (dim " + std::to_string(i) + ")");
  }
}

float* cache(py::array& a, const Net& net, py::ssize_t& slots, const char* name) {
  const Config& c = net.config();
  if (!py::isinstance<py::array_t<float>>(a) || !(a.flags() & py::array::c_style) || !a.writeable())
    throw std::invalid_argument(std::string("act: ") + name + " must be a writable C-contiguous float32 array");
  expect(a, {c.causal_depth, -1, net.R(), c.d}, name);
  slots = a.shape(1);
  return static_cast<float*>(a.mutable_data());
}

}  // namespace

PYBIND11_MODULE(ataraxos_net, m) {
  m.doc() = "The causal net's forward pass without torch (src/net.hpp).";
  m.attr("COLUMNS") = COLUMNS;
  m.attr("REQUEST_FIELDS") = REQUEST_FIELDS;
  m.attr("STATE_FIELDS") = STATE_FIELDS;
  m.attr("CHOICES") = CHOICES;
  m.def(
      "set_threads",
      [](int n) {
        threads_setting() = std::max(1, n);
#ifndef __APPLE__
        Eigen::setNbThreads(std::max(1, n));
#endif
      },
      py::arg("n"));
  py::class_<Net>(m, "Net")
      .def(py::init<std::string>(), py::arg("path"))
      .def_property_readonly("config",
                             [](const Net& net) {
                               const Config& c = net.config();
                               py::dict d;
                               d["d"] = c.d;
                               d["heads"] = c.heads;
                               d["window"] = c.window;
                               d["ff"] = c.ff;
                               d["field_d"] = c.field_d;
                               d["line_ff"] = c.line_ff;
                               d["max_lines"] = c.max_lines;
                               d["max_pos"] = c.max_pos;
                               d["depth"] = c.depth;
                               d["update_depth"] = c.update_depth;
                               d["causal_depth"] = c.causal_depth;
                               d["state"] = c.state;
                               d["features"] = c.features;
                               d["q_head"] = c.q_head;
                               return d;
                             })
      .def(
          "set_gpu",
          [](Net& net, bool on) -> std::string {
#ifdef ATARAXOS_METAL
            return net.set_gpu(on);
#else
            (void)net;
            return on ? "not built with Metal (macOS only)" : "";
#endif
          },
          py::arg("on"))
      .def_property_readonly("gpu",
                             [](const Net& net) {
#ifdef ATARAXOS_METAL
                               return net.gpu();
#else
                               (void)net;
                               return false;
#endif
                             })
      .def_property_readonly("timing",
                             [](const Net& net) {
                               py::dict d;
                               d["calls"] = net.calls();
                               d["rows"] = net.rows();
                               d["seconds"] = net.seconds();
                               d["gpu_seconds"] = net.gpu_seconds();
                               return d;
                             })
      .def(
          "act",
          [](Net& net, In<int16_t> tokens, In<int64_t> offsets, In<int16_t> request, py::object state,
             In<uint8_t> choices, In<uint8_t> n, In<int64_t> keys, In<int64_t> t, py::array K, py::array V,
             bool decide) -> py::object {
            const py::ssize_t D = n.shape(0);
            expect(tokens, {-1, COLUMNS}, "tokens");
            expect(offsets, {D + 1}, "offsets");
            expect(request, {D, REQUEST_FIELDS}, "request");
            expect(choices, {D, CHOICES}, "choices");
            expect(n, {D}, "n");
            expect(keys, {D}, "keys");
            expect(t, {D}, "t");
            const int64_t* off = offsets.data();
            if (D > 0 && (off[0] < 0 || off[D] > tokens.shape(0))) throw std::invalid_argument("act: offsets");
            In<int16_t> st;
            const bool reads_state = net.config().state && decide;  // a prefill doesn't
            if (reads_state) {
              if (state.is_none()) throw std::invalid_argument("act: the net reads the state");
              st = In<int16_t>::ensure(state);
              if (!st) throw std::invalid_argument("act: state");
              expect(st, {D, STATE_FIELDS}, "state");
            }
            py::ssize_t sk = 0, sv = 0;
            float* k = cache(K, net, sk, "K");
            float* v = cache(V, net, sv, "V");
            if (sk != sv) throw std::invalid_argument("act: K and V differ");
            Batch b;
            b.D = D;
            b.tokens = tokens.data();
            b.offsets = off;
            b.request = request.data();
            b.state = reads_state ? st.data() : nullptr;
            b.choices = choices.data();
            b.n = n.data();
            b.keys = keys.data();
            b.t = t.data();
            if (!decide) {
              py::gil_scoped_release release;
              net.act(b, k, v, sk, nullptr, nullptr);
              return py::none();
            }
            py::array_t<float> logits({D, py::ssize_t(CHOICES)}), value({D, py::ssize_t(3)});
            float* lp = logits.mutable_data();
            float* vp = value.mutable_data();
            {
              py::gil_scoped_release release;
              net.act(b, k, v, sk, lp, vp);
            }
            return py::make_tuple(logits, value);
          },
          py::arg("tokens"), py::arg("offsets"), py::arg("request"), py::arg("state"), py::arg("choices"),
          py::arg("n"), py::arg("keys"), py::arg("t"), py::arg("K"), py::arg("V"), py::arg("decide") = true);
}
