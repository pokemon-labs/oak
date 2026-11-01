// Python binding for src/client.hpp's Client (birch's birch_client.cpp):
// one Showdown battle as one player sees it, for ataraxos/play/showdown.py.
// Built with ataraxos_env by CMake (CMakeLists.txt) into ataraxos/_ext/.
//
// Client():
//   feed(message): one message of the battle room, its ">room" line removed,
//     `|request|` lines included (read past here; hand them to `request`).
//   request(json) -> bool: whether it asks for a choice (not a wait).
//   take() -> dict: the decision for the last request, in the env's layout
//     for one decision (tokens i16 [T, COLUMNS] since the last take,
//     offsets i64 [2], request i16 [1, REQUEST_FIELDS], choices u8 [1,
//     MAX_CHOICES], n u8 [1], state i16 [1, STATE_FIELDS]); the tokens are
//     then cleared.
//   n: the last request's number of choices.
//   choice(i) -> str: "/choose ...|rqid" for choice i of the last request.
//   choose(i) -> str: the same, and the bookkeeping hears it: call it for
//     the choice actually sent.
//   revealed(i) -> bool: whether choice i names only what the log has shown
//     the other player (a Pokémon that has appeared, a move used).
//   unknown() -> list: tags neither read nor read past, then cleared.
// Text or JSON it can't read raises RuntimeError (ParseError).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "client.hpp"

namespace py = pybind11;
using ataraxos::Client;

PYBIND11_MODULE(ataraxos_client, m) {
  m.doc() = "One Showdown battle as one player sees it (src/client.hpp).";
  py::class_<Client>(m, "Client")
      .def(py::init<>())
      .def("feed", [](Client& c, const std::string& message) { c.feed(message); }, py::arg("message"))
      .def("request", [](Client& c, const std::string& text) { return c.request(text); }, py::arg("json"))
      .def("choice", &Client::choice, py::arg("i"))
      .def("choose", &Client::choose, py::arg("i"))
      .def("revealed", &Client::revealed, py::arg("i"))
      .def_property_readonly("n", [](const Client& c) { return int(c.n); })
      .def("viewer", &Client::viewer)
      .def("unknown",
           [](Client& c) {
             std::vector<std::string> out = std::move(c.unknown);
             c.unknown.clear();
             return out;
           })
      .def("take", [](Client& c) {
        const size_t T = c.tokens.size();
        py::array_t<int16_t> tokens({py::ssize_t(T), py::ssize_t(ataraxos::COLUMNS)});
        std::memcpy(tokens.mutable_data(), c.tokens.data(), T * sizeof(ataraxos::Token));
        c.tokens.clear();
        py::array_t<int64_t> offsets(2);
        offsets.mutable_data()[0] = 0;
        offsets.mutable_data()[1] = int64_t(T);
        py::array_t<int16_t> request({py::ssize_t(1), py::ssize_t(ataraxos::REQUEST_FIELDS)});
        std::memcpy(request.mutable_data(), c.fields.data(), ataraxos::REQUEST_FIELDS * sizeof(int16_t));
        py::array_t<uint8_t> choices({py::ssize_t(1), py::ssize_t(ataraxos::MAX_CHOICES)});
        for (size_t i = 0; i < ataraxos::MAX_CHOICES; i++) choices.mutable_data()[i] = i < c.n ? c.options[i] : 0;
        py::array_t<uint8_t> n(1);
        n.mutable_data()[0] = c.n;
        py::array_t<int16_t> state({py::ssize_t(1), py::ssize_t(ataraxos::STATE_FIELDS)});
        std::memcpy(state.mutable_data(), c.state().data(), ataraxos::STATE_FIELDS * sizeof(int16_t));
        py::dict out;
        out["tokens"] = tokens;
        out["offsets"] = offsets;
        out["request"] = request;
        out["choices"] = choices;
        out["n"] = n;
        out["state"] = state;
        return out;
      });
}
