"""A checkpoint's EMA weights to src/net.hpp's weight file, without torch
(2026-11-21, for the pkmn.ai bundle: the arena has no torch).

    python -m ataraxos.arena.export CKPT OUT [--weights ema|net]

A torch checkpoint is a zip: `data.pkl` (a pickle whose tensors are
persistent ids naming raw storages) and `data/KEY` (the storages' bytes).
`load` reads it with a restricted unpickler: torch's tensor rebuilders and
storage classes become numpy, a few plain containers and numpy's own
classes are allowed, and anything else becomes an inert stub (the
optimizer's or the RNG's odds and ends; never called).

The output (`write`): "ATXNET1\\n", text lines "cfg KEY VALUE" and "t NAME
OFFSET DIM...", "end\\n", zero padding to HEADER bytes, then each tensor as
little-endian float32 at OFFSET (from the data's start, 64-byte aligned).
The config is what the shapes don't say: heads and window, from the
checkpoint's `size` (SIZES below mirrors ataraxos/nets/causal_net.py's)."""

import argparse
import io
import pickle
import sys
import zipfile

import numpy as np

MAGIC = b"ATXNET1\n"
HEADER = 1 << 16  # the text header's room; the data starts here
ALIGN = 64

# causal_net.SIZES' (heads, window) per size name
SIZES = {"cs": (4, 32), "cs2": (4, 32), "cs3": (4, 8), "cs_tiny": (2, 6), "cs3_tiny": (2, 6)}

DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "BFloat16Storage": "bfloat16", "LongStorage": np.int64, "IntStorage": np.int32,
    "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class Stub:
    """Whatever the checkpoint pickles that the net doesn't need."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return Stub()

    def __setstate__(self, state):
        pass


class StorageType:
    def __init__(self, name):
        self.name = name


def rebuild_tensor(storage, offset, size, stride, *rest):
    data, dtype = storage
    item = np.dtype(dtype).itemsize
    if not size:
        return data[offset].copy()
    return np.lib.stride_tricks.as_strided(data[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()


def rebuild_parameter(data, *rest):
    return data


class Unpickler(pickle.Unpickler):
    ALLOWED = {("collections", "OrderedDict"), ("builtins", "set"), ("builtins", "frozenset"),
               ("builtins", "slice"), ("builtins", "complex"),
               ("_codecs", "encode")}  # numpy arrays' bytes in protocol 2

    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return rebuild_parameter
        if module == "torch" and name in DTYPES:
            return StorageType(name)
        if (module, name) in self.ALLOWED or module.split(".")[0] == "numpy" and name in (
                "_reconstruct", "ndarray", "dtype", "scalar", "_frombuffer"):
            return super().find_class(module, name)
        return Stub

    def persistent_load(self, pid):
        kind, stype, key, _location, numel = pid[:5]
        assert kind == "storage", pid
        name = stype.name if isinstance(stype, StorageType) else "?"
        if name not in DTYPES:
            raise ValueError(f"storage type {stype!r}")
        if key not in self.cache:
            raw = self.zf.read(f"{self.prefix}data/{key}")
            dtype = DTYPES[name]
            if dtype == "bfloat16":
                a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
                dtype = np.float32
            else:
                a = np.frombuffer(raw, dtype)
            self.cache[key] = (a, dtype)
        return self.cache[key]


def load(path):
    """The checkpoint's pickled object, its tensors as numpy arrays."""
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith("data.pkl") and n.count("/") == 1]
        if len(pkl) != 1:
            raise ValueError(f"{path}: not a torch zip checkpoint ({pkl})")
        prefix = pkl[0][:-len("data.pkl")]
        order = prefix + "byteorder"
        if order in zf.namelist() and zf.read(order).strip() != b"little":
            raise ValueError("big-endian checkpoint")
        return Unpickler(io.BytesIO(zf.read(pkl[0])), zf, prefix).load()


def net_config(config, weights):
    """The C++ net's config: what the shapes give, heads and window by size."""
    size = config["size"]
    if size not in SIZES:
        raise ValueError(f"size {size!r} not in export.SIZES (mirror causal_net.SIZES)")
    heads, window = SIZES[size]
    count = lambda p: len({k.split(".")[len(p.split("."))] for k in weights if k.startswith(p + ".")})
    return dict(
        d=weights["norm.weight"].shape[0], heads=heads, window=window,
        ff=weights["layers.0.ff.0.weight"].shape[0],
        field_d=weights["line_encoder.slot.0.weight"].shape[0],
        line_ff=weights["line_encoder.mlp.0.weight"].shape[0],
        max_lines=weights["line_pos"].shape[0], max_pos=weights["pos"].shape[0],
        depth=count("history_encoder.layers"), update_depth=count("update_encoder.encoder.layers"),
        causal_depth=count("layers"), state=int("state_type" in weights),
        features=int("active_extra.weight" in weights), q_head=int("q_head.0.weight" in weights))


def write(path, config, weights):
    lines, off, blobs = [MAGIC.decode()], 0, []
    for k, v in config.items():
        lines.append(f"cfg {k} {v}\n")
    for name in sorted(weights):
        a = np.ascontiguousarray(weights[name], dtype="<f4")
        lines.append(f"t {name} {off} {' '.join(str(s) for s in a.shape)}\n".replace("  ", " "))
        blobs.append((off, a))
        off += -(-a.nbytes // ALIGN) * ALIGN
    lines.append("end\n")
    head = "".join(lines).encode()
    if len(head) > HEADER:
        raise ValueError(f"header {len(head)} > {HEADER}")
    with open(path, "wb") as f:
        f.write(head + b"\0" * (HEADER - len(head)))
        for o, a in blobs:
            f.seek(HEADER + o)
            f.write(a.tobytes())
        f.truncate(HEADER + off)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--weights", default="ema", choices=["ema", "net"])
    a = ap.parse_args(argv)
    ck = load(a.ckpt)
    w = {k: np.asarray(v) for k, v in ck[a.weights].items()}
    cfg = net_config(ck["config"], w)
    write(a.out, cfg, w)
    params = sum(v.size for v in w.values())
    print(f"{a.out}: {len(w)} tensors, {params / 1e6:.2f}M params, config {cfg}", file=sys.stderr)


if __name__ == "__main__":
    main()
