"""The pkmn.ai arena bot (2026-11-21): one game per process over stdin and
stdout (https://pkmn.ai/rules; the arena's side is pkmn/ai's
arena/arbiter.ts), torch-free: the C++ net (ataraxos/arena/actor.py) and
the numpy search (ataraxos/arena/search.py) with applemagpie's settings by
default (the joint per_type search, 3x3 types, 4 rounds, budget 1024,
depth 4, the sampler pool oak1500.txt).

    python -m ataraxos.arena.main --net net.bin --pool oak1500.txt [--threads N] ...

The protocol, as the arbiter runs it:
- we print `/ready` once everything is loaded (60 s, or the game is lost);
- stdin then carries `|utm|<packed team>` (ours, read past: the requests
  show the team), the battle's update chunks (our view, no room line),
  each `|request|<JSON>` as its own line after its chunk, and after each
  request that wants a choice, `|inactive|Time left: X sec this turn | ...`;
- we answer such a request with one line, a bare choice (`move 1`,
  `switch 3`, `default`), flushed; nothing else goes to stdout (logs go to
  stderr, whose last 4 KiB the account page shows on a failed smoke test);
- an `|error|` right after our choice means it was refused, and the
  arbiter sends the request again; the game ends with `|win|` or `|tie`
  and stdin closes.

Lines are buffered and fed to the Client (src/client.hpp, as
play/showdown.py does a room message) as one chunk when a request comes,
the chunk the sim sent before it. A request is answered once its
`|inactive|` line comes, or SETTLE seconds after it if none does. The
choice is heard by the bookkeeping (Client.choose) only when the next line
isn't an `|error|`. A request with one choice is answered without the net
(its tokens join the next decision, as in training). The search is skipped
when the turn's time left is under max(MIN_LEFT, 3 x the last search's
time), and a failed search plays the net's pick. If the client can't read
the battle, the rest of it is played with `default`."""

import argparse
import json
import os
import re
import select
import sys
import time
import traceback

import numpy as np

from ataraxos._ext import ataraxos_client
from ataraxos.arena.actor import Actor, load
from ataraxos.arena.search import Search, SearchConfig

SETTLE = 0.5  # seconds a request waits for its |inactive| line
MIN_LEFT = 10.0  # seconds left this turn under which the net plays alone
HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


class Lines:
    """stdin's lines, with a timeout (select on the raw descriptor, so no
    buffered data hides from it)."""

    def __init__(self, fd=0):
        self.fd, self.buf, self.eof = fd, b"", False

    def get(self, timeout=None):
        """The next line (no newline), None after `timeout` seconds; EOFError
        at the end."""
        end = None if timeout is None else time.monotonic() + timeout
        while b"\n" not in self.buf:
            if self.eof:
                if self.buf:
                    line, self.buf = self.buf, b""
                    return line.decode("utf-8", "replace")
                raise EOFError
            if end is not None:
                left = end - time.monotonic()
                if left <= 0 or not select.select([self.fd], [], [], left)[0]:
                    return None
            chunk = os.read(self.fd, 1 << 16)
            if chunk:
                self.buf += chunk
            else:
                self.eof = True
        line, _, self.buf = self.buf.partition(b"\n")
        return line.decode("utf-8", "replace")


def send(text):
    os.write(1, (text + "\n").encode())


def bare(cmd):
    """The Client's "/choose move 1|rqid" as the arena takes it: "move 1"."""
    if cmd.startswith("/choose "):
        cmd = cmd[len("/choose "):]
    return cmd.split("|", 1)[0]


class Game:
    def __init__(self, args, net, search):
        self.args, self.search = args, search
        self.client = ataraxos_client.Client()
        self.actor = Actor(net, 1, np.random.default_rng(args.seed))
        self.broken = False
        self.chosen = None  # the choice sent, heard by the bookkeeping once the next line isn't an |error|
        self.buffer = []  # the chunk so far
        self.request = None  # the request waiting for its |inactive| line
        self.left = None  # seconds left this turn, from |inactive|
        self.turn = 0
        self.decisions = 0
        self.last_search = 0.0

    def fail(self, where, e):
        if not self.broken:
            log(f"client failed in {where}, playing on with default:", repr(e))
            traceback.print_exc(file=sys.stderr)
        self.broken = True

    def flush(self):
        if self.buffer and not self.broken:
            try:
                self.client.feed("\n".join(self.buffer))
                unknown = self.client.unknown()
                if unknown:
                    log("unknown tags:", unknown)
            except Exception as e:  # noqa: BLE001
                self.fail("feed", e)
        self.buffer = []

    def line(self, line):
        """One stdin line; returns whether the game is over."""
        if self.chosen is not None:
            i, self.chosen = self.chosen, None
            if line.startswith("|error|"):
                log("refused:", line)
            elif not self.broken:
                try:
                    self.client.choose(i)
                except Exception as e:  # noqa: BLE001
                    self.fail("choose", e)
        if line.startswith("|utm|"):
            return False
        if line.startswith("|request|"):
            self.flush()
            self.request = line[len("|request|"):]
            try:
                wait = json.loads(self.request).get("wait") if self.request not in ("", "null") else True
            except ValueError:
                wait = False
            if wait:  # the bookkeeping still reads it
                self.answer(self.request)
                self.request = None
            return False
        if line.startswith("|inactive|"):
            m = re.match(r"\|inactive\|Time left: ([\d.]+) sec this turn", line)
            if m:
                self.left = float(m.group(1))
            if self.request is not None:
                self.answer(self.request)
                self.request = None
            return False
        if line.startswith("|error|"):
            log(line)
            return False
        if line.startswith("|turn|"):
            self.turn = int(line[6:]) if line[6:].isdigit() else self.turn
        self.buffer.append(line)
        if line.startswith("|win|") or line == "|tie" or line.startswith("|tie|"):
            self.flush()
            log("ended:", line, f"after {self.decisions} decisions, turn {self.turn}")
            return True
        return False

    def answer(self, text):
        """Decides a request (a wait is only read)."""
        if not self.broken:
            try:
                if not self.client.request(text):
                    return
                if self.client.n == 1:
                    return self.choose(0, self.client.choice(0))
                return self.decide()
            except Exception as e:  # noqa: BLE001
                self.fail("request", e)
        try:
            if json.loads(text).get("wait"):
                return
        except (ValueError, AttributeError):
            pass
        send("default")

    def decide(self):
        c = self.client
        d = c.take()
        obs = dict(d, battle=np.zeros(1, np.uint32), game=np.zeros(1, np.uint32), player=np.zeros(1, np.uint8))
        picks, logp, value = self.actor.act(obs)
        pick, how = int(picks[0]), "net"
        left = self.left
        if self.search is not None and (left is None or left >= max(MIN_LEFT, 3 * self.last_search)):
            tic = time.monotonic()
            try:
                sp, _, _ = self.search.run_client(c, self.actor, 0, obs, logp)
                pick, how = int(sp[0]), "search"
            except Exception as e:  # noqa: BLE001
                log("search failed, playing the net's pick:", repr(e))
                traceback.print_exc(file=sys.stderr)
            self.last_search = time.monotonic() - tic
        p = np.exp(value[0] - value[0].max())
        p /= p.sum()
        self.decisions += 1
        cmd = bare(c.choice(pick))
        log(f"turn {self.turn}: {cmd} ({how}, {self.last_search:.2f} s, P(win) {p[2]:.2f}, left {left})")
        self.choose(pick, None, cmd)

    def choose(self, i, text=None, cmd=None):
        self.chosen = i
        self.left = None
        send(cmd if cmd is not None else bare(text))


def warm_up(net):
    """One net call on a dummy decision, so the first real one isn't the
    first touch of the weights."""
    a = Actor(net, 1)
    obs = {"tokens": np.zeros((1, 11), np.int16), "offsets": np.array([0, 1], np.int64),
           "request": np.zeros((1, 107), np.int16), "state": np.zeros((1, 367), np.int16),
           "choices": np.zeros((1, 9), np.uint8), "n": np.ones(1, np.uint8), "battle": np.zeros(1, np.uint32),
           "game": np.zeros(1, np.uint32), "player": np.zeros(1, np.uint8)}
    a.act(obs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default=os.path.join(HERE, "net.bin"), help="export.py's weight file")
    ap.add_argument("--pool", default=os.path.join(HERE, "oak1500.txt"), help="the history sampler's pool")
    default_threads = min(8, int(os.environ.get("OMP_NUM_THREADS", "8") or 8))
    ap.add_argument("--threads", type=int, default=default_threads, help="the net's and the rollouts' threads")
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--joint", default="per_type", choices=["", "per_type", "per_pair"])
    ap.add_argument("--types", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    tic = time.monotonic()
    net = load(a.net, a.threads)
    search = None
    if not a.no_search:
        cfg = SearchConfig(budget=a.budget, reps=a.reps, depth=a.depth, joint=bool(a.joint),
                           joint_mode=a.joint or "per_type", types_a=a.types, types_b=a.types,
                           joint_budget=a.budget, rounds=a.rounds)
        seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "little") >> 2
        search = Search(net, [a.pool], cfg, threads=a.threads, seed=seed)
    warm_up(net)
    log(f"loaded in {time.monotonic() - tic:.1f} s ({a.threads} threads, search {'off' if search is None else a.joint or 'vanilla'})")
    send("/ready")
    game = Game(a, net, search)
    lines = Lines()
    try:
        while True:
            line = lines.get(SETTLE if game.request is not None else None)
            if line is None:  # no |inactive| after the request: answer anyway
                game.answer(game.request)
                game.request = None
                continue
            if game.line(line):
                break
    except EOFError:
        log("stdin closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
