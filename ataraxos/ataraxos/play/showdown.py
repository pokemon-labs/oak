"""Plays Showdown with a checkpoint's policy (draft, 2026-10-26, the user's
ask): logs in, accepts every gen1ou challenge (or one user's, with
`--accept-from`) with a team drawn at random from a pool, rejects other
formats, and plays each battle; several at once. With `--schedule` it
challenges one user instead (below). With `--games N` (2026-10-31, the
user's ask) it stops taking battles after N and exits 0 once the last
ends; box restart loops stop on that exit.

    python -m ataraxos.play.showdown --ckpt runs/cs2b/ckpt2900.pt --teams pool.tsv \
        --login login.txt [--games-dir games]

`--teams` has a line per team, its name, a tab, and the packed team.

Pacing and the ladder (2026-10-31, the user's ask, first for the account
"glegle merchant"; each off by default): each request is answered a random
`--delay MIN,MAX` seconds after it arrives (at least SETTLE after the
room's last message), or `--play-faster-wait` seconds when the last
decision's P(win) was under `--play-faster-threshold` (the request's own
isn't known until it's answered); a battle room is left `--leave-after`
seconds after its game ends; the battle timer is turned on when
`--idle-timer` seconds pass after one of our choices with no new request
(the foe stalling), unless the room said it's on; with `--ladder`, it
queues for the gen1ou ladder with a random pool team, one game at a time,
`--ladder-delays` minutes after each game ends (the list in turn,
repeating), and on login. glegle merchant's settings:

    python -m ataraxos.play.showdown --ckpt CKPT --teams pool.tsv --login login.txt --ladder \
        --delay 2,5 --leave-after 5 --idle-timer 20 --play-faster-threshold 0.1
`--login` is a file of two lines, the name and the password. `--server
local` plays on a local Showdown server (`--url`, by default
tests/showdown/local_server.cjs's), named `--name` with no login server
(the server's noguestsecurity): for testing the driver end to end.
`--server arena` plays on pkmn.bot instead (https://pkmn.bot/llms.txt): logged in as
ARENA_BOT_NAME with the token ARENA_BOT_TOKEN from the environment, a random
`--teams` team sent on login and after each battle, battles paired by the
arena (no challenges, no schedule). The policy is
the checkpoint's EMA weights (`--weights`), sampled from its softmax at
`--temperature` (1, the actor's own sample, by default; 0 is greedy), then
with every probability under `--min-prob`/k (k the legal choices; 1 prunes
whatever is under uniform; 0, the default, prunes nothing) set to 0 and the
rest renormalized (the most probable choice always stays), on
the CPU, through causal_actor with the net's own window and the client's
bookkeeping state. With `--search-pool` (2026-11-01), every decision
with two or more choices is searched (ataraxos/search.py: `--search-budget`
rollouts, `--search-reps` per world and choice, `--search-depth` turns, the
history sampler's pool from that Showdown export file, on
`--search-device`) and the pick is drawn from the search's policy; the
jsonl keeps the net's log-probs and q too. A failed search plays the net's
pick.

Schedule (2026-10-28, the user's ask, for a bot-vs-bot match):
`--schedule FILE --challenge USER` challenges USER to one game at a time,
in the file's order, each with its own team. FILE has a line per game: the
game's number, a tab, a name, a tab, the packed team. The next game's
number is kept in FILE.next, so a restart resumes there; it advances when
a battle against USER that we challenged ends. A challenge goes when no
battle against USER is running and none of ours is pending, and again
CHALLENGE_RETRY seconds after one that was declined or failed. USER's own
challenges are rejected (their team wouldn't be the schedule's). The
control command `schedule` shows it, sets the next game {game} and pauses
it {paused}.

The battle room's text and requests go through src/client.hpp
(ataraxos_client), which gives the tokens, request fields and choices training
gives. A request with one choice is answered without the net and its tokens
join the next decision, as in training. Showdown sends a log chunk and then
the requests (sim/battle.ts sendUpdates), but the live server is unchecked,
so a request is answered SETTLE seconds after it arrives, every room message
fed by then. If the client can't read the battle, the rest of it is played
with "/choose default".

Each battle's raw room text goes to GAMES_DIR/ROOM.log and its decisions
(choices, probabilities, value) to GAMES_DIR/ROOM.jsonl, to replay failures
through tests/cpp/check_client.

Control (2026-10-26, the user's ask): with `--control-port`, a local TCP
port (127.0.0.1 only) through which a harness watches and steers the bot,
JSON lines both ways; ataraxos/play/botctl.py is its CLI. A connection that sends
{"cmd": "events"} gets every event from then on (the decisions still
pending first); any other command gets one reply, {"ok": ...}.

- Events (`event`): `battle` (a battle starts), `decision` (a request
  waits for a choice: its choices with labels, the policy's probabilities,
  pick and P(win), the request JSON, the room's lines since the last
  decision, the seconds until the fallback), `committed` (what was sent and
  by whom: `policy`, `harness` or `deadline`), `superseded` (a pending
  decision dropped for a newer request), `chat` (the battle room's chat),
  `join` / `leave`, `inactive` (the timer's lines), `error` (|error|),
  `end`, `pm`, `challenge`.
- Modes, per battle (`--mode` the default for new ones): `auto` plays as
  before; `propose` waits for the harness and sends the policy's pick after
  `--deadline` seconds; `manual` waits until the timer (when on) is about to
  run out. Under either, the fallback comes TIMER_MARGIN seconds before the
  timer's own limit, read from `|inactive|Time left`. A request with one
  choice is always answered at once.
- Commands (`cmd`): `choose` {room, i} sends choice i; `auto` {room,
  temperature, greedy, top_k, ban, min_prob} re-picks from the policy's
  log-probabilities (no options: its own sample); `raw` {room, text} sends
  any line to the room (a /choose commits the pending decision); `mode`
  {room or none for the default, mode}; `say` {room, text}; `pm` {user,
  text}; `accept` / `reject` {user}; `forfeit` {room}; `timer` {room, on};
  `state`; `schedule` {game, paused}; `games` {n}: n more games from now (0:
  quit once no battle is open; "off": no limit; none: just the counts). `say` and `pm` refuse text starting with / or ! (use `raw`).

The client's bookkeeping hears each choice the bot sends (Client.choose),
as BattleData's does in training; a harness's `raw` /choose isn't heard,
so the state may be off after one.

A decision line in ROOM.jsonl records its `source`, so games steered by a
harness aren't read as the policy's. Training doesn't read these logs.

Chat (`--chat`, the user's ask, 2026-10-27): each decision the net made is
said in the battle room, its turn, the most probable choices with their
probabilities, the choice sent and P(win). A line waits until the next
request or the end, so the foe never reads a choice before making their
own. A probability shows as "net/used" when the one the pick was drawn from
(after temperature, top_k, bans) differs from the net's. `--chat redacted`
(2026-10-28) names a choice only if the log had shown it when the decision
was made (a Pokémon that has appeared, a move the active one has used;
client.revealed); the others are just "move" or "switch". `full` names
every choice, `off` (the default) says nothing."""

import argparse
import asyncio
import json
import os
import random
import re
import time
import traceback
import urllib.parse
import urllib.request

import numpy as np
import torch
import websockets

from ataraxos._ext import ataraxos_client  # noqa: E402
from ataraxos.nets.causal_actor import make_actor  # noqa: E402
from ataraxos.search import Search, SearchConfig  # noqa: E402
from ataraxos.eval.agents import load_policy  # noqa: E402

SERVER = "wss://sim3.psim.us/showdown/websocket"
LOGIN = "https://play.pokemonshowdown.com/api/login"
# pkmn.bot (https://pkmn.bot/llms.txt): Showdown's protocol, bots paired for rated
# battles by the arena; the name and token come from ARENA_BOT_NAME, ARENA_BOT_TOKEN
ARENA_SERVER = "wss://pkmn.bot/showdown/websocket"
ARENA_RECONNECT = 5
LOCAL_SERVER = "ws://127.0.0.1:8123/showdown/websocket"
FORMAT = "gen1ou"
SETTLE = 0.5  # seconds a request waits for more of the room's messages
RECONNECT = 30  # seconds before reconnecting
TIMER_MARGIN = 10  # seconds before the timer's limit that the fallback goes
QUIT_AFTER = 2  # seconds after the last game ends (and --leave-after) before the connection closes (--games)
TICK = 15  # seconds between checks whether the schedule's next challenge is due
CHALLENGE_RETRY = 60  # seconds after a challenge before another goes
MODES = ("auto", "propose", "manual")


def to_id(s):
    return "".join(c for c in s.lower() if c.isalnum())


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def user_name(identity):
    """A chat identity without its rank character (" ", "+", "@", ...)."""
    return identity[1:] if identity and not identity[0].isalnum() else identity


def label(req, cmd):
    """'move 2 (Earthquake)' for "/choose move 2|rqid", from the request."""
    what = cmd.split("|")[0][len("/choose "):]
    kind, _, k = what.partition(" ")
    try:
        if kind == "move":
            return f"{what} ({req['active'][0]['moves'][int(k) - 1]['move']})"
        if kind == "switch":
            return f"{what} ({req['side']['pokemon'][int(k) - 1]['details'].split(',')[0]})"
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    return what


def short(lbl):
    """'Earthquake' for 'move 2 (Earthquake)', 'switch Skarmory' for 'switch 3 (Skarmory)'."""
    kind, _, rest = lbl.partition(" ")
    m = re.search(r"\((.*)\)", rest)
    if not m:
        return lbl
    return m.group(1) if kind == "move" else f"{kind} {m.group(1)}"


def chat_line(turn, p, i, redact=False, top=4):
    """A decision for the room's chat: its turn, the `top` most probable
    choices (net/used when they differ), the one sent and P(win), at most
    300 characters. With `redact`, a choice not yet shown is just "move"
    or "switch"."""
    probs = np.exp(np.array(p["logp"], np.float64))
    used = np.array(p["used"], np.float64) if p.get("used") is not None else probs
    order = np.argsort(-probs)[:top]

    def name(k):
        if redact and not p["revealed"][k]:
            return p["labels"][k].split(" ")[0]
        return short(p["labels"][k])

    def prob(k):
        a, b = f"{probs[k]:.0%}", f"{used[k]:.0%}"
        return a if a == b else f"{a}/{b}"

    parts = [f"{name(k)} {prob(k)}" for k in order]
    sent = name(i) if i is not None else "?"
    win = f"{p['p_win']:.0%}" if p["p_win"] is not None else "?"
    return (f"T{turn}: " + ", ".join(parts) + f" | sent {sent} | P(win) {win}")[:300]


def used_probs(logp, temperature=1.0, greedy=False, top_k=0, ban=(), min_prob=0.0):
    """The probabilities a pick is drawn from, given the policy's
    log-probabilities over the legal choices [n]: one-hot on the best when
    greedy, else the softmax at `temperature` over the `top_k` best (0: all),
    the choices in `ban` left out, then those under `min_prob` / n set to 0
    and the rest renormalized (the best always kept; 0: none pruned)."""
    lp = np.array(logp, np.float64)
    for i in ban:
        if 0 <= i < len(lp):
            lp[i] = -np.inf
    if not np.isfinite(lp).any():
        raise ValueError("every choice banned")
    if greedy or temperature <= 0:
        w = np.zeros(len(lp))
        w[int(np.argmax(lp))] = 1.0
        return w
    if top_k:
        lp[np.argsort(-lp)[top_k:]] = -np.inf
    w = np.exp((lp - lp.max()) / temperature)
    w /= w.sum()
    if min_prob > 0:
        w[(w < min_prob / len(w)) & (w < w.max())] = 0.0
        w /= w.sum()
    return w


def repick(logp, rng, temperature=1.0, greedy=False, top_k=0, ban=(), min_prob=0.0):
    """A choice drawn from used_probs, and those probabilities."""
    w = used_probs(logp, temperature, greedy, top_k, ban, min_prob)
    return int(rng.choice(len(w), p=w)), w


def login(name, password, challstr):
    data = urllib.parse.urlencode({"name": name, "pass": password, "challstr": challstr}).encode()
    # Cloudflare refuses urllib's default User-Agent (error 1010)
    request = urllib.request.Request(LOGIN, data=data, headers={"User-Agent": "ataraxos-bot/0.1"})
    with urllib.request.urlopen(request, timeout=30) as r:
        text = r.read().decode()
    j = json.loads(text[1:] if text.startswith("]") else text)
    assertion = j.get("assertion") or ""
    if not assertion or assertion.startswith(";;"):
        raise RuntimeError(f"login failed: {text[:300]}")
    return assertion


def p_win(value):
    """P(win), a draw worth 1/2, from the (loss, draw, win) logits."""
    p = torch.softmax(value.float(), -1)
    return float(p[2] + p[1] / 2)


class Battle:
    def __init__(self, bot, room):
        self.bot, self.room = bot, room
        self.client = ataraxos_client.Client()
        self.actor = make_actor(bot.policy, 1, "cpu", amp=False)
        self.game = bot.games
        self.request = None  # the latest request's JSON, not yet answered
        self.due = 0.0  # when it's answered (monotonic), --delay after it came
        self.last_p_win = None  # the last net decision's P(win), for the next request's delay
        self.idle = None  # the --idle-timer handle after our choice
        self.timer_on = False  # the room said the battle timer is on
        self.timer = None
        self.broken = False
        self.decisions = 0
        self.mode = bot.mode
        self.title, self.turn = "", 0
        self.foe = None  # the other player's ID
        self.sched = None  # the schedule's game index, for a game we challenged to
        self.pending = None  # a decision waiting for the harness
        self.deadline = None  # its fallback's timer handle
        self.recent = []  # the room's lines since the last decision
        self.time_left = None  # (seconds this turn, when read), from |inactive|
        self.chat = None  # the last decision's chat line, said once the turn moves on
        self.raw = open(os.path.join(bot.args.games_dir, room + ".log"), "a")
        self.out = open(os.path.join(bot.args.games_dir, room + ".jsonl"), "a")

    def fail(self, where, e):
        if not self.broken:
            say(self.room, f"client failed in {where}, playing on with /choose default:", repr(e))
            self.out.write(json.dumps({"error": where, "what": repr(e), "trace": traceback.format_exc()}) + "\n")
            self.out.flush()
        self.broken = True

    def message(self, text):
        self.raw.write(text + "\n\x00\n")  # messages separated by a NUL line
        self.raw.flush()
        lines = text.split("\n")
        for line in lines:
            if line.startswith("|request|"):
                self.stop_idle()
                self.request = line[len("|request|"):]
                self.due = time.monotonic() + random.uniform(*self.delay())
                continue
            if line:
                self.recent.append(line)
            p = line.split("|")
            tag = p[1] if len(p) > 1 else ""
            if tag == "error":
                say(self.room, line)
                self.bot.emit("error", room=self.room, text="|".join(p[2:]))
            elif tag == "turn" and len(p) > 2:
                self.turn = int(p[2]) if p[2].isdigit() else self.turn
            elif tag == "c" and len(p) > 3:
                self.bot.emit("chat", room=self.room, user=user_name(p[2]), text="|".join(p[3:]))
            elif tag == "c:" and len(p) > 4:
                self.bot.emit("chat", room=self.room, user=user_name(p[3]), text="|".join(p[4:]))
            elif tag in ("j", "J", "l", "L") and len(p) > 2:
                self.bot.emit("join" if tag in "jJ" else "leave", room=self.room, user=user_name(p[2]))
            elif tag == "inactive":
                what = "|".join(p[2:])
                self.bot.emit("inactive", room=self.room, text=what)
                if re.search(r"timer is (now )?ON", what, re.I):
                    self.timer_on = True
                elif re.search(r"timer is (now )?OFF", what, re.I):
                    self.timer_on = False
                m = re.match(r"Time left: (\d+) sec this turn", what)
                if m:
                    self.time_left = (int(m.group(1)), time.monotonic())
                    if self.pending:
                        self.arm()
        if not self.broken:
            try:
                self.client.feed(text)
                unknown = self.client.unknown()
                if unknown:
                    say(self.room, "unknown tags:", unknown)
            except Exception as e:  # noqa: BLE001
                self.fail("feed", e)
        if self.request is not None:
            if self.timer:
                self.timer.cancel()
            wait = max(self.due - time.monotonic(), SETTLE)
            self.timer = asyncio.get_running_loop().call_later(wait, self.decide)
        for line in lines:
            if line.startswith("|win|") or line == "|tie" or line.startswith("|tie|"):
                won = line.startswith("|win|") and to_id(line[5:]) == to_id(self.bot.args.name)
                result = "win" if won else "tie" if line.startswith("|tie") else "loss"
                self.say_chat()
                self.stop_idle()
                say(self.room, "ended:", result, f"after {self.decisions} decisions")
                self.out.write(json.dumps({"result": result}) + "\n")
                self.out.flush()
                self.drop()
                self.bot.emit("end", room=self.room, result=result)
                self.bot.ended(self.room)

    def delay(self):
        """The next answer's delay range: --play-faster-wait once the last
        decision's P(win) is under --play-faster-threshold, else --delay."""
        a = self.bot.args
        if self.last_p_win is not None and self.last_p_win < a.play_faster_threshold:
            return a.play_faster_wait
        return a.delay

    def send_choice(self, cmd):
        """Sends a choice and arms the idle timer: no new request within
        --idle-timer seconds (0: off) turns the battle timer on."""
        self.bot.send(self.room, cmd)
        self.stop_idle()
        if self.bot.args.idle_timer > 0:
            self.idle = asyncio.get_running_loop().call_later(self.bot.args.idle_timer, self.idle_timer)

    def stop_idle(self):
        if self.idle:
            self.idle.cancel()
            self.idle = None

    def idle_timer(self):
        self.idle = None
        if not self.timer_on and self.room in self.bot.battles:
            say(self.room, f"no request {self.bot.args.idle_timer:g} s after our choice: timer on")
            self.bot.send(self.room, "/timer on")

    def say_chat(self):
        if self.chat is not None:
            self.bot.send(self.room, self.chat)
            self.chat = None

    def decide(self):
        self.timer = None
        text, self.request = self.request, None
        if text is None:
            return
        self.say_chat()
        rqid = None
        try:
            req = json.loads(text) if text != "null" else None
        except ValueError:
            req = None
        if isinstance(req, dict):
            rqid = req.get("rqid")
        else:
            req = {}
        self.drop(superseded=True)
        if not self.broken:
            try:
                if not self.client.request(text):
                    return  # a wait
                if self.client.n == 1:  # answered without the net, as in training
                    self.send_choice(self.client.choose(0))
                    return
                d = self.client.take()
                obs = dict(d, battle=np.zeros(1, np.uint32), game=np.array([self.game], np.uint32),
                           player=np.zeros(1, np.uint8))
                with torch.no_grad():
                    picks, logp, value = self.actor.act(obs)
                n = self.client.n
                cmds = [self.client.choice(k) for k in range(n)]
                net_lp, q, searched = logp[0, :n].tolist(), None, None
                if self.bot.search is not None:
                    try:
                        tic = time.monotonic()
                        sp, qs, slp = self.bot.search.run_client(self.client, self.actor, 0, obs, logp)
                        picks, logp, q = sp, slp, [round(float(x), 4) for x in qs[0]]
                        searched = round(time.monotonic() - tic, 2)
                    except Exception as e:  # noqa: BLE001
                        say(self.room, "search failed, playing the net's pick:", repr(e))
                        self.out.write(json.dumps({"error": "search", "what": repr(e),
                                                   "trace": traceback.format_exc()}) + "\n")
                lp = logp[0, :n].tolist()
                t, floor = self.bot.args.temperature, self.bot.args.min_prob
                if t == 1 and floor <= 0:
                    pick, used = int(picks[0]), used_probs(lp)
                else:
                    pick, used = repick(lp, self.bot.rng, t, min_prob=floor)
                self.last_p_win = p_win(value[0])
                self.propose(dict(rqid=rqid, cmds=cmds, labels=[label(req, c) for c in cmds], logp=lp,
                                  used=used.tolist(), revealed=[self.client.revealed(k) for k in range(n)],
                                  pick=pick, p_win=round(self.last_p_win, 4),
                                  tokens=int(d["offsets"][1]), request=req,
                                  **({"net_logp": net_lp, "q": q, "search_s": searched} if q is not None else {})))
                return
            except Exception as e:  # noqa: BLE001
                self.fail("request", e)
        if req.get("wait"):
            return
        fallback = f"/choose default|{rqid}" if rqid is not None else "/choose default"
        self.propose(dict(rqid=rqid, cmds=[], labels=[], logp=None, used=None, revealed=None, pick=None,
                          p_win=None, tokens=None,
                          request=req, fallback=fallback))

    def propose(self, p):
        """Plays the policy's pick (auto), or offers the decision to the
        harness until its deadline."""
        self.pending = p
        if self.mode == "auto":
            return self.commit("policy")
        self.arm()
        self.bot.emit("decision", **self.event())

    def event(self):
        p = self.pending
        if "log" not in p:  # the lines since the last decision, taken once
            p["log"] = self.take_recent()
        return dict(room=self.room, rqid=p["rqid"], turn=self.turn, mode=self.mode,
                    choices=p["labels"], probs=None if p["logp"] is None else [round(float(np.exp(x)), 4) for x in p["logp"]],
                    pick=p["pick"], p_win=p["p_win"], broken=p["logp"] is None,
                    deadline=self.seconds(), request=p["request"], log=p["log"])

    def take_recent(self):
        lines, self.recent = self.recent, []
        return lines

    def seconds(self):
        """Seconds until the pending decision's fallback (None: no limit)."""
        limits = []
        if self.mode == "propose":
            limits.append(self.bot.args.deadline)
        if self.time_left is not None:
            left, when = self.time_left
            limits.append(max(2.0, left - (time.monotonic() - when) - TIMER_MARGIN))
        return round(min(limits), 1) if limits else None

    def arm(self):
        if self.deadline:
            self.deadline.cancel()
        t = self.seconds()
        self.deadline = None if t is None else asyncio.get_running_loop().call_later(t, self.commit, "deadline")

    def drop(self, superseded=False):
        if self.deadline:
            self.deadline.cancel()
            self.deadline = None
        if self.pending is not None and superseded:
            self.bot.emit("superseded", room=self.room, rqid=self.pending["rqid"])
            self.recent = self.pending.get("log", []) + self.recent  # offered again with the next one
        self.pending = None

    def commit(self, source, i=None, cmd=None, used=None):
        """Sends choice `i` (None: the policy's pick, or the fallback when the
        client can't read the battle) or the text `cmd`, and logs it. `used`
        replaces the probabilities the pick was drawn from (a harness's
        repick)."""
        p = self.pending
        if p is None:
            return
        if used is not None:
            p["used"] = used.tolist()
        if cmd is None:
            if i is None:
                i = p["pick"]
            cmd = p["cmds"][i] if i is not None else p["fallback"]
            if i is not None and not self.broken:
                self.client.choose(i)  # the bookkeeping hears the choice sent, as in training
        self.drop()
        self.time_left = None  # this turn's; the next turn's comes with it
        if p["logp"] is not None:
            self.decisions += 1
        self.out.write(json.dumps({"rqid": p["rqid"], "source": source, "pick": i, "policy_pick": p["pick"],
                                   "cmd": cmd, "choices": p["labels"],
                                   "probs": None if p["logp"] is None else [round(float(np.exp(x)), 4) for x in p["logp"]],
                                   "used": None if p["used"] is None else [round(x, 4) for x in p["used"]],
                                   "p_win": p["p_win"], "tokens": p["tokens"],
                                   **({"net_probs": [round(float(np.exp(x)), 4) for x in p["net_logp"]], "q": p["q"],
                                       "search_s": p["search_s"]} if p.get("q") is not None else {})}) + "\n")
        self.out.flush()
        if "log" not in p:  # never offered: its lines are still in `recent`
            self.take_recent()
        self.send_choice(cmd)
        if self.bot.args.chat != "off" and p["logp"] is not None and p["labels"]:
            self.chat = chat_line(self.turn, p, i, redact=self.bot.args.chat == "redacted",
                                  top=self.bot.args.chat_top)
        self.bot.emit("committed", room=self.room, rqid=p["rqid"], source=source, pick=i, cmd=cmd)

    def set_mode(self, mode):
        self.mode = mode
        if self.pending is None:
            return
        if mode == "auto":
            self.commit("policy")
        else:
            self.arm()


class Bot:
    def __init__(self, args):
        self.args = args
        torch.set_num_threads(args.threads)
        self.policy = load_policy(args.ckpt, "cpu", args.weights)
        self.search = None
        if args.search_pool:
            cfg = SearchConfig(budget=args.search_budget, reps=args.search_reps, depth=args.search_depth)
            self.search = Search(load_policy(args.ckpt, args.search_device, args.weights), [args.search_pool],
                                 args.search_device, cfg, threads=args.threads)
        self.pool = [line.rstrip("\n").split("\t") for line in open(args.teams) if "\t" in line]
        self.rng = np.random.default_rng()
        self.battles = {}
        self.team_for = {}  # challenger's ID -> the team name we accepted with
        self.games = 0
        self.games_left = args.games or None  # games still to play (None: no limit); see quit_if_done
        self.done = False  # the last game is over: the connection closes and main() returns
        self.ws = None
        self.mode = args.mode  # for new battles
        self.listeners = set()  # control connections that asked for events
        self.named = False  # logged in: challenges are accepted only then
        self.schedule = [line.rstrip("\n").split("\t", 2) for line in open(args.schedule)
                         if line.count("\t") >= 2] if args.schedule else []
        self.sched_i = 0  # the next game's index in the schedule
        if self.schedule and os.path.exists(args.schedule + ".next"):
            self.sched_i = self.game_index(open(args.schedule + ".next").read().strip())
        self.sched_paused = False
        self.challenge_game = None  # the index the challenge in flight is for
        self.challenge_sent = None  # when it went (monotonic)
        self.challenge_to = None  # the server's: our pending challenge, if any
        self.arena = args.server == "arena"
        self.arena_team = "?"  # the pool team last sent with /utm (arena)
        self.searching = False  # a ladder search is open (|updatesearch|)
        self.ladder_team = "?"  # the pool team the last search went with
        self.ladder_i = 0  # the next requeue's place in --ladder-delays
        self.requeue = None  # the pending requeue's handle

    def game_index(self, game):
        for i, row in enumerate(self.schedule):
            if row[0] == str(game):
                return i
        if str(game) == "done":
            return len(self.schedule)
        raise ValueError(f"no game {game!r} in the schedule")

    def sched_state(self):
        i = self.sched_i
        return dict(next=self.schedule[i][0] if i < len(self.schedule) else "done",
                    team=self.schedule[i][1] if i < len(self.schedule) else None,
                    games=len(self.schedule), paused=self.sched_paused, opponent=self.args.challenge,
                    in_flight=None if self.challenge_game is None else self.schedule[self.challenge_game][0])

    def sched_save(self):
        with open(self.args.schedule + ".next", "w") as f:
            f.write(self.sched_state()["next"] + "\n")

    def tick(self):
        """Sends the schedule's next challenge when it's due."""
        if (not self.schedule or not self.named or self.sched_paused or self.sched_i >= len(self.schedule)
                or not self.room_for_more()):
            return
        opp = to_id(self.args.challenge)
        if self.challenge_to or any(b.foe == opp for b in self.battles.values()):
            return
        if self.challenge_sent is not None and time.monotonic() - self.challenge_sent < CHALLENGE_RETRY:
            return
        game, name, packed = self.schedule[self.sched_i]
        self.team_for[opp] = f"game {game}: {name}"
        say("schedule: challenging", self.args.challenge, f"to game {game} of {len(self.schedule)} with", name)
        self.send("", f"/utm {packed}")
        self.send("", f"/challenge {self.args.challenge}, {FORMAT}")
        self.challenge_game, self.challenge_sent = self.sched_i, time.monotonic()
        self.emit("schedule", **self.sched_state())

    async def ticker(self):
        while True:
            await asyncio.sleep(TICK)
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                say("error in the schedule:", traceback.format_exc())

    def emit(self, event, **fields):
        if not self.listeners:
            return
        line = (json.dumps(dict(event=event, time=round(time.time(), 2), **fields)) + "\n").encode()
        for w in list(self.listeners):
            try:
                w.write(line)
            except Exception:  # noqa: BLE001
                self.listeners.discard(w)

    def send(self, room, text):
        shown = "/trn ..." if text.startswith("/trn") else text[:120]
        say(room or "global", ">", shown)
        asyncio.get_running_loop().create_task(self.ws.send(f"{room}|{text}"))

    def accept(self, user):
        # a fresh account on a datacenter address may not challenge, but may
        # accept; /utm sets the team for the next battle
        name, packed = self.pool[self.rng.integers(len(self.pool))]
        self.team_for[to_id(user)] = name
        say("accepting", user, "with", name)
        self.send("", f"/utm {packed}")
        self.send("", f"/accept {user}")

    def send_team(self):
        """The arena pairs bots that have sent a team: a random pool team, on
        login and after each battle."""
        self.arena_team, packed = self.pool[self.rng.integers(len(self.pool))]
        say("team for the next battle:", self.arena_team)
        self.send("", f"/utm {packed}")

    def challenged(self, user, fmt):
        if not self.named:
            return
        if self.arena:  # the arena starts the battle itself; /accept and /reject do nothing
            self.emit("challenge", user=user, format=fmt, action="arena")
            return
        if self.schedule and to_id(user) == to_id(self.args.challenge):
            self.emit("challenge", user=user, format=fmt, action="rejected")
            self.send("", f"/reject {user}")  # we challenge them, with the schedule's team
            return
        if not self.room_for_more():
            self.emit("challenge", user=user, format=fmt, action="rejected")
            self.send("", f"/reject {user}")  # --games / `games` reached
            return
        if self.args.accept_from and to_id(user) != to_id(self.args.accept_from):
            self.emit("challenge", user=user, format=fmt, action="ignored")
            return
        if to_id(fmt) == FORMAT:
            self.emit("challenge", user=user, format=fmt, action="accepted")
            self.accept(user)
        else:
            self.emit("challenge", user=user, format=fmt, action="rejected")
            self.send("", f"/reject {user}")

    def queue(self):
        """Queues for the ladder with a random pool team, unless a search
        is open, a battle is running or it's not logged in."""
        self.requeue = None
        if not self.args.ladder or not self.named or self.searching or self.battles or not self.room_for_more():
            return
        self.ladder_team, packed = self.pool[self.rng.integers(len(self.pool))]
        say("ladder: searching", FORMAT, "with", self.ladder_team)
        self.send("", f"/utm {packed}")
        self.send("", f"/search {FORMAT}")

    def queue_later(self, seconds):
        if self.requeue:
            self.requeue.cancel()
        self.requeue = asyncio.get_running_loop().call_later(seconds, self.queue)

    def ended(self, room):
        b = self.battles.pop(room, None)
        self.games += 1
        if self.games_left is not None:
            self.games_left = max(0, self.games_left - 1)
        if b is not None and b.sched is not None:
            self.sched_i = max(self.sched_i, b.sched + 1)
            self.sched_save()
            say("schedule: game", self.schedule[b.sched][0], "over; next", self.sched_state()["next"])
            asyncio.get_running_loop().call_later(3, self.tick)
        if self.arena:
            if self.room_for_more():
                self.send_team()
            self.limit_reached()
            return
        if self.args.leave_after > 0:
            asyncio.get_running_loop().call_later(self.args.leave_after, self.send, "", f"/leave {room}")
        else:
            self.send("", f"/leave {room}")
        self.limit_reached()
        if self.args.ladder and self.room_for_more():
            delays = self.args.ladder_delays
            minutes = delays[self.ladder_i % len(delays)]
            self.ladder_i += 1
            say(f"ladder: next search in {minutes:g} min")
            self.queue_later(minutes * 60)

    def global_message(self, line):
        p = line.split("|")
        tag = p[1] if len(p) > 1 else ""
        if tag == "challstr" and self.args.server == "local":
            self.send("", f"/trn {self.args.name},0,")
        elif tag == "challstr" and self.arena:
            self.send("", f"/trn {self.args.name},0,{self.args.password}")
        elif tag == "challstr":
            challstr = "|".join(p[2:])
            try:
                assertion = login(self.args.name, self.args.password, challstr)
            except Exception as e:  # noqa: BLE001
                # a guest can't chat, so it doesn't stay one: reconnect and try again
                # (the user's, 2026-10-27; the login server fails under DDoS)
                say("login failed, reconnecting:", repr(e)[:200])
                asyncio.get_running_loop().create_task(self.ws.close())
                return
            self.send("", f"/trn {self.args.name},0,{assertion}")
        elif tag == "updateuser":
            say("logged in as", repr(p[2]), "named" if p[3] == "1" else "guest")
            was, self.named = self.named, p[3] == "1"
            if self.arena:
                if self.named and not was:
                    self.send_team()
                return
            # the avatar once per login (|updateuser|NAME|NAMED|AVATAR|SETTINGS)
            if self.named and not was and self.args.avatar and (len(p) < 5 or p[4] != self.args.avatar):
                self.send("", f"/avatar {self.args.avatar}")
            if self.named and not was and self.args.ladder:
                self.queue_later(2)
        elif tag == "updatesearch":
            data = json.loads(p[2]) if len(p) > 2 and p[2] else {}
            self.searching = bool(data.get("searching"))
        elif tag == "updatechallenges":
            data = json.loads(p[2])
            self.challenge_to = data.get("challengeTo")
            for user, fmt in (data.get("challengesFrom") or {}).items():
                self.challenged(user, fmt)
        elif tag == "popup":
            say("popup:", "|".join(p[2:])[:300])
            if "already challenging" in line:
                self.challenge_to = True  # one of ours is still pending
            # a refused search (a rejected team, a blocked address): try again in a minute
            if self.args.ladder and not self.searching and not self.battles and self.requeue is None:
                self.queue_later(60)
        elif tag == "pm":
            say("pm:", "|".join(p[2:])[:300])
            # our own challenge, as the server reports it: "|pm| US|THEM|/challenge FORMAT|...",
            # and "/challenge" with nothing after once it's accepted, declined or cancelled
            if len(p) > 4 and to_id(p[2]) == to_id(self.args.name) and p[4].startswith("/challenge"):
                self.challenge_to = p[4][len("/challenge"):].strip() or None
            if len(p) > 4 and not p[4].startswith("/"):
                self.emit("pm", user=user_name(p[2]), to=user_name(p[3]), text="|".join(p[4:]))
            # a challenge: "|pm| SENDER|RECEIVER|/challenge FORMAT|..."
            if len(p) > 4 and p[4].startswith("/challenge ") and to_id(p[2]) != to_id(self.args.name):
                self.challenged(p[2].strip(), p[4][len("/challenge "):])

    def room_message(self, room, body):
        if room not in self.battles:
            if "|init|battle" not in body:
                return  # a battle we left
            title = next((l[len("|title|"):] for l in body.split("\n") if l.startswith("|title|")), "")
            foes = [u for u in (to_id(x) for x in title.split(" vs. ")) if u != to_id(self.args.name)]
            team = self.team_for.pop(foes[0], "?") if foes else "?"
            if team == "?" and self.args.ladder:
                team = self.ladder_team
                self.searching = False
            if self.arena:
                team = self.arena_team
            say("battle", room, title, "| team:", team)
            self.battles[room] = Battle(self, room)
            self.battles[room].title = title
            self.battles[room].foe = foes[0] if foes else None
            if self.challenge_game is not None and foes and foes[0] == to_id(self.args.challenge):
                self.battles[room].sched = self.challenge_game  # the challenge we sent
                say("schedule: game", self.schedule[self.challenge_game][0], "is", room)
                self.challenge_game = self.challenge_sent = self.challenge_to = None  # accepted
            self.battles[room].out.write(json.dumps({"title": title, "team": team, "ckpt": self.args.ckpt}) + "\n")
            self.emit("battle", room=room, title=title, team=team, foe=foes[0] if foes else None,
                      mode=self.battles[room].mode)
        self.battles[room].message(body)

    def battle(self, c):
        b = self.battles.get(c.get("room", ""))
        if b is None:
            raise ValueError(f"no battle {c.get('room')!r}")
        return b

    @staticmethod
    def chat_text(c):
        text = str(c.get("text", "")).replace("\n", " ").strip()
        if not text or text[0] in "/!":
            raise ValueError("empty, or starts with / or ! (use raw)")
        return text

    def control(self, c):
        """One command from the control port; returns the reply's fields."""
        cmd = c.get("cmd")
        if cmd == "state":
            return dict(mode=self.mode, played=self.games, left=self.games_left, battles={
                r: dict(title=b.title, turn=b.turn, mode=b.mode, decisions=b.decisions,
                        pending=b.event() if b.pending else None) for r, b in self.battles.items()})
        if cmd == "mode":
            if c.get("mode") not in MODES:
                raise ValueError(f"mode is one of {MODES}")
            if c.get("room"):
                self.battle(c).set_mode(c["mode"])
            else:
                self.mode = c["mode"]
            return {}
        if cmd in ("choose", "auto"):
            b = self.battle(c)
            p = b.pending
            if p is None:
                raise ValueError("no decision pending")
            if c.get("rqid") is not None and c["rqid"] != p["rqid"]:
                raise ValueError(f"the pending decision is rqid {p['rqid']}")
            used = None
            if cmd == "choose":
                i = int(c["i"])
                if not 0 <= i < len(p["cmds"]):
                    raise ValueError(f"choice {i} of {len(p['cmds'])}")
            elif p["logp"] is None:
                i = None  # the fallback
            elif any(k in c for k in ("temperature", "greedy", "top_k", "ban", "min_prob")):
                i, used = repick(p["logp"], self.rng, float(c.get("temperature", 1.0)),
                                 bool(c.get("greedy", False)), int(c.get("top_k", 0)),
                                 [int(x) for x in c.get("ban", [])], float(c.get("min_prob", 0.0)))
            else:
                i = p["pick"]
            b.commit("harness", i, used=used)
            return dict(pick=i)
        if cmd == "raw":
            room, text = c.get("room", ""), str(c.get("text", ""))
            if "\n" in text:
                raise ValueError("one line")
            b = self.battles.get(room)
            if b is not None and b.pending is not None and text.startswith("/choose"):
                b.commit("harness", cmd=text)
            else:
                self.send(room, text)
            return {}
        if cmd == "say":
            self.battle(c)
            self.send(c["room"], self.chat_text(c))
            return {}
        if cmd == "pm":
            self.send("", f"/msg {c['user']}, {self.chat_text(c)}")
            return {}
        if cmd == "accept":
            self.accept(c["user"])
            return {}
        if cmd == "reject":
            self.send("", f"/reject {c['user']}")
            return {}
        if cmd == "forfeit":
            self.battle(c)
            self.send(c["room"], "/forfeit")
            return {}
        if cmd == "timer":
            self.battle(c)
            self.send(c["room"], "/timer on" if c.get("on", True) else "/timer off")
            return {}
        if cmd == "games":
            if c.get("n") == "off":
                self.games_left = None
            elif c.get("n") is not None:
                self.games_left = max(0, int(c["n"]))
                if not self.room_for_more() and self.searching:
                    self.send("", "/cancelsearch")
                self.limit_reached()
            return dict(played=self.games, left=self.games_left)
        if cmd == "schedule":
            if not self.schedule:
                raise ValueError("no --schedule")
            if c.get("game") is not None:
                self.sched_i = self.game_index(c["game"])
                self.sched_save()
            if c.get("paused") is not None:
                self.sched_paused = bool(c["paused"])
            return self.sched_state()
        raise ValueError(f"unknown cmd {cmd!r}")

    def room_for_more(self):
        """Whether another battle may start under --games / `games`."""
        return self.games_left is None or self.games_left > len(self.battles)

    def limit_reached(self):
        """Called after the count changes: quits when it's spent and no
        battle is open."""
        if self.games_left == 0 and not self.battles and not self.done:
            say(f"games: {self.games} played, none left; quitting")
            self.done = True
            # after the room's /leave (and the arena's last messages) go out
            asyncio.get_running_loop().call_later(QUIT_AFTER + self.args.leave_after, lambda: asyncio.ensure_future(self.ws.close()))

    async def serve(self, reader, writer):
        try:
            async for line in reader:
                try:
                    c = json.loads(line)
                    if c.get("cmd") == "events":
                        for b in self.battles.values():
                            if b.pending is not None:
                                writer.write((json.dumps(dict(event="decision", replay=True, **b.event())) + "\n").encode())
                        self.listeners.add(writer)
                        continue
                    reply = dict(ok=True, **self.control(c))
                except Exception as e:  # noqa: BLE001
                    reply = dict(ok=False, error=repr(e))
                writer.write((json.dumps(reply) + "\n").encode())
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self.listeners.discard(writer)
            writer.close()

    async def run(self):
        server = None
        if self.args.control_port:
            server = await asyncio.start_server(self.serve, "127.0.0.1", self.args.control_port, reuse_address=True)
        try:
            await self.connect()
        finally:
            self.listeners.clear()
            if server:
                server.close()

    async def connect(self):
        url = ARENA_SERVER if self.arena else self.args.url or LOCAL_SERVER if self.args.server == "local" else SERVER
        async with websockets.connect(url, max_size=None) as ws:
            self.ws = ws
            ticker = asyncio.get_running_loop().create_task(self.ticker())
            try:
                await self.receive(ws)
            finally:
                ticker.cancel()

    async def receive(self, ws):
        async for text in ws:
            try:
                if text.startswith(">"):
                    room, _, body = text[1:].partition("\n")
                    if room.startswith("battle-"):
                        self.room_message(room, body)
                    continue
                for line in text.split("\n"):
                    if line.startswith("|"):
                        self.global_message(line)
            except Exception:  # noqa: BLE001
                say("error handling a message:", traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "net"])
    ap.add_argument("--teams", required=True, help="the team pool: lines of NAME, a tab, a packed team")
    ap.add_argument("--accept-from", default="", help="accept only this user's challenges (none: anyone's)")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (0: greedy)")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="after the temperature, probabilities under this / the number of choices go to 0 "
                         "and the rest renormalize (1: under uniform; 0: off)")
    ap.add_argument("--schedule", default="", help="games to challenge --challenge to: NUMBER, NAME, packed team")
    ap.add_argument("--challenge", default="", help="the user the schedule's games are against")
    ap.add_argument("--login", default="", help="a file: the name, then the password (showdown)")
    ap.add_argument("--server", default="showdown", choices=["showdown", "arena", "local"],
                    help="arena: pkmn.bot, as ARENA_BOT_NAME with the token ARENA_BOT_TOKEN; "
                         "local: a local server with no login (--url, --name)")
    ap.add_argument("--url", default="", help="the websocket for --server local (default LOCAL_SERVER)")
    ap.add_argument("--name", default="", help="the name for --server local")
    ap.add_argument("--avatar", default="", help="the avatar set on login (empty: leave it)")
    ap.add_argument("--ladder", action="store_true",
                    help="queue for the gen1ou ladder, one game at a time, --ladder-delays minutes apart")
    ap.add_argument("--ladder-delays", type=lambda v: [float(x) for x in v.split(",")], default=[0.1],
                    help="minutes before each requeue after a game, comma-separated, used in turn and repeating "
                         "(e.g. 0.5,0.5,5,0.5,10,0.5,5)")
    ap.add_argument("--delay", type=lambda v: tuple(float(x) for x in v.split(",")), default=(0.0, 0.0),
                    help="seconds, MIN,MAX uniform, from a request's arrival to its answer (at least SETTLE)")
    ap.add_argument("--leave-after", type=float, default=0.0, help="seconds after a game ends before its room is left")
    ap.add_argument("--idle-timer", type=float, default=0.0,
                    help="seconds after our choice with no new request before /timer on (0: off)")
    ap.add_argument("--play-faster-threshold", type=float, default=0.0,
                    help="answer within --play-faster-wait once the last decision's P(win) is under this (0: off)")
    ap.add_argument("--play-faster-wait", type=lambda v: tuple(float(x) for x in v.split(",")), default=(1.0, 2.0),
                    help="seconds, MIN,MAX uniform, before each answer while P(win) is under the threshold")
    ap.add_argument("--games", type=int, default=0,
                    help="play this many games, then exit 0 once no battle is open (0: no limit); "
                         "the control command `games` changes it")
    ap.add_argument("--games-dir", default="games")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chat", default="off", choices=["off", "full", "redacted"],
                    help="say each decision's policy and P(win) in the room; redacted: unshown choices "
                         "as just move or switch")
    ap.add_argument("--chat-top", type=int, default=4, help="the most probable choices the chat line names")
    ap.add_argument("--control-port", type=int, default=0, help="the harness's port on 127.0.0.1 (0: none)")
    ap.add_argument("--mode", default="auto", choices=MODES, help="how new battles decide")
    ap.add_argument("--search-pool", default="",
                    help="search (ataraxos/search.py) with this team file as the history sampler's pool (none: no search)")
    ap.add_argument("--search-budget", type=int, default=128, help="rollouts per decision")
    ap.add_argument("--search-reps", type=int, default=2, help="rollouts per (world, choice)")
    ap.add_argument("--search-depth", type=int, default=10, help="turns before the leaf")
    ap.add_argument("--search-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deadline", type=float, default=60.0,
                    help="seconds a proposed decision waits before the policy's pick goes")
    args = ap.parse_args()
    if bool(args.schedule) != bool(args.challenge):
        ap.error("--schedule and --challenge go together")
    if args.server == "arena":
        if args.schedule:
            ap.error("the arena pairs bots itself: no --schedule")
        args.name, args.password = os.environ["ARENA_BOT_NAME"], os.environ["ARENA_BOT_TOKEN"]
    elif args.server == "local":
        if not args.name:
            ap.error("--name is needed for local")
        args.password = ""
    elif args.login:
        args.name, args.password = open(args.login).read().split("\n")[:2]
    else:
        ap.error("--login is needed for showdown")
    os.makedirs(args.games_dir, exist_ok=True)
    bot = Bot(args)
    say(f"policy {args.ckpt} ({args.weights}), temperature {args.temperature}, min prob {args.min_prob}; "
        + (f"search: pool {args.search_pool}, budget {args.search_budget}, reps {args.search_reps}, "
           f"depth {args.search_depth} on {args.search_device}; " if args.search_pool else "")
        + f"{len(bot.pool)} teams; accepting {FORMAT} challenges from {args.accept_from or 'anyone'}; mode {args.mode}"
        + (f"; schedule {args.schedule} against {args.challenge}, next game {bot.sched_state()['next']}"
           if bot.schedule else "")
        + (f", control on 127.0.0.1:{args.control_port}" if args.control_port else "")
        + (f"; stopping after {args.games} games" if args.games else ""))
    while True:  # reconnect; battles in progress are lost to the server's timer
        try:
            asyncio.run(bot.run())
            if bot.done:
                return
            say("disconnected")
        except Exception:  # noqa: BLE001
            say("connection failed:", traceback.format_exc())
        bot.battles.clear()
        bot.named = False
        bot.challenge_game = bot.challenge_sent = bot.challenge_to = None
        time.sleep(ARENA_RECONNECT if args.server != "showdown" else RECONNECT)


if __name__ == "__main__":
    main()
