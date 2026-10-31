"""The control port's CLI (2026-10-26): watches and steers ataraxos/play/showdown.py
run with `--control-port`. Run it on the bot's machine; the port listens on
127.0.0.1 only. Each command prints the bot's JSON reply and exits nonzero
when it says ok false.

    python -m ataraxos.play.botctl events [--room R] [--type decision,chat] [--count N]
    python -m ataraxos.play.botctl state
    python -m ataraxos.play.botctl choose ROOM I [--rqid Q]
    python -m ataraxos.play.botctl auto ROOM [--temperature T] [--greedy] [--top-k K] [--ban 0,3] [--min-prob P]
    python -m ataraxos.play.botctl raw ROOM "/choose move 2"
    python -m ataraxos.play.botctl mode [ROOM] propose
    python -m ataraxos.play.botctl say ROOM TEXT...
    python -m ataraxos.play.botctl pm USER TEXT...
    python -m ataraxos.play.botctl accept USER | reject USER | forfeit ROOM | timer ROOM on|off
    python -m ataraxos.play.botctl schedule [--game N] [--pause | --resume]
    python -m ataraxos.play.botctl games [N | off]       # N more games, then quit; no N: the counts

`events` prints one JSON line per event (decisions still pending first),
filtered by room and type, and exits after `--count` of them (0: never), so
`events --type decision --count 1` waits for the next decision. The port is
`--port`, else $ATARAXOS_BOT_PORT, else 8765."""

import argparse
import json
import os
import socket
import sys


def connect(port):
    s = socket.create_connection(("127.0.0.1", port))
    return s, s.makefile("r", encoding="utf-8")


def command(port, c):
    s, f = connect(port)
    s.sendall((json.dumps(c) + "\n").encode())
    line = f.readline()
    s.close()
    print(line, end="")
    return 0 if line and json.loads(line).get("ok") else 1


def events(port, room, types, count):
    s, f = connect(port)
    s.sendall(b'{"cmd": "events"}\n')
    seen = 0
    for line in f:
        e = json.loads(line)
        if room and e.get("room") != room:
            continue
        if types and e.get("event") not in types:
            continue
        print(line, end="", flush=True)
        seen += 1
        if count and seen >= count:
            break
    s.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("ATARAXOS_BOT_PORT", 8765)))
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("events")
    e.add_argument("--room", default="")
    e.add_argument("--type", default="", help="comma-separated event types (none: all)")
    e.add_argument("--count", type=int, default=0)
    sub.add_parser("state")
    c = sub.add_parser("choose")
    c.add_argument("room")
    c.add_argument("i", type=int)
    c.add_argument("--rqid", type=int)
    a = sub.add_parser("auto")
    a.add_argument("room")
    a.add_argument("--rqid", type=int)
    a.add_argument("--temperature", type=float)
    a.add_argument("--greedy", action="store_true")
    a.add_argument("--top-k", type=int)
    a.add_argument("--ban", default="", help="comma-separated choice indices")
    a.add_argument("--min-prob", type=float)
    r = sub.add_parser("raw")
    r.add_argument("room")
    r.add_argument("text")
    m = sub.add_parser("mode")
    m.add_argument("args", nargs="+", help="[ROOM] MODE")
    for name, key in (("say", "room"), ("pm", "user")):
        p = sub.add_parser(name)
        p.add_argument(key)
        p.add_argument("text", nargs="+")
    for name, key in (("accept", "user"), ("reject", "user"), ("forfeit", "room")):
        sub.add_parser(name).add_argument(key)
    t = sub.add_parser("timer")
    t.add_argument("room")
    t.add_argument("on", choices=["on", "off"])
    sub.add_parser("games").add_argument("n", nargs="?", help="games still to play (0: quit when idle; off: no limit)")
    sc = sub.add_parser("schedule")
    sc.add_argument("--game", help="the next game's number")
    sc.add_argument("--pause", action="store_true")
    sc.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.cmd == "events":
        return events(args.port, args.room, set(filter(None, args.type.split(","))), args.count)
    c = {"cmd": args.cmd}
    if args.cmd == "choose":
        c.update(room=args.room, i=args.i, rqid=args.rqid)
    elif args.cmd == "auto":
        c.update(room=args.room, rqid=args.rqid)
        if args.temperature is not None:
            c["temperature"] = args.temperature
        if args.greedy:
            c["greedy"] = True
        if args.top_k is not None:
            c["top_k"] = args.top_k
        if args.ban:
            c["ban"] = [int(x) for x in args.ban.split(",")]
        if args.min_prob is not None:
            c["min_prob"] = args.min_prob
    elif args.cmd == "raw":
        c.update(room=args.room, text=args.text)
    elif args.cmd == "mode":
        if len(args.args) > 2:
            ap.error("mode takes [ROOM] MODE")
        c.update(mode=args.args[-1], room=args.args[0] if len(args.args) == 2 else None)
    elif args.cmd in ("say", "pm"):
        c.update(text=" ".join(args.text), **({"room": args.room} if args.cmd == "say" else {"user": args.user}))
    elif args.cmd in ("accept", "reject"):
        c["user"] = args.user
    elif args.cmd == "forfeit":
        c["room"] = args.room
    elif args.cmd == "timer":
        c.update(room=args.room, on=args.on == "on")
    elif args.cmd == "games":
        if args.n is not None:
            c["n"] = "off" if args.n == "off" else int(args.n)
    elif args.cmd == "schedule":
        if args.pause and args.resume:
            ap.error("--pause or --resume")
        if args.game is not None:
            c["game"] = args.game
        if args.pause or args.resume:
            c["paused"] = args.pause
    return command(args.port, c)


if __name__ == "__main__":
    sys.exit(main())
