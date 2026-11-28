// A local stand-in for pkmn.ai's arbiter (pkmn/ai arena/arbiter.ts,
// 2026-11-21), to drive ataraxos/arena/main.py end to end: Gen 1 OU in
// Showdown's simulator, each player a subprocess speaking the arena's
// stdin/stdout protocol, or "random" (in process: a uniform legal choice).
//
//   node tests/showdown/arena.cjs GAMES SEED TEAM_FILE 'P1 COMMAND' ['P2 COMMAND' | random]
//
// Per game, as the arbiter: start the players and wait for `/ready` (60 s);
// send `|utm|<packed team>`; forward each player-stream chunk plus "\n";
// after a request that wants a choice, `|inactive|Time left: ...`, then
// read one stdout line as the choice (`/info` lines are telemetry, skipped); on `[Invalid choice]` resend the last
// request (3 strikes lose); SIGSTOP a subprocess after its choice and
// SIGCONT it at its next request (the ladder's time-slicing); close stdin
// at the end. Teams are drawn from TEAM_FILE (Showdown export, "=== name
// ===" headers). Prints each game's result and the players' stderr to
// files beside SEED: arena-SEED-GAME-pN.err. The simulator: @pkmn/sim
// if require() finds it (NODE_PATH), else a built pokemon-showdown at SHOWDOWN
// (default ~/pokemon-showdown). Battles seed themselves.
'use strict';
const path = require('path');
const fs = require('fs');
const os = require('os');
const {spawn} = require('child_process');
// The simulator: @pkmn/sim (the arena's, 0.11.0) when require() finds it
// (NODE_PATH), else a built pokemon-showdown at SHOWDOWN.
let Teams, BattleStream, getPlayerStreams;
try {
  if (process.env.SHOWDOWN) throw new Error('SHOWDOWN set');
  const sim = require('@pkmn/sim');
  ({Teams} = sim);
  ({BattleStream, getPlayerStreams} = sim.BattleStreams);
} catch {
  const showdown = process.env.SHOWDOWN || path.join(os.homedir(), 'pokemon-showdown');
  ({Teams} = require(path.join(showdown, 'dist/sim')));
  ({BattleStream, getPlayerStreams} = require(path.join(showdown, 'dist/sim/battle-stream')));
}

const [games, seed, teamFile, cmd1, cmd2 = 'random'] = [Number(process.argv[2]), Number(process.argv[3]),
  process.argv[4], process.argv[5], process.argv[6]];
const pool = fs.readFileSync(teamFile, 'utf8').split(/^=== .* ===$/m).map(t => t.trim()).filter(Boolean);
// a small LCG for the teams and the random player (the sim seeds itself)
let state = (seed * 2654435761) >>> 0;
const prng = {random(n) { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state % n; }};
const INACTIVE = '|inactive|Time left: 150.0 sec this turn | 150.0 sec total | 0.0 sec grace';

function packed() {
  const t = Teams.import(pool[prng.random(pool.length)]);
  for (const s of t) s.ability = 'No Ability';
  return Teams.pack(t);
}

// A subprocess player: lines from its stdout, one at a time.
class Proc {
  constructor(command, errFile) {
    this.p = spawn('bash', ['-c', command], {stdio: ['pipe', 'pipe', fs.openSync(errFile, 'w')]});
    this.lines = [];
    this.waiters = [];
    this.exited = false;
    this.stdoutLines = 0;
    this.infoLines = 0;
    this.badInfo = 0;
    let buf = '';
    this.p.stdout.on('data', d => {
      buf += d.toString();
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        this.stdoutLines++;
        const body = line.trim();
        if (body.toLowerCase() === '/info' || body.toLowerCase().startsWith('/info ')) {
          // pkmn.ai's telemetry line (server/protocol.ts): not a choice
          this.infoLines++;
          try { JSON.parse(body.slice(6)); } catch { this.badInfo++; }
          continue;
        }
        if (this.waiters.length) this.waiters.shift()(line); else this.lines.push(line);
      }
    });
    this.p.stdin.on('error', () => undefined);
    this.p.on('exit', code => {
      this.exited = true;
      this.code = code;
      for (const w of this.waiters.splice(0)) w(undefined);
    });
  }
  line(ms) {
    if (this.lines.length) return Promise.resolve(this.lines.shift());
    if (this.exited) return Promise.resolve(undefined);
    return new Promise(resolve => {
      const t = setTimeout(() => { this.waiters = this.waiters.filter(w => w !== f); resolve(null); }, ms);
      const f = line => { clearTimeout(t); resolve(line); };
      this.waiters.push(f);
    });
  }
  write(s) { if (!this.exited && this.p.stdin.writable) this.p.stdin.write(s + '\n'); }
  stop() { if (!this.exited) try { process.kill(this.p.pid, 'SIGSTOP'); } catch {} }
  cont() { if (!this.exited) try { process.kill(this.p.pid, 'SIGCONT'); } catch {} }
}

function randomChoice(req) {
  if (req.forceSwitch) {
    const ok = req.side.pokemon.map((p, i) => i + 1).filter(i => i > 1 && !req.side.pokemon[i - 1].condition.endsWith(' fnt'));
    return ok.length ? `switch ${ok[prng.random(ok.length)]}` : 'pass';
  }
  const moves = req.active[0].moves.map((m, i) => [m, i + 1]).filter(([m]) => !m.disabled).map(([, i]) => `move ${i}`);
  const switches = req.active[0].trapped ? [] : req.side.pokemon.map((p, i) => i + 1)
    .filter(i => i > 1 && !req.side.pokemon[i - 1].condition.endsWith(' fnt')).map(i => `switch ${i}`);
  const all = [...moves, ...switches];
  return all.length ? all[prng.random(all.length)] : 'default';
}

async function game(g) {
  const cmds = [cmd1, cmd2];
  const teams = [packed(), packed()];
  const procs = cmds.map((c, i) => c === 'random' ? null : new Proc(c, `arena-${seed}-${g}-p${i + 1}.err`));
  const t0 = Date.now();
  for (const p of procs) {
    if (!p) continue;
    const ready = await p.line(60000);
    if (ready !== '/ready') throw new Error(`no /ready (${ready})`);
  }
  const readyMs = Date.now() - t0;
  procs.forEach((p, i) => p && p.write(`|utm|${teams[i]}`));
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  let ended = null;
  const stats = procs.map(() => ({decisions: 0, maxMs: 0, totalMs: 0, invalid: 0}));
  const sides = [streams.p1, streams.p2].map((s, i) => (async () => {
    let last = null, strikes = 0;
    for await (const chunk of s) {
      const p = procs[i];
      if (p) p.write(chunk);
      for (const l of chunk.split('\n')) {
        if (l.startsWith('|win|') || l === '|tie') ended = l;
        if (l.startsWith('|error|[Invalid choice]')) {
          stats[i].invalid++;
          if (++strikes >= 3) { ended = `|forfeit|p${i + 1} invalid`; void stream.write(`>forcewin p${2 - i}`); }
          else if (last && p) p.write(`|request|${last}`);
          if (last) await choose(i, last);
        }
      }
      const r = chunk.split('\n').find(l => l.startsWith('|request|'));
      if (!r) continue;
      const json = r.slice('|request|'.length);
      if (!json || json === 'null' || JSON.parse(json).wait) continue;
      last = json;
      strikes = 0;
      await choose(i, json);
    }
  })());
  async function choose(i, json) {
    const p = procs[i];
    let c;
    if (!p) {
      c = randomChoice(JSON.parse(json));
    } else {
      p.cont();
      p.write(INACTIVE);
      const tic = Date.now();
      c = await p.line(150000);
      const ms = Date.now() - tic;
      stats[i].decisions++;
      stats[i].maxMs = Math.max(stats[i].maxMs, ms);
      stats[i].totalMs += ms;
      p.stop();
      if (c == null || c === undefined) { ended = `|forfeit|p${i + 1} ${c === null ? 'timeout' : 'crash'}`; void stream.write(`>forcewin p${2 - i}`); return; }
    }
    void stream.write(`>p${i + 1} ${c}`);
  }
  void streams.omniscient.write(`>start {"formatid":"gen1ou"}\n` +
    `>player p1 {"name":"p1","team":"${teams[0]}"}\n>player p2 {"name":"p2","team":"${teams[1]}"}`);
  await Promise.race([Promise.all(sides), (async () => { while (!ended) await new Promise(r => setTimeout(r, 50)); })()]);
  for (const p of procs) if (p) { p.cont(); p.p.stdin.end(); }
  await new Promise(r => setTimeout(r, 1000));
  const codes = procs.map(p => p ? (p.exited ? p.code : 'running') : '-');
  for (const p of procs) if (p && !p.exited) p.p.kill('SIGKILL');
  const s = stats.map((x, i) => procs[i] ? `p${i + 1} ${x.decisions} decisions, mean ${(x.totalMs / Math.max(1, x.decisions)).toFixed(0)} ms, max ${x.maxMs} ms, invalid ${x.invalid}, stdout lines ${procs[i].stdoutLines}, /info ${procs[i].infoLines} (bad ${procs[i].badInfo})` : `p${i + 1} random`).join('; ');
  console.log(`game ${g}: ${ended} (ready ${readyMs} ms; exit ${codes.join(' ')}; ${s})`);
}

(async () => {
  for (let g = 0; g < games; g++) await game(g);
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
