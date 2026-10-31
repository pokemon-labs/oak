// Plays Gen 1 OU in Showdown's simulator for tests/cpp/check_showdown.cpp:
// our teams (Showdown export files), uniform choices over each request's
// legal ones, and after each exchange the text every stream got and the
// simulator's truth. Needs a built Showdown (SHOWDOWN, default
// ~/oak-ataraxos/pokemon-showdown on the box).
//
//   node tests/showdown/play.cjs GAMES SEED TEAM_FILE > dump.txt
//
// The dump, line by line:
//   B SEED                      a battle starts
//   P SIDE NICK\tNICK...        a side's nicknames in party order
//   U CHANNEL                   an update's text for CHANNEL (0: the raw log,
//   ...lines...                 |split| triplets and all; 1, 2: that player's
//   .                           stream), up to the "." line
//   T SIDE NICK\tHP\tMAX\tSTATUS  the truth after that update
//   R SIDE JSON                 a side's request after that update
//   C SIDE CHOICE\tCHOICE...     the choices it could make (for check_client)
//   X SIDE CHOICE               the one it made
//   E RESULT                    the battle ended (or "cap" at 300 turns)
'use strict';
const path = require('path');
const fs = require('fs');
const os = require('os');
const showdown = process.env.SHOWDOWN || path.join(os.homedir(), 'oak-ataraxos', 'pokemon-showdown');
const {Battle, Teams, Dex, PRNG} = require(path.join(showdown, 'dist/sim'));
const {extractChannelMessages} = require(path.join(showdown, 'dist/sim/battle'));

const [games, seed, teamFile] = [Number(process.argv[2] || 100), Number(process.argv[3] || 1), process.argv[4]];
const dex = Dex.mod('gen1');
const teams = fs.readFileSync(teamFile, 'utf8').split(/^=== .* ===$/m).map(t => t.trim()).filter(Boolean)
  .map(t => Teams.import(t));
// Gen 1 has none; the validator, which a Battle skips, would set it
for (const team of teams) for (const p of team) p.ability = 'No Ability';
const rng = new PRNG([seed, seed + 1, seed + 2, seed + 3]);
const out = [];
const emit = s => out.push(s);
const flush = () => {
  if (out.length) process.stdout.write(out.join('\n') + '\n');
  out.length = 0;
};

// The side's legal choices, shuffled ([] when it waits).
function choices(side) {
  const r = side.activeRequest;
  if (!r || r.wait) return [];
  const bench = r.side.pokemon.map((p, i) => ({p, i}))
    .filter(({p}) => !p.active && !p.condition.endsWith(' fnt')).map(({i}) => `switch ${i + 1}`);
  let opts;
  if (r.forceSwitch) {
    opts = bench.length ? bench : ['pass'];
  } else {
    const a = r.active[0];
    opts = a.moves.map((m, i) => ({m, i})).filter(({m}) => !m.disabled).map(({i}) => `move ${i + 1}`);
    if (!a.trapped) opts.push(...bench);
  }
  for (let i = opts.length - 1; i > 0; i--) {
    const j = rng.random(i + 1);
    [opts[i], opts[j]] = [opts[j], opts[i]];
  }
  return opts;
}

for (let g = 0; g < games; g++) {
  const b = new Battle({formatid: 'gen1ou', seed: [seed, g, 7, 11]});
  const sets = [teams[rng.random(teams.length)], teams[rng.random(teams.length)]];
  emit(`B ${g}`);
  sets.forEach((team, s) => emit(`P ${s} ${team.map(p => p.name || dex.species.get(p.species).name).join('\t')}`));
  b.setPlayer('p1', {name: 'Alice', team: Teams.pack(sets[0])});
  b.setPlayer('p2', {name: 'Bob 2', team: Teams.pack(sets[1])});
  let at = 0;
  const update = () => {
    const raw = b.log.slice(at).join('\n');
    at = b.log.length;
    const ch = extractChannelMessages(raw, [1, 2]);
    emit('U 0');
    emit(raw);
    emit('.');
    for (const p of [1, 2]) {
      emit(`U ${p}`);
      for (const l of ch[p]) emit(l);
      emit('.');
    }
    b.sides.forEach((side, s) => {
      for (const p of side.pokemon) emit(`T ${s} ${p.name}\t${p.hp}\t${p.maxhp}\t${p.fainted ? '' : p.status}`);
    });
    if (!b.ended) b.sides.forEach((side, s) => side.activeRequest && emit(`R ${s} ${JSON.stringify(side.activeRequest)}`));
  };
  update();
  while (!b.ended && b.turn <= 300) {
    b.sides.forEach((side, s) => {
      const opts = choices(side);
      if (!opts.length) return;
      emit(`C ${s} ${[...opts].sort().join('\t')}`);
      // the first the simulator takes (a hidden trap refuses a switch)
      for (const c of opts) {
        if (b.choose(side.id, c)) {
          emit(`X ${s} ${c}`);
          break;
        }
      }
    });
    update();
  }
  emit(`E ${b.ended ? (b.winner || 'tie') : 'cap'}`);
  flush();
}
