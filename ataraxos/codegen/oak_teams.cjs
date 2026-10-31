// oak's one-line teams ("species move move ...; species ...", one team a
// line, as ~/teams1500) -> Showdown's export format for Gen 1, the names as
// Showdown's dex spells them. Needs a built Showdown (SHOWDOWN, default
// ~/pokemon-showdown).
//
//   node codegen/oak_teams.cjs ~/teams1500 > teams/selfplay/oak1500.txt
const path = require('path');
const fs = require('fs');
const os = require('os');
const showdown = process.env.SHOWDOWN || path.join(os.homedir(), 'pokemon-showdown');
const {Dex} = require(path.join(showdown, 'dist/sim'));
const dex = Dex.mod('gen1');

const name = (table, id) => {
  const x = table.get(id);
  if (!x.exists) throw new Error(`unknown ${id}`);
  return x.name;
};

const lines = fs.readFileSync(process.argv[2], 'utf8').split('\n').filter(l => l.trim());
const out = [];
lines.forEach((line, i) => {
  out.push(`=== [gen1ou] team ${i + 1} ===\n`);
  for (const mon of line.split(';')) {
    const [species, ...moves] = mon.trim().split(/\s+/);
    out.push(name(dex.species, species));
    for (const m of moves) out.push(`- ${name(dex.moves, m)}`);
    out.push('');
  }
});
process.stdout.write(out.join('\n'));
