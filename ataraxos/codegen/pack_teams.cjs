// A team pool for ataraxos/play/showdown.py (`--teams`) from a Showdown
// export file: a line per team, its name, a tab, and the team packed by
// Showdown's Teams.pack.
//
//   node codegen/pack_teams.cjs teams/selfplay/oak150.txt > pool.tsv
'use strict';
const path = require('path');
const fs = require('fs');
const os = require('os');
const showdown = process.env.SHOWDOWN || path.join(os.homedir(), 'oak-ataraxos', 'pokemon-showdown');
const {Teams} = require(path.join(showdown, 'dist/sim'));

const text = fs.readFileSync(process.argv[2], 'utf8');
const parts = text.split(/^(=== .* ===)$/m);
let n = 0;
for (let i = 1; i < parts.length; i += 2) {
  const name = parts[i].replace(/^=== (\[[^\]]*\] )?/, '').replace(/ ===$/, '').replace(/\t/g, ' ');
  const team = Teams.import(parts[i + 1].trim());
  if (!team || !team.length) continue;
  process.stdout.write(`${name || 'team ' + n}\t${Teams.pack(team)}\n`);
  n++;
}
