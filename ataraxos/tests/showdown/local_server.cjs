// A local Showdown server for testing ataraxos/play/showdown.py end to end
// (`--server local`): Showdown's own server (SHOWDOWN, default
// ~/oak-ataraxos/pokemon-showdown on the box) on 127.0.0.1:PORT (8123),
// with noguestsecurity so a bot takes its name with no login server. The
// settings are set on the loaded config, config/config.js is left alone.
//
//   node tests/showdown/local_server.cjs [PORT]
'use strict';
const path = require('path');
const os = require('os');
const showdown = process.env.SHOWDOWN || path.join(os.homedir(), 'oak-ataraxos', 'pokemon-showdown');
const port = Number(process.argv[2] || 8123);
process.chdir(showdown);
const config = require(path.join(showdown, 'config/config.js'));
Object.assign(config, {port, bindaddress: '127.0.0.1', noguestsecurity: true, repl: false, workers: 1});
require('module')._load(path.join(showdown, 'dist/server/index.js'), module, true);
