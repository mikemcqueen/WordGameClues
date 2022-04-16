/* eslint no-eval: off, no-restricted-globals: off */

'use strict';

const ClueManager = require('../dist/modules/clue-manager');
const ComboMaker  = require('../dist/modules/combo-maker');

const Clues       = require('./clue-types');
const Stringify   = require('stringify-object');

let logging = false;
let iter = 0;

//

let loadClues = (clues, max) =>  {
    if (logging) console.log('loading all clues...');
    ClueManager.loadAllClues({
        clues,
	max,
	ignoreLoadErrors: false,
        validateAll: true,
	fast: true
    });
    if (logging) console.log('done.');
    return true;
};

const MILLY = 1000000n;
let last = 0n;

let log = (prefix) => {
    let stamp = process.hrtime.bigint();
    if (last) {
	let since = (stamp - last) / MILLY;
	console.error(`${prefix} ${since}ms`); // pass/use index in args
    }
    else console.error(prefix);
    last = stamp;
};

let bootstrap = (args) => {
    if (logging) console.log(`bootstrapping sum(${args.sum}) max(${args.max}) args: ${Stringify(args)}`);

    if (0) {
	let onesources = [];
	if (args.useSourcesList && args.useSourcesList.length) onesources.push(args.useSourcesList[0]);
	console.log(`bootstrap sources: ${Stringify(onesources)}`);
	console.log(`JSON.stringify bootstrap sources: ${JSON.stringify(onesources)}`);
    }
    ClueManager.logging = false;
    if (!ClueManager.loaded) {
	loadClues(Clues.getByOptions(args), args.load_max);
    }
    let combos = ComboMaker.makeCombosForSum(args.sum, args.max, args);
    return combos;
};

let entrypoint = (args) => {
    return bootstrap(args);
};

process.once('message', code => {
    let d = JSON.parse(code).data;
    //console.log(d);
    eval(d);
});

module.exports = {
    entrypoint
};
