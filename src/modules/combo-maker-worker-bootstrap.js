/* eslint no-eval: off, no-restricted-globals: off */

'use strict';

const ClueManager = require('./clue-manager');
const Clues       = require('./clue-types');
const ComboMaker  = require('./combo-maker');
const Stringify   = require('stringify-object');

let logging = true;
let iter = 0;

//

let loadClues = (clues, max) =>  {
    if (logging) console.log('loading all clues...');
    ClueManager.loadAllClues({
        clues,
	max,
	ignoreLoadErrors: false,
        validateAll: true,
    });
    if (logging) console.log('done.');
    return true;
};

let bootstrap = (args) => {
    if (logging) console.log(`bootstrapping sum(${args.sum}) max(${args.max}) args: ${Stringify(args)}`);
    ClueManager.logging = false;
    if (!ClueManager.loaded) {
	loadClues(Clues.getByOptions(args), 15); // sum - 1; TODO: task item
    }
    const MILLY = 1000000n;
    let start = process.hrtime.bigint();
    let combos = ComboMaker.makeCombos(args);
    let end = process.hrtime.bigint();
    let ms = (end - start) / MILLY;
    console.error(`. ${ms}ms`); // pass/use index in args
    //process.stderr.write('.');
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
