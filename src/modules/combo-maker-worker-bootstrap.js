/* eslint no-eval: off, no-restricted-globals: off */

'use strict';

const ClueManager = require('./clue-manager');
const Clues       = require('./clue-types');
const ComboWorker = require('./combo-maker-worker');
const Stringify   = require('stringify-object');

let logging = true;

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
    loadClues(Clues.getByOptions(args), 10); // sum - 1; TODO: task item
    return ComboWorker.doWork(args);
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
