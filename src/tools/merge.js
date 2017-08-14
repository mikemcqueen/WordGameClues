//
// merge.js
//

'use strict';

const _            = require('lodash');
const ClueManager  = require('../clue-manager');
const Clues        = require('../clue-types');
const Debug        = require('debug')('merge');
const Dir          = require('node-dir');
const Expect       = require('should/as-function');
const Fs           = require('fs');
const Path         = require('path');
const Promise      = require('bluebird');
const Readlines    = require('n-readlines');
const Result       = require('./result-mod');

const Opt          = require('node-getopt')
   .create(_.concat(Clues.Options, [
       ['',  'save',                'save to results'],
       ['',  'force',               'force merge, ignoring warnings. USE CAUTION.'],
//     ['v', 'verbose',             'show logging'],
       ['h', 'help',                'this screen']
   ])).bindHelp().parseSystem();

//

function loadClueLists (from, to) {
    return [ClueManager.loadClueList(from.clueCount, { dir: from.baseDir }),
	    ClueManager.loadClueList(1, { dir: to.baseDir })];
}

//
//
//

function main () {
    const from = Clues.getByOptions(Opt.options);
    if (!from) throw new Error('oops');
    const to = Clues.cloneAsSynth(from);
    console.log(`from: ${from}[${from.baseDir}], to: ${to}[${to.baseDir}]`);
    if (Opt.options.verbose) {
	console.log('verbose: true');
    }
    let [fromClueList, toClueList] = loadClueLists(from, to);
    let fromLength = fromClueList.length;
    let toLength = toClueList.length;
    fromClueList = fromClueList.sortedBySrc();
    toClueList = toClueList.sortedBySrc();
    // verify that length doesn't change when we sort
    Expect(fromClueList.length, 'from').is.equal(fromLength);
    Expect(toClueList.length, 'to').is.equal(toLength);
    Debug(`from(${fromClueList.length}), to(${toClueList.length})`);
    let [merged, warnings] = toClueList.mergeFrom(fromClueList);
    console.log(merged.toJSON());
    console.log(`src: ${fromClueList.toJSON()}`);
    console.log(`merged: ${merged.toJSON()}`);
    console.log(`length(${merged.length}), warnings(${warnings})`);
    Expect(merged.length, 'list length mismatch').is.equal(fromClueList.length);
    if (Opt.options.save) {
	if (warnings > 0 && !Opt.options.force) {
	    console.log('warnings found, save aborted. use --force to override');
	} else {
	    ClueManager.saveClueList(fromClueList, from.clueCount, { dir: from.baseDir });
	    ClueManager.saveClueList(merged, 1, { dir: to.baseDir });
	}
    }
}

//

try {
    main();
}
catch(e) {
    console.error(e.stack);
}
finally {
}



