//
// MERGE.JS
//

'use strict';

const _            = require('lodash');
const ClueManager  = require('../clue_manager');
const Clues        = require('../clue-types');
const Dir          = require('node-dir');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Path         = require('path');
const Promise      = require('bluebird');
const Readlines    = require('n-readlines');
const Result       = require('./result-mod');

const Opt          = require('node-getopt').create([
    ['f', 'from=NAME',           'merge from: meta(m), synth(s), harm(h)'],
    ['t', 'to=NAME',             'merge to: synth(s), harm(h), final(f)'],
    ['',  'force',               'force merge, ignoring warnings. USE CAUTION.'],
//    ['v', 'verbose',             'show logging'],
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

//

const ValidFromList = [ Clues.META, Clues.SYNTH, Clues.HARMONY ];
const ValidToList   = [ Clues.SYNTH, Clues.HARMONY, Clues.FINAL ];
const ValidFromToMap = {
    [Clues.META.name]:    Clues.SYNTH,
    [Clues.SYNTH.name]:   Clues.HARMONY,
    [Clues.HARMONY.name]: Clues.FINAL
}

// name: e.g., 'meta' or 'm'
//
function getFrom (name) {
    return Clues.isValidName(name) ? _.find(ValidFromList, [ 'name', Clues.getFullName(name) ]) : undefined;
}

// name: e.g., 'meta' or 'm'
//
function getTo (name) {
    return Clues.isValidName(name) ? _.find(ValidToList, [ 'name', Clues.getFullName(name) ]) : undefined;
}

// from: e.g, Clues.META
function isValidFromTo (from, to) {
    return _.has(ValidFromToMap, from.name) && ValidFromToMap[from.name].name === to.name;
}

//

function loadClueLists (from, to) {
    return [ ClueManager.loadClueList(from.MAX_CLUE_COUNT, { dir: from.name }),
	     ClueManager.loadClueList(1, { dir: to.name }) ];
}

//
//
//

function main () {
    console.log(`from: ${Opt.options.from}, to: ${Opt.options.to}`);
    const from = getFrom(Opt.options.from);
    const to = getTo(Opt.options.to);
    Expect(from, '--from NAME must be one of: meta(m), synth(s), harmony(h)').to.exist;
    Expect(to, '--to NAME must be one of: synth(s), harmony(h), final(f)').to.exist;
    console.log(`${Opt.options.from}[${from.name}], to: ${Opt.options.to}[${to.name}]`);
    Expect(isValidFromTo(from, to), `Can't merge from ${from.name} to ${to.name}`).to.be.true;
    if (Opt.options.verbose) {
	console.log('verbose: true');
    }

    let [ fromClueList, toClueList ] = loadClueLists(from, to);
    let fromLength = fromClueList.length;
    let toLength = toClueList.length;
    fromClueList = fromClueList.sortedBySrc();
    toClueList = toClueList.sortedBySrc();
    Expect(fromClueList.length === fromLength, 'from').to.be.true;
    Expect(toClueList.length === toLength, 'to').to.be.true;
    console.log(`from(${fromClueList.length}), to(${toClueList.length})`);

    let options = { copySrc: true };
    let [merged, warnings] = toClueList.mergeFrom(fromClueList, options);
    console.log(merged.toJSON());
    console.log(`length(${merged.length}), warnings(${warnings})`);
    if (warnings === 0 || Opt.options.force) {
	ClueManager.saveClueList(merged, 1, { dir: to.name });
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



