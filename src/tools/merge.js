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
    ['f', 'from=BASEDIR',        'merge from: meta(m), synth(s), harmony(h)'],
    ['t', 'to=BASEDIR',          'merge to: synth(s), harmony(h), final(f)'],
    ['',  'save',                'save to results'],
    ['',  'force',               'force merge, ignoring warnings. USE CAUTION.'],

//  ['v', 'verbose',             'show logging'],
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

//

const ValidFromList = [Clues.META, Clues.SYNTH, Clues.HARMONY];
const ValidToList   = [Clues.SYNTH, Clues.HARMONY, Clues.FINAL];
const ValidFromToMap = {
    [Clues.META.baseDir]:    Clues.SYNTH,
    [Clues.SYNTH.baseDir]:   Clues.HARMONY,
    [Clues.HARMONY.baseDir]: Clues.FINAL
}

// name: e.g., 'meta' or 'm'
//
function getFrom (name) {
    return Clues.isValidBaseDirOption(name)
	? _.find(ValidFromList, ['baseDir', Clues.getByBaseDirOption(name).baseDir])
	: undefined;
}

// name: e.g., 'meta' or 'm'
//
function getTo (name) {
    return Clues.isValidBaseDirOption(name)
	? _.find(ValidToList, ['baseDir', Clues.getByBaseDirOption(name)])
	: undefined;
}

// from: e.g, Clues.META
function isValidFromTo (from, to) {
    return _.has(ValidFromToMap, from.name) && ValidFromToMap[from.name].name === to.name;
}

//

function loadClueLists (from, to) {
    return [ClueManager.loadClueList(from.MAX_CLUE_COUNT, { dir: from.name }),
	     ClueManager.loadClueList(1, { dir: to.name })];
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
    let [fromClueList, toClueList] = loadClueLists(from, to);
    let fromLength = fromClueList.length;
    let toLength = toClueList.length;
    fromClueList = fromClueList.sortedBySrc();
    toClueList = toClueList.sortedBySrc();
    // verify that length doesn't change when we sort
    Expect(fromClueList.length, 'from').to.equal(fromLength);
    Expect(toClueList.length, 'to').to.equal(toLength);
    console.log(`from(${fromClueList.length}), to(${toClueList.length})`);
    let [merged, warnings] = toClueList.mergeFrom(fromClueList);
    console.log(merged.toJSON());
    console.log(`length(${merged.length}), warnings(${warnings})`);
    Expect(merged.length, 'list length mismatch').to.equal(fromClueList.length);
    if (Opt.options.save) {
	if (warnings > 0 && !Opt.options.force) {
	    console.log('warnings found, save aborted. use --force to override');
	} else {
	    ClueManager.saveClueList(merged, 1, { dir: to.name });
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



