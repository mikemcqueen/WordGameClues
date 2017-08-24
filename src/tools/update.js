//
// update.js
//

'use strict';

const _            = require('lodash');
const ClueManager  = require('../clue-manager');
const Clues        = require('../clue-types');
const Dir          = require('node-dir');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Path         = require('path');
const Promise      = require('bluebird');
const Readlines    = require('n-readlines');
const SearchResult = require('../modules/search-result');

const Opt          = require('node-getopt')
    .create(_.concat(Clues.Options, [
	['d', 'dir=NAME',            'directory name'],
	['',  'save',                'save clues'],
	['v', 'verbose',             'show logging'],
	['h', 'help',                'this screen']
    ])).bindHelp().parseSystem();
/*
    ])).bindHelp(
	"Usage: node filter <options> [wordListFile]\n\n" +
	    "[[OPTIONS]]\n"
    )
 */

//

const Start =   Symbol('start');
const Src =     Symbol('src');
const Url =     Symbol('url');
const Clue =    Symbol('clue');
const Maybe =   Symbol('maybe');
const Known =   Symbol('known');
const Done =    Symbol('done');

const ClueSuffix =    'c';
const RejectSuffix =  'x';
const ValidSuffixes = [ ClueSuffix, RejectSuffix ];

// state machine
const SM = {
    [Start] : { next: [ Src ] },
    [Src]   : { next: [ Url,   Clue,  Maybe, Known/*, Src should test this*/ ], func: processSrc },
    [Url]   : { next: [ Url,   Clue,  Maybe, Known, Src ], func: processUrl },
    [Clue]  : { next: [ Url,   Clue,  Maybe, Known, Src ], func: processClue },
    [Maybe] : { next: [ Url,   Clue,  Maybe, Known, Src ], func: processMaybe },
    [Known] : { next: [ Known, Src ],                      func: processKnown }
}

//
function getLineState (line) {
    Expect(line.length).to.be.at.least(1);
    let state;
    if (line[0] === SearchResult.SRC_PREFIX) state = Src;
    if (line[0] === SearchResult.MAYBE_PREFIX) state = Maybe;
    if (line[0] === SearchResult.KNOWN_PREFIX) state = Known;
    // all of the above get first char sliced off
    if (state) {
	line = line.slice(1);
    } else if (line.startsWith('http')) {
	state = Url;
    } else {
	// else, probably a clue
	state = Clue;
    }
    return [line, state];
}

// process a line beginning with '@' that represents
// a comma-separated list of words

function processSrc (line, args, options) {
    Expect(line).to.be.a('string');
    Expect(args.dir).to.be.a('string');

    // process ,x suffix
    let rejected = false;
    let index = line.lastIndexOf(',');
    let rejectSuffix = false;
    if (index > -1) {
	if (rejectSuffix = (line.slice(index + 1, line.length).trim() === RejectSuffix)) {
	    line = line.slice(0, index);
	}
    }
    console.log(`src: ${line}`);
    let nameList = line.split(',');
    Expect(nameList.length).to.be.at.least(2);
    if (rejectSuffix) {
	if (ClueManager.addReject(nameList)) {
	    args.count.rejectClues += 1;
	    args.count.rejectCountSet.add(nameList.length);
	}
    }
    if (rejectSuffix || ClueManager.isRejectSource(line)) {
	return {
	    count     : args.count,
	    nextState : Src
	};
    }

    let dir = `${SearchResult.DIR}${args.dir}`;
    let path = `${dir}/${SearchResult.makeFilteredFilename(nameList)}`;
    let content;
    try {
	content = Fs.readFileSync(path, 'utf8');
    } catch(err) {
	// file not existing is OK.
	if (options.verbose) {
	    console.log('filtered file not found');
	}
    }
    let flags = {};
    let filteredUrls;
    if (!_.isUndefined(content)) {
	filteredUrls = JSON.parse(content);
	console.log(`loaded: ${path}`);
	// maybe were added later, some files don't have it
	if (!filteredUrls.maybeUrls) filteredUrls.maybeUrls = [];
    }
    else {
	filteredUrls = {
	    knownUrls:  [],
	    rejectUrls: [],
	    maybeUrls: []
	};
    }
    return { 
	count:           args.count,
	nameList:        nameList,
	filteredUrlPath: path,
	filteredUrls,
	flags
    };
}

//

function getUrlSuffix (line, options) {
    let url = line;
    let suffix;

    let index = line.lastIndexOf(',');
    if (index > -1) {
	let maybeSuffix = line.slice(index + 1, line.lenghth).trim();
	if (maybeSuffix.length === 1) {
	    url = line.slice(0, index).trim();
	    suffix = maybeSuffix;
	    if (options.verbose) {
		console.log(`suffix: ${suffix}`);
	    }
	    Expect(ValidSuffixes.includes(suffix), `bad url suffix, ${line}`).to.be.true;
	}
    }
    return [url, suffix];
}

//

function processUrl (line, args, options) {
    Expect(line).to.be.a('string');
    Expect(args.filteredUrls).to.be.an('object');

    let [url, suffix] = getUrlSuffix(line, options);
    args.url = url;
    if (suffix === ClueSuffix) {
	args.flags.clue = true;
    }
    else if (suffix === RejectSuffix) {
	args.flags.reject = true;
    }
    else {
	Expect(suffix).to.be.undefined;
    }
    return args;
}

//

function addClues (countList, name, src) { // , add = ClueManager.addClue) {
    Expect(countList).to.be.an('array');
    Expect(name).to.be.a('string');
    Expect(src).to.be.a('string');

    let updatedCountList = [];
    countList.forEach(count => {
	if (ClueManager.addClue(count, { name, src }, false, true)) { // save = false, nothrow = true
	    updatedCountList.push(count);
	}
    });
    return updatedCountList;
}

//

function getNameNote (line) {
    let name;
    let note;
    let firstComma = line.indexOf(',');
    if (firstComma === -1) {
	name = line.trim();
    } else {
	name = line.slice(0, firstComma);
	if (line.length > firstComma + 1) {
	    note = line.slice(firstComma + 1, line.length);
	}
    }
    return [name, note];
}

//

function processClue (line, args, options) {
    Expect(line).to.be.a('string');

    let [name, note] = getNameNote(line);
    // we're about to update known clues. double-sanity check.
    Expect(ClueManager.isRejectSource(args.nameList)).to.be.false;
    let countList = ClueManager.getCountList(args.nameList);
    console.log(`countList: ${_.isEmpty(countList) ? "empty" : countList}`) ;
    countList = addClues(countList, name, args.nameList.toString());
    if (!_.isEmpty(countList)) {
	args.count.knownClues += 1;
	countList.forEach(count => args.count.knownCountSet.add(count));
	console.log(`added clue, ${name} : ${args.nameList} - ${note} : [${countList}]`);
    }
    args.flags.clue = true;
    return args;
}

//

function processMaybe (line, args, options) {
    Expect(line).to.be.a('string');
    if (options.verbose) {
	console.log(`adding maybe clue, ${line}`);
    }
    Expect(ClueManager.isRejectSource(args.nameList)).to.be.false;
    let [name, note] = getNameNote(line);
    if (ClueManager.addMaybe(name, args.nameList, note)) {
	args.count.maybeClues += 1;
	args.count.maybeCountSet.add(args.nameList.length);
	console.log(`added maybe clue, ${name} : ${args.nameList} - ${note}`);
    }
    args.flags.maybe = true;
    return args;
}

//

function processKnown (line, args, options) {
    Expect(line).to.be.a('string');
    if (options.verbose) {
	console.log(`skipping known clue, ${line}`);
    }
    // TODO: if note present, and no note in known list, add note
    // do nothing
    return args;
}

//

function addUrl (urlList, url) {
    Expect(urlList).to.be.an('array');
    Expect(url).to.be.a('string');

    if (!urlList.includes(url)) {
	urlList.push(url);
	return true;
    }
    return false;
}

//

function updateFilteredUrls (args) {
    if (!args.flags || !args.url) return args;
    if (args.flags.clue) {
	// we added a clue. add url to knownUrls if not already
	if (addUrl(args.filteredUrls.knownUrls, args.url)) {
	    console.log(`added clue url, ${args.url}`);
	    args.count.knownUrls += 1;
	    args.filteredUrls.anyChange = true;
	}
    }
    // note that, we add maybe even if already added as known; a bit strange
    if (args.flags.maybe) {
	// we added a maybe clue. add url to maybeUrls if not already
	if (addUrl(args.filteredUrls.maybeUrls, args.url)) {
	    console.log(`added maybe url, ${args.url}`);
	    args.count.maybeUrls += 1;
	    args.filteredUrls.anyChange = true;
	}
    }
    if (args.flags.reject) {
	if (addUrl(args.filteredUrls.rejectUrls, args.url)) {
	    console.log(`added reject url, ${args.url}`);
	    args.count.rejectUrls += 1;
	    args.filteredUrls.anyChange = true;
	}
    }
    return args;
}

//

function writeFilteredUrls (result) {
    if (result.filteredUrls && result.filteredUrls.anyChange) {
	const fu = result.filteredUrls;
	Expect(fu).to.exist;
	fu.knownUrls = _.sortedUniq(fu.knownUrls.sort());
	fu.maybeUrls = _.sortedUniq(fu.maybeUrls.sort());
	fu.rejectUrls = _.sortedUniq(fu.rejectUrls.sort());
	Fs.writeFileSync(result.filteredUrlPath, JSON.stringify(fu));
	result.filteredUrls.anyChange = false;
    }
}

//

function preProcess (state, dir, args) {
    switch (state) {
    case Src:
    case Url:
    case Done:
	writeFilteredUrls(updateFilteredUrls(args));
	break;

    default:
	break;
    }
    switch (state) {
    case Src:
	// clear everything but counts and dir for each new word list
	args = {
	    count : args.count,
	    dir
	};
	break;

    case Url:
	// clear flags for each new URL
	args.flags = {};
	break;

    default:
	// default case, pass through all values
	break;
    }
    return args;
}

//

function skipState (state, result, options) {
    let skip = _.isUndefined(result.nextState) ? false : result.nextState !== state;
    if (skip && options.verbose) {
	console.log(`skipping line: ${state.toString()}`);
    }
    return skip;
}

//
//

function updateResults(inputFilename, dir, options) {
    Expect(dir).to.be.a('string');
    Expect(inputFilename).to.be.a('string');

    let readLines = new Readlines(inputFilename);
    let state = Start;
    let lineNumber = 0;
    let inputLine;
    let result = {
	count : {
	    knownUrls      : 0,
	    rejectUrls     : 0,
	    maybeUrls     : 0,
	    knownClues     : 0,
	    rejectClues    : 0,
	    maybeClues    : 0,
	    knownCountSet  : new Set(),
	    rejectCountSet : new Set(),
	    maybeCountSet : new Set()
	}
    };
    // TODO: result.requiredNextState = Src

    while ((inputLine = readLines.next()) !== false) {
	lineNumber += 1;
	inputLine = inputLine.toString().trim();
	if (_.isEmpty(inputLine)) continue;
	let [ line, nextState ] = getLineState(inputLine);
	if (!SM[state].next.includes(nextState)) {
	    throw new Error(`Cannot transition from ${state.toString()} to ${nextState.toString()}` +
			    `, line ${inputLine}`);
	}
	state = nextState;
	if (skipState(state, result, options)) continue;
	const args = preProcess(state, dir, result);
	// TODO: try/catch block
	result = SM[state].func(line, args, options);
    }
    return result;
}

//
//
//

function main() {
    Expect(Opt.argv.length, 'exactly one FILE argument is required').to.equal(1);
    Expect(Opt.options.dir, '-d DIR is required').to.exist;

    if (Opt.options.verbose) {
	console.log('verbose: true');
    }

    ClueManager.loadAllClues({ clues: Clues.getByOptions(Opt.options) });

    let inputFilename = Opt.argv[0];
    console.log(`file: ${inputFilename}`);
    let result = updateResults(inputFilename, Opt.options.dir, {
	verbose: Opt.options.verbose
    });
    // hacky !?
    preProcess(Done, Opt.options.dir, result);
    if (Opt.options.save) {
	if (result.count.knownClues > 0) {
	    // save clues
	    let countList = Array.from(result.count.knownCountSet);
	    Expect(countList).to.be.not.empty;
	    ClueManager.saveClues(countList);
	}
	if (result.count.maybeClues > 0) {
	    // save maybes
	    let countList = Array.from(result.count.maybeCountSet);
	    Expect(countList).to.be.not.empty;
	    ClueManager.saveMaybes(countList);
	}	    
	if (result.count.rejectClues > 0) {
	    // save rejects
	    let countList = Array.from(result.count.rejectCountSet);
	    Expect(countList).to.be.not.empty;
	    ClueManager.saveRejects(countList);
	}	    
    }
    console.log(`updated knownClues(${result.count.knownClues})` +
		`, maybeClues(${result.count.maybeClues})` +
		`, rejectClues(${result.count.rejectClues})` +
		`, knownUrls(${result.count.knownUrls})` +
		`, maybeUrls(${result.count.maybeUrls})` +
		`, rejectUrls(${result.count.rejectUrls})`);
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

