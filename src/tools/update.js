//
// UPDATE.JS
//

'use strict';

const _            = require('lodash');
const ClueManager  = require('../clue_manager');
const Dir          = require('node-dir');
const expect       = require('chai').expect;
const fs           = require('fs');
const Path         = require('path');
const Promise      = require('bluebird');
const Readlines    = require('n-readlines');
const Result       = require('./result-mod');

const Opt          = require('node-getopt').create([
    ['c', 'count',               'show result/url counts only'],
    ['d', 'dir=NAME',            'directory name'],
    ['s', 'synthesis',           'use synth clues'],
//    ['v', 'verbose',             'show logging']
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

//---------------------------------------------

const Start =   Symbol('start');
const Src =     Symbol('src');
const Url =     Symbol('url');
const Clue =    Symbol('clue');
const Maybe =   Symbol('maybe');
const Known =   Symbol('known');

const ClueSuffix =    'c';
const RejectSuffix =  'x';
const ValidSuffixes = [ ClueSuffix, RejectSuffix ];

// state machine
const SM = {
    [Start] : { next: [ Src ] },
    [Src]   : { next: [ Url ],                             func: processSrc },      
    [Url]   : { next: [ Url,   Clue,  Maybe, Known, Src ], func: processUrl },
    [Clue]  : { next: [ Url,   Clue,  Maybe, Known, Src ], func: processClue },
    [Maybe] : { next: [ Url,  /*no*/  Maybe, Known, Src ], func: processMaybe },
    [Known] : { next: [ Known, Src ],                      func: processKnown }
}

//---------------------------------------------

//
function getLineType(line) {
    expect(line.length).to.be.at.least(1);
    let lineType;
    if (line[0] === Result.SRC_PREFIX) lineType = Src;
    if (line[0] === Result.MAYBE_PREFIX) lineType = Maybe;
    if (line[0] === Result.KNOWN_PREFIX) lineType = Known;
    // all of the above get first char sliced off
    if (!_.isUndefined(lineType)) {
	line = line.slice(1);
    }

    if (line.startsWith('http')) lineType = Url;
    if (_.isUndefined(lineType)) {
	// else, probably a clue
	lineType = Clue;
    }
    return {
	line:     line,
	lineType: lineType
    };
}

//
function processSrc(line, args, options) {
    expect(line).to.be.a('string');
    expect(args.dir).to.be.a('string');

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
    expect(nameList.length).to.be.at.least(2);
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

    let dir = `${Result.DIR}${args.dir}`;
    let path = `${dir}/${Result.makeFilteredFilename(nameList)}`;
    let content;
    try {
	content = fs.readFileSync(path, 'utf8');
    } catch(err) {
	// file not existing is OK.
	if (options.verbose) {
	    console.log('filtered file not found');
	}
    }
    // if file exists, notify if it doesn't parse correctly.
    let filteredUrls;
    if (!_.isUndefined(content)) {
	filteredUrls = JSON.parse(content);
	console.log(`loaded: ${path}`);
    }
    else {
	filteredUrls = {
	    knownUrls:  [],
	    rejectUrls: []
	};
    }
    return { 
	count:           args.count,
	nameList:        nameList,
	filteredUrlPath: path,
	filteredUrls:    filteredUrls
    };
}

//
function getUrlSuffix(line, options) {
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
	    expect(ValidSuffixes.includes(suffix), `bad url suffix, ${line}`).to.be.true;
	}
    }
    return {
	url:    url,
	suffix: suffix
    };
}

//
function addUrl(urlList, url) {
    expect(urlList).to.be.an('array');
    expect(url).to.be.a('string');

    if (!urlList.includes(url)) {
	urlList.push(url);
	return true;
    }
    return false;
}

//
function processUrl(line, args, options) {
    expect(line).to.be.a('string');
    expect(args.filteredUrls).to.be.an('object');

    // for clue suffix or undefined suffix, save the URL
    args.url = undefined;
    let urlSuffix = getUrlSuffix(line, options);
    if (urlSuffix.suffix === ClueSuffix) {
	if (addUrl(args.filteredUrls.knownUrls, urlSuffix.url)) {
	    console.log(`added clue url, ${urlSuffix.url}`);
	    args.count.knownUrls += 1;
	    args.urlChange = true;
	}
	args.url = urlSuffix.url; // save for ClueSuffix
	args.urlAdded = true;
    }
    else if (urlSuffix.suffix === RejectSuffix) {
	if (addUrl(args.filteredUrls.rejectUrls, urlSuffix.url)) {
	    console.log(`added reject url, ${urlSuffix.url}`);
	    args.count.rejectUrls += 1;
	    args.urlChange = true;
	}
    }
    else {
	expect(urlSuffix.suffix).to.be.undefined;
	args.url = urlSuffix.url; // save for undefined
    }
    return args;
}

//

function addClues(countList, name, src) {
    expect(countList).to.be.an('array');
    expect(name).to.be.a('string');
    expect(src).to.be.a('string');

    let updatedCountList = [];
    countList.forEach(count => {
	if (ClueManager.addClue(count, {
	    name: name,
	    src:  src
	}, false, true)) { // save = false, nothrow = true
	    updatedCountList.push(count);
	}
    });
    return updatedCountList;
}

//
function processClue(line, args, options) {
    expect(line).to.be.a('string');

    let name;
    let note;
    let firstComma = line.indexOf(',');
    if (firstComma !== -1) {
	name = line.slice(0, firstComma);
	if (line.length > firstComma + 1) {
	    note = line.slice(firstComma + 1, line.length);
	}
    }
    // we're about to update known clues. double-sanity check.
    expect(ClueManager.isRejectSource(args.nameList)).to.be.false;
    let countList = addClues(ClueManager.getCountList(args.nameList), name,
			     args.nameList.toString());
    if (!_.isEmpty(countList)) {
	args.count.knownClues += 1;
	countList.forEach(count => args.count.knownCountSet.add(count));
	console.log(`added clue, ${name} : ${args.nameList} - ${note} : [${countList}]`);
    }
    // we added a clue, add url to knownUrls if not already
    if (!args.urlAdded) {
	if (addUrl(args.filteredUrls.knownUrls, args.url, options)) {
	    console.log(`added clue url, ${args.url}`);
	    args.count.knownUrls += 1;
	    args.urlChange = true;
	}
	args.urlAdded = true;
    }
    return args;
}

//
function processMaybe(line, args, options) {
    expect(line).to.be.a('string');
    if (options.verbose) {
	console.log(`adding maybe clue, ${line}`);
    }
    return args;
}

//
function processKnown(line, args, options) {
    expect(line).to.be.a('string');
    if (options.verbose) {
	console.log(`skipping known clue, ${line}`);
    }
    // do nothing
    return args;
}

//
function writeFilteredUrls(result) {
    if (result.urlChange) {
	const fu = result.filteredUrls;
	expect(fu).is.not.undefined;
	fu.knownUrls = _.sortedUniq(fu.knownUrls.sort());
	fu.rejectUrls = _.sortedUniq(fu.rejectUrls.sort());
	fs.writeFileSync(result.filteredUrlPath, JSON.stringify(fu));
	result.urlChange = false; // shouldn't need to do this, but..
    }
}

//
function preProcess(state, dir, result) {
    if (state === Src) {
	writeFilteredUrls(result);
	return {
	    count : result.count,
	    dir   : dir
	};
    }
    return result;
}

//
function skipState(state, result, options) {
    let skip = (!_.isUndefined(result.nextState) && (result.nextState !== state)) 
    if (skip && options.verbose) {
	console.log(`skipping line: ${state.toString()}`);
    }
    return skip;
}

//
//
function updateResults(inputFilename, dir, options) {
    expect(dir).to.be.a('string');
    expect(inputFilename).to.be.a('string');

    let readLines = new Readlines(inputFilename);
    let state = Start;
    let lineNumber = 0;
    let inputLine;
    let result = {
	count : {
	    knownUrls      : 0,
	    rejectUrls     : 0,
	    knownClues     : 0,
	    rejectClues    : 0,
	    knownCountSet  : new Set(),
	    rejectCountSet : new Set()
	}
    };
    // TODO: result.requiredNextState = Src

    while ((inputLine = readLines.next()) !== false) {
	lineNumber += 1;
	if ((inputLine = inputLine.toString().trim()) === '') continue;
	// TODO: should be nextState = getNextState
	let { line, lineType } = getLineType(inputLine);
	if (!SM[state].next.includes(lineType)) {
	    throw new Error(`Cannot transition from ${state} to ${lineType}, line ${inputLine}`);
	}
	state = lineType;
	if (skipState(state, result, options)) {
	    continue;
	}
	const args = preProcess(state, dir, result);
	// TODO: try/catch block
	result = SM[state].func(line, args, options);
    }
    writeFilteredUrls(result);
    return result;
}

//
//
//

function main() {
    expect(Opt.argv.length, 'exactly one FILE argument is required').to.equal(1);
    expect(Opt.options.dir, '-d DIR is required').to.exist;

    let base = Opt.options.synth ? 'synth' : 'meta';
    let saveClues = Opt.options.dir !== 'tmp';
    let verbose = !saveClues;
    
    if (verbose) {
	console.log('verbose: true');
    }

    ClueManager.loadAllClues({
	baseDir:  base
    });

    let inputFilename = Opt.argv[0];
    console.log(`file: ${inputFilename}`);
    let result = updateResults(inputFilename, Opt.options.dir, {
	verbose: verbose
    });
    if (saveClues) {
	if (result.count.knownClues > 0) {
	    // save clues
	    let countList = Array.from(result.count.knownCountSet);
	    expect(countList).to.be.not.empty;
	    ClueManager.saveClues(countList);
	}
	if (result.count.rejectClues > 0) {
	    // save rejects
	    let countList = Array.from(result.count.rejectCountSet);
	    expect(countList).to.be.not.empty;
	    ClueManager.saveRejects(countList);
	}	    
    }
    console.log(`updated knownClues(${result.count.knownClues})` +
		`, rejectClues(${result.count.rejectClues})` +
		`, knownUrls(${result.count.knownUrls})` +
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

