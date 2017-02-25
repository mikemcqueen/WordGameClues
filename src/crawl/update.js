//
// UPDATE.JS
//

'use strict';

const _            = require('lodash');
const Promise      = require('bluebird');
const Dir          = require('node-dir');
const fs           = require('fs');
const Path         = require('path');
const Readlines    = require('n-readlines');
const expect       = require('chai').expect;

const ClueManager  = require('../clue_manager.js');

const Opt          = require('node-getopt').create([
    ['c', 'count',               'show result/url counts only'],
    ['d', 'dir=NAME',            'directory name'],
    ['s', 'synthesis',           'use synth clues'],
//    ['v', 'verbose',             'show logging']
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

//

const RESULT_DIR = '../../data/results/';
const FILTERED_SUFFIX = '_filtered';

//

function makeFilename(wordList, suffix) {
    expect(wordList.length).to.be.at.least(2);

    let filename = '';
    wordList.forEach(word => {
	if (_.size(filename) > 0) {
	    filename += '-';
	}
	filename += word;
    });
    if (!_.isUndefined(suffix)) {
	filename += suffix;
    }
    return filename + '.json';
}

//

function makeFilteredFilename(wordList) {
    return makeFilename(wordList, FILTERED_SUFFIX);
}

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

//
function getLineType(line) {
    expect(line.length).to.be.at.least(1);
    let lineType;
    if (line[0] === '@') lineType =         Src;
    if (line[0] === ':') lineType =         Maybe;
    if (line[0] === '#') lineType =        Known;
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

    // TODO: process ,x

    console.log(`src: ${line}`);
    let nameList = line.split(',');
    expect(nameList.length).to.be.at.least(2);
    //if (_.isUndefined(dir)) dir = nameList.length; // huh?
    let dir = `${RESULT_DIR}${args.dir}`;
    let path = `${dir}/${makeFilteredFilename(nameList)}`;
    console.log(`file: ${path}`);
    let filteredUrls;

    let content;
    try {
	content = fs.readFileSync(path, 'utf8');
    } catch(err) {
	// file not existing is OK.
	if (options.logOnly) {
	    console.log('filtered file not found');
	}
    }
    // if file exists, notify if it doesn't parse correctly.
    if (!_.isUndefined(content)) {
	filteredUrls = JSON.parse(content);
    }
    else {
	filteredUrls = {
	    knownUrls:  [],
	    rejectUrls: []
	};
    }

    return { 
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
	    if (options.logOnly) {
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
function addClueUrl(urlList, url, options) {
    expect(urlList).to.be.an('array');
    expect(url).to.be.a('string');
    expect(options).to.be.an('object');

    if (options.logOnly) {
	console.log(`adding clue URL, ${url}`);
    }
    urlList.push(url);
}

//
function processUrl(line, args, options) {
    expect(line).to.be.a('string');
    expect(args.filteredUrls).to.be.an('object');

    // for clue suffix or undefined suffix, save the URL
    args.url = undefined;
    let urlSuffix = getUrlSuffix(line, options);
    if (urlSuffix.suffix === ClueSuffix) {
	addClueUrl(args.filteredUrls.knownUrls, urlSuffix.url, options);
	args.url = urlSuffix.url; // save for ClueSuffix
	args.urlAdded = true;
    }
    else if (urlSuffix.suffix === RejectSuffix) {
	if (options.logOnly) {
	    console.log(`adding reject URL, ${urlSuffix.url}`);
	}
	args.filteredUrls.rejectUrls.push(urlSuffix.url);
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

    countList.forEach(count => {
	if (ClueManager.addClue(count, {
	    name: name,
	    src:  src
	})) { // save = false
	    console.log('updated ' + count);
	}
    });
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
    if (options.logOnly) {
	console.log(`adding known clue, ${name} : ${args.nameList} - ${note}`);
    }
    addClues(ClueManager.getCountList(args.nameList), name, args.nameList.toString());
    if (!args.urlAdded) {
	addClueUrl(args.filteredUrls.knownUrls, args.url, options);
	args.urlAdded = true;
    }

    return args;
}

//
function processMaybe(line, args, options) {
    expect(line).to.be.a('string');
    console.log(`adding maybe clue, ${line}`);
    return args;
}

//
function processKnown(line, args, options) {
    expect(line).to.be.a('string');
    console.log(`skipping known clue, ${line}`);
    // do nothing
    return args;
}

//
function writeFilteredUrls(result) {
    if (!_.isUndefined(result)) {
	const fu = result.filteredUrls;
	fu.knownUrls = _.sortedUniq(fu.knownUrls.sort());
	fu.rejectUrls = _.sortedUniq(fu.rejectUrls.sort());
	fs.writeFileSync(result.filteredUrlPath, JSON.stringify(fu));
    }
}

//
function preProcess(state, dir, result) {
    if (state === Src) {
	writeFilteredUrls(result);
	return {
	    dir: dir
	};
    }
    return result;
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
    let result;
    while ((inputLine = readLines.next()) !== false) {
	lineNumber += 1;
	if ((inputLine = inputLine.toString().trim()) === '') continue;
	let { line, lineType } = getLineType(inputLine);
	if (!SM[state].next.includes(lineType)) {
	    throw new Error(`Cannot transition from ${state} to ${lineType}, line ${inputLine}`);
	}
	state = lineType;
	const args = preProcess(state, dir, result);
	// TODO: try/catch block
	result = SM[state].func(line, args, options);
    }
}

//
//
//

function main() {
    expect(Opt.argv.length, 'filter FILE argument required').to.equal(1);

    let base = Opt.options.synth ? 'synth' : 'meta';
    let logOnly = Opt.options.dir === 'tmp';
    if (logOnly) {
	console.log('logOnly: true');
    }

    ClueManager.loadAllClues({
	baseDir:  base
    });

    let inputFilename = Opt.argv[0];
    console.log(`file: ${inputFilename}`);
    updateResults(inputFilename, Opt.options.dir, {
	logOnly: logOnly
    });
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

