//
// update.js
//
// somewhat brittle, ugly, mostly sychronous code that should be cleaned up,
// but it performs its function.
//

'use strict';

const _            = require('lodash');

const ClueManager  = require('../dist/modules/clue-manager');

const Clues        = require('./clue-types');
const Dir          = require('node-dir');
const Duration     = require('duration');
const Expect       = require('should/as-function');
const Filter       = require('./filter');
const Fs           = require('fs-extra');
const Log          = require('./log')('update');
const Markdown     = require('./markdown');
const My           = require('./util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Readlines    = require('n-readlines');
const SearchResult = require('./search-result');

//

const Start =   Symbol('start');
const Src =     Symbol('src');
const Url =     Symbol('url');
const Clue =    Symbol('clue');
const Maybe =   Symbol('maybe');
const Known =   Symbol('known');
const Done =    Symbol('done');

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
    Expect(line.length >= 1).is.true(); // at.least(1)
    let state;
    if (Markdown.hasSourcePrefix(line)) state = Src;
    if (Markdown.hasMaybePrefix(line)) state = Maybe;
    if (Markdown.hasKnownPrefix(line)) state = Known;
    // all of the above get first char sliced off
    if (state) {
        line = line.slice(1);
    } else if (line.startsWith('http')) {
        state = Url;
    } else {
	// TODO: enforce alpha-numeric 1st char

        // else, probably a clue
        state = Clue;
    }
    return [line, state];
}

// process a line beginning with '@' that represents
// a comma-separated list of words

function processSrc (rawLine, args, options) {
    Expect(rawLine).is.a.String();
    Expect(args.dir).is.a.String();

    let done = false;
    const [reject, line] = Markdown.hasSuffix(rawLine, Markdown.Suffix.reject);
    Log.debug(`src: ${line}`);
    let nameList = line.split(',');
    Expect(nameList.length).is.above(1); // at.least(2)
    if (reject) {
        if (ClueManager.addReject(nameList)) {
            args.count.rejectClues += 1;
            args.count.rejectCountSet.add(nameList.length);
        }
        done = true;
    } else if (ClueManager.isRejectSource(line)) {
        done = true;
    } else {
        const [clue, unused] = Markdown.hasSuffix(rawLine, Markdown.Suffix.clue);
        if (clue && options.yes_mode) {
            console.log(line);
            done = true;
        }
    }
    if (done) {
        return {
            timing:    args.timing,
            count:     args.count,
            nextState: Src
        };
    }

    //let dir = `${SearchResult.DIR}${args.dir}`;
    //let path = `${dir}/${SearchResult.makeFilteredFilename(nameList)}`;
    let path = SearchResult.pathFormat({
        //root: args.root,
        dir:  args.dir, // || _.toString(wordList.length),
        base: SearchResult.makeFilteredFilename(nameList)
    }, options);

    let content;
    try {
        content = Fs.readFileSync(path, 'utf8');
    } catch(err) {
        // file not existing is OK.
        // check err.code NOENT
    }
    let flags = {};
    let filteredUrls;
    if (content) {
        filteredUrls = JSON.parse(content);
        Log.debug(`loaded: ${path}`);
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
        timing:          args.timing,
        count:           args.count,
        nameList:        nameList,
        filteredUrlPath: path,
        filteredUrls,
        flags
    };
}

//

function processUrl (line, args, options) {
    Expect(line).is.a.String();
    Expect(args.filteredUrls).is.an.Object();

    let [url, suffix] = Markdown.getSuffix(line);
    args.url = url;
    if (suffix === Markdown.Suffix.clue) {
        args.flags.clue = true;
        if (options.yes_mode) {
            console.log(args.nameList.join(','));
            return {
                timing:    args.timing,
                count:     args.count,
                nextState: Src
            };
        }
    }
    else if (suffix === Markdown.Suffix.reject) {
        args.flags.reject = true;
    }
    else {
        Expect(suffix).is.undefined();
    }
    return args;
}

//

function addClues (countList, name, src) { // , add = ClueManager.addClue) {
    Expect(countList).is.an.Array();
    Expect(name).is.a.String();
    Expect(src).is.a.String();

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
    Expect(line).is.a.String();

    let [name, note] = getNameNote(line);
    // we're about to update known clues. double-sanity check.
    Expect(ClueManager.isRejectSource(args.nameList)).is.false();
    
    let countList = args.countList;
    if (!countList) {
        let start = new Date();
        countList = ClueManager.getCountList(args.nameList);
        let ms = new Duration(start, new Date()).milliseconds;
        args.timing.getCountList += ms;
        args.countList = countList;
    }
    Log.debug(`countList: ${_.isEmpty(countList) ? "empty" : countList}`) ;
    countList = addClues(countList, name, args.nameList.toString());
    if (!_.isEmpty(countList)) {
        args.count.knownClues += 1;
	// TODO: can I knownCountSet.addAll(...countList) ?
        countList.forEach(count => args.count.knownCountSet.add(count));
        Log.info(`added clue, ${name} : ${args.nameList} - ${note} : [${countList}]`);
    }
    args.flags.clue = true;
    return args;
}

//

function processMaybe (line, args, options) {
    Expect(line).is.a.String();
    Expect(ClueManager.isRejectSource(args.nameList)).is.false();
    let [name, note] = getNameNote(line);
    if (ClueManager.addMaybe(name, args.nameList, note)) {
        args.count.maybeClues += 1;
        args.count.maybeCountSet.add(args.nameList.length);
        Log.info(`added maybe clue, ${name} : ${args.nameList} - ${note}`);
    }
    args.flags.maybe = true;
    return args;
}

//

function processKnown (line, args, options) {
    Expect(line).is.a.String();
    if (options.verbose) {
        console.log(`skipping known clue, ${line}`);
    }
    // TODO: if note present, and no note in known list, add note
    // do nothing
    return args;
}

//

function addUrl (urlList, url) {
    Expect(urlList).is.an.Array();
    Expect(url).is.a.String();

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
            Log.debug(`added clue url, ${args.url}`);
            args.count.knownUrls += 1;
            args.filteredUrls.anyChange = true;
        }
    }
    // note that, we add maybe even if already added as known; a bit strange
    if (args.flags.maybe) {
        // we added a maybe clue. add url to maybeUrls if not already
        if (addUrl(args.filteredUrls.maybeUrls, args.url)) {
            Log.debug(`added maybe url, ${args.url}`);
            args.count.maybeUrls += 1;
            args.filteredUrls.anyChange = true;
        }
    }
    if (args.flags.reject) {
        if (addUrl(args.filteredUrls.rejectUrls, args.url)) {
            Log.debug(`added reject url, ${args.url}`);
            args.count.rejectUrls += 1;
            args.filteredUrls.anyChange = true;
        }
    }
    return args;
}

//

function writeFilteredUrls (result) {
    const fu = result.filteredUrls;
    if (fu && fu.anyChange) {
        fu.knownUrls = _.sortedUniq(fu.knownUrls.sort());
        fu.maybeUrls = _.sortedUniq(fu.maybeUrls.sort());
        fu.rejectUrls = _.sortedUniq(fu.rejectUrls.sort());
        Fs.writeFileSync(result.filteredUrlPath, JSON.stringify(fu));
        result.filteredUrls.anyChange = false;
    }
}

//

function preProcess (state, args, options) {
    switch (state) {
    case Src:
    case Url:
    case Done:
        if (options.production && !options.dry_run) {
            writeFilteredUrls(updateFilteredUrls(args));
        }
        break;

    default:
        break;
    }
    switch (state) {
    case Src:
        // clear everything but timing, counts and dir for each new source
        args = {
            timing: args.timing,
            count : args.count,
            dir   : args.dir
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
    return result.nextState ? result.nextState !== state : false;
}

//

function saveClues (result, options) {
    if (options.save && !options.production && !options.force) {
        throw new Error('--save only allowed with --production (or --force)');
    }
    let totalClueCount = 0;
    if (result.count.knownClues > 0) {
        totalClueCount += result.count.knownClues;
        if (options.save || options.verbose) {
            // save known clues
            let countList = Array.from(result.count.knownCountSet);
            Expect(countList).is.not.empty();
            // TODO: My.optlog(options, msg...)
            // some way to tweak util.Debug()('name') to current module's debug instance name?
            // can i say My=require('util')(Debug) or (MODULE_NAME)
            Log.info(`knownList: ${countList}`);
            if (options.save) {
                ClueManager.saveClues(countList);
            }
        }
    }
    if (result.count.maybeClues > 0) {
        totalClueCount += result.count.maybeClues;
        if (options.save || options.verbose) {
            // save maybes
            let countList = Array.from(result.count.maybeCountSet);
            Expect(countList).is.not.empty();
            Log.info(`maybeList: ${countList}`);
            if (options.save) {
                ClueManager.saveMaybes(countList);
            }
        }
    }
    if (result.count.rejectClues > 0) {
        totalClueCount += result.count.rejectClues;
        if (options.save || options.verbose) {
            // save rejects
            let countList = Array.from(result.count.rejectCountSet);
            Expect(countList).is.not.empty();
            Log.info(`rejectList: ${countList}`);
            if (options.save) {
                ClueManager.saveRejects(countList);
            }
        }
    }
    if (!totalClueCount) {
        Log.info(`no new known clues, maybes, or rejects`);
    }
    return totalClueCount;
}

//

function initResult () {
    return {
        count : {
            knownUrls      : 0,
            rejectUrls     : 0,
            maybeUrls      : 0,
            knownClues     : 0,
            rejectClues    : 0,
            maybeClues     : 0,
            knownCountSet  : new Set(),
            rejectCountSet : new Set(),
            maybeCountSet  : new Set()
        },
        timing : {
            getCountList : 0
        }
    };
}

//

async function _updateFromFile(filename, options) {
    Expect(filename).is.a.String();
    Log.info(`file: ${Path.basename(filename)}`);

    if (!ClueManager.loaded) {
        Log.debug('updateFromFile: calling ClueManager.loadAllClues()');
        ClueManager.loadAllClues({ clues: Clues.getByOptions(options) });
    }

    const dir = options.dir || '2'; // lil haxy
    let result = initResult();
    // TODO: result.requiredNextState = Src

    let state = Start;
    let lineNumber = 0;
    let inputLine;
    let readLines = new Readlines(filename);

    while ((inputLine = readLines.next()) !== false) {
        lineNumber += 1;
        inputLine = inputLine.toString().trim();
        if (_.isEmpty(inputLine)) continue;
        let [line, nextState] = getLineState(inputLine);
	Log.debug(`nextState: ${nextState.toString()}, line: ${line}`);
        if (!SM[state].next.includes(nextState)) {
            throw new Error(`Cannot transition from ${state.toString()}` +
                            ` to ${nextState.toString()}, line ${inputLine}`);
        }
        state = nextState;
        if (skipState(state, result, options)) {
            Log.debug(`skipping line: ${state.toString()}, ${line}`);
            continue;
        }

        result.dir = dir;
        const args = preProcess(state, result, options);
        // TODO: try/catch block
        result = SM[state].func(line, args, options);
    }
    // hacky !? yes
    result.dir = dir;
    preProcess(Done, result, options);
    return result;
}

//

async function updateFromFile(filename, options) {
    const start = new Date();
    return _updateFromFile(filename, options)
        .then(result => {
            if (options.perf) {
                const duration = new Duration(start, new Date()).milliseconds;
                Log.info(`updateFromFile duration(${PrettyMs(duration)})` +
                         `, getCountList(${PrettyMs(result.timing.getCountList)})`);
            }
            return saveClues(result, options);
        });
}
    
//

function aggregate (result, allResults) {
    allResults.timing.getCountList += result.timing.getCountList;

    allResults.count.knownClues += result.count.knownClues;
    allResults.count.rejectClues += result.count.rejectClues;
    allResults.count.maybeClues += result.count.maybeClues;
    allResults.count.knownCountSet = new Set([...result.count.knownCountSet, ...allResults.count.knownCountSet]);
    allResults.count.maybeCountSet = new Set([...result.count.maybeCountSet, ...allResults.count.maybeCountSet]);
    allResults.count.rejectCountSet = new Set([...result.count.rejectCountSet, ...allResults.count.rejectCountSet]);
    return allResults;
}

//

async function updateFromPathList(pathList, options) {
    Expect(pathList).is.an.Array();
//    if (options.dry_run) return;
    let allResults = initResult();
    let start = new Date();
    for (let path of pathList) {
        let result = await _updateFromFile(path, options)
            .then(result => {
                allResults = aggregate(result, allResults);
                return result;
            }).catch(err => {
                console.log(`updateFromFile, ${err}`);
                if (err && err.code !== 'ENOENT') {
                    throw err;
                }
            });
    }
    if (options.perf) {
        let duration = new Duration(start, new Date()).milliseconds;
        Log.info(`updateFromPathList duration(${PrettyMs(duration)})` +
                 `, getCountList(${PrettyMs(allResults.timing.getCountList)})`);
    }
    const count = allResults.count;
    Log.message(`new: known(${count.knownClues}), maybe(${count.maybeClues}), rejects(${count.rejectClues})`);

    return saveClues(allResults, options);
}

//

module.exports = {
    updateFromFile,
    updateFromPathList
}
