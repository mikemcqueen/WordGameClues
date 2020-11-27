//
//
// clue-manager.js
//

'use strict';

// export a singleton

module.exports = new ClueManager();

const _              = require('lodash');
const ClueList       = require('../types/clue-list');
const Clues          = require('./clue-types');
const Debug          = require('debug')('clue-manager');
const Duration       = require('duration');
const Expect         = require('should/as-function');
const Log            = require('./log')('clue-manager');
const NameCount      = require('../types/name-count');
const Path           = require('path');
const Peco           = require('./peco');
const PrettyMs       = require('pretty-ms');
//const Stringify      = require('javascript-stringify');
const Validator      = require('./validator');

const stringify = require('javascript-stringify').stringify;
//let Stringify = stringify;

function Stringify(val) {
    return stringify(val, (value, indent, stringify) => {
	if (typeof value == 'function') return "function";
	return stringify(value);
    }, " ");
}

//

const DATA_DIR              =  Path.normalize(`${Path.dirname(module.filename)}/../../data/`);
const REJECTS_DIR           = 'rejects';

// constructor

function ClueManager () {
    this.clueListArray = [];         // the JSON clue files in an array
    this.maybeListArray = [];        // the JSON maybe files in an array
    this.rejectListArray = [];       // the JSON reject files in an array
    this.knownClueMapArray = [];     // map clue name to clue src
    this.knownSourceMapArray = [];   // map known source to list of clues
    this.rejectSourceMap = {};       // map reject source to true/false (currently)

    this.rvsSuccessSeconds = 0;
    this.rvsFailDuration = 0;

    this.loaded = false;
    this.maxClues = 0;

//    this.logging = false;
    this.logging = true;

    this.logLevel = 0;
}

//

function showStrList (strList) {
    let result = "";
    let first = true;
    for (let str of strList) {
        if (!first) result += ' - ';
	result += str;
        first = false;
    }
    return _.isEmpty(result) ? "[]" : result;
}

//

ClueManager.prototype.log = function (text) {
    let pad = '';
    let index;
    if (this.logging) {
        for (let index = 0; index < this.logLevel; index += 1) { pad += ' '; }
        console.log(pad + text);
    }
};

//

ClueManager.prototype.saveClueList = function (list, count, options = {}) {
    list.save(this.getKnownFilename(count, options.dir));
};


const autoSource = (clueList) => {
    let source = 0;
    for (let clue of clueList) {
	if (clue.src != 'same') source += 1;
	clue.src = `${source}`;
    }
    console.log(`autoSource: ${source} primary clues`);
    //console.log(Stringify(clueList));
};



// args:
//  baseDir:  base directory (meta, synth)
//  ignoreErrors:
//  validateAll:
//

ClueManager.prototype.loadAllClues = function (args) {
    this.dir = Clues.getDirectory(args.clues);
    if (args.ignoreErrors) {
        this.ignoreLoadErrors = true;
    }
    this.maxClues = args.clues.clueCount;

    for (let count = 1; count <= this.maxClues; ++count) {
        let knownClueList = this.loadClueList(count);
        this.clueListArray[count] = knownClueList;
        if (count === 1) {
	    if (knownClueList[0].src == 'auto') {
		autoSource(knownClueList);
	    }
            this.addKnownPrimaryClues(knownClueList);
        }
        else {
            this.addKnownCompoundClues(knownClueList, count, args.validateAll);
        }
    }

    for (let count = 2; count <= this.maxClues; ++count) {
        let rejectClueList;
        try {
            rejectClueList = ClueList.makeFrom({
                filename : this.getRejectFilename(count)
            });
        }
        catch (err) {
            console.log(`WARNING! reject file: ${this.getRejectFilename(count)}, ${err}`);
        }
        if (rejectClueList) {
            this.rejectListArray[count] = rejectClueList;
            this.addRejectCombos(rejectClueList, count);
        }
    }

    this.loaded = true;

    return this;
};

//

ClueManager.prototype.loadClueList = function (count, options = {}) {
    return ClueList.makeFrom({
        filename : this.getKnownFilename(count, options.dir)
    });
};

//

ClueManager.prototype.addPrimaryClueToMap = function (clue) {
    const count = 1;
    const clueMap = this.knownClueMapArray[count];
    if (!_.has(clueMap, clue.name)) {
        clueMap[clue.name] = [];
    }
    clueMap[clue.name].push(clue.src);
}

//

ClueManager.prototype.addKnownPrimaryClues = function (clueList) {
    const count = 1;
    let clueMap = this.knownClueMapArray[count] = {};
    clueList.forEach(clue => {
        if (clue.ignore) {
            return; // continue
        }
	this.addPrimaryClueToMap(clue);
    });
    return this;
};

//

ClueManager.prototype.getKnownFilename = function (count, dir = undefined) {
    return Path.format({
        dir:  _.isUndefined(dir) ? this.dir : `${DATA_DIR}${dir}`,
        base: `clues${count}.json`
    });
};

//

ClueManager.prototype.getRejectFilename = function (count) {
    return Path.format({
        dir:  `${DATA_DIR}${REJECTS_DIR}`,
        base: `rejects${count}.json`
    });
};


function validateAndSaveResults (dst, args) {
    let vsResult = Validator.validateSources(args);
    if (args.validateAll) {
        // this is where the magic happens
        dst.results = vsResult.list;
    }
    return vsResult;
}

ClueManager.prototype.addCompoundClue = function (clue, count, validateAll = true) {
    Expect(clue).is.an.Object();
    Expect(count).is.a.Number();
    let nameList = clue.src.split(',').sort();
    let srcMap = this.knownSourceMapArray[count];
    let srcKey = nameList.toString();
    // new sources need to be validated
    let vsResult = { success: true };
    if (!_.has(srcMap, srcKey)) {
        srcMap[srcKey] = { clues: [] };
        Debug(`## validating Known compound clue: ${srcKey}:${count}`);
	vsResult = validateAndSaveResults(srcMap[srcKey], {
	    sum: count,
	    nameList,
	    count: nameList.length,
	    validateAll
	});
    }
    srcMap[srcKey].clues.push(clue);
    return vsResult;
}

//

ClueManager.prototype.addKnownCompoundClues = function (clueList, clueCount, validateAll) {
    // so this is currently only callable once per clueCount.
    Expect(this.knownClueMapArray[clueCount]).is.undefined();
    Expect(clueCount > 1);
    this.knownClueMapArray[clueCount] = {};
    //if (clueCount > 1) {
        Expect(this.knownSourceMapArray[clueCount]).is.undefined();
        this.knownSourceMapArray[clueCount] = {};
    //}
    clueList.forEach(clue => {
        if (clue.ignore) {
            return; // continue
        }
	let nameList = clue.src.split(',').sort();
	clue.src = nameList.toString();
	//if (clueCount > 1) {
	let result = this.addCompoundClue(clue, clueCount, validateAll);
        if (!this.ignoreLoadErrors) {
	    if (!result.success) {
		console.log(`VALIDATE FAILED KNOWN COMPOUND CLUE: '${clue.src}':${clueCount}`);
	    }
            Expect(result.success);
        }
	//}
        this.addKnownClue(clueCount, clue.name, clue.src);
    }, this);
    return this;
};

//

ClueManager.prototype.addKnownClue = function (count, name, source, nothrow) {
    Expect(count).is.a.Number();
    Expect(name).is.a.String();
    Expect(source).is.a.String();
    let clueMap = this.knownClueMapArray[count];
    if (!_.has(clueMap, name)) {
        this.log(`clueMap[${name}] = [ ${source} ]`);
        clueMap[name] = [ source ];
    } else if (!clueMap[name].includes(source)) {
        this.log(`clueMap[${name}] += ${source} (${count})`);
        clueMap[name].push(source);
    } else {
        if (nothrow) return false;
        throw new Error('duplicate clue name/source' + 
                        '(' + count + ') ' +
                        name + ' : ' + source);
    }
    return true;
};

//

ClueManager.prototype.removeKnownClue = function (count, name, source, nothrow) {
    Expect(count).is.a.Number();
    Expect(name).is.a.String();
    Expect(source).is.a.String();
    let clueMap = this.knownClueMapArray[count];
    if (!_.has(clueMap, name) || !clueMap[name].includes(source)) {
        if (nothrow) return false;
        throw new Error(`removeKnownClue, missing clue: ${name}:${source} at count: ${count}`);
    } else {
        Debug(`before clueMap[${name}]: len(${clueMap[name].length}), sources:${clueMap[name]}`);
        Debug(`removing clue: [${name}] : ${source} from count: ${count}`);
        _.pull(clueMap[name], source);
        Debug(`after clueMap[${name}]: len(${clueMap[name].length}), sources: ${clueMap[name]}`);
    }
    return true;
};

//

ClueManager.prototype.saveClues = function (counts) {
    if (_.isNumber(counts)) {
        counts = [ counts ];
    }
    Debug(`saving clueLists ${counts}`);
    Expect(counts).is.an.Array();
    for (const count of counts) {
        this.saveClueList(this.clueListArray[count], count);
        Debug(`saved clueList ${count}, length: ${this.clueListArray[count].length}`);
    }
};

//

ClueManager.prototype.addClue = function (count, clue, save = false, nothrow = false) {
    // sort src
    clue.src = clue.src.split(',').sort().toString();
    if (this.addKnownClue(count, clue.name, clue.src, nothrow)) {
        this.clueListArray[count].push(clue);
        if (save) {
            this.saveClues(count);
        }
        return true;
    }
    return false;
};

//

ClueManager.prototype.removeClue = function (count, clue, save = false, nothrow = false) {
    // sort src
    clue.src = clue.src.split(',').sort().toString();
    if (this.removeKnownClue(count, clue.name, clue.src, nothrow)) {
        _.remove(this.clueListArray[count], elem => {
            return (elem.name === clue.name) && (elem.src === clue.src);
        });
        if (save) {
            this.saveClues(count);
        }
        return true;
    }
    return false;
};

//

ClueManager.prototype.addMaybe = function (name, srcNameList, note, save = false) {
    return false;
    if (_.isString(srcNameList)) {
        srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).is.an.Array();
    let count = _.size(srcNameList);
    Expect(count).is.above(1); // at.least(2);
    let clue = {
        name: name,
        src: _.toString(srcNameList)
    };
    if (!_.isUndefined(note)) {
        clue.note = note;
    }
    this.maybeListArray[count].push(clue);
    if (save) {
        this.maybeListArray[count].save(this.getMaybeFilename(count));
    }
    return true;
};

//

ClueManager.prototype.addRejectCombos = function (clueList, clueCount) {
    clueList.forEach(clue => {
        let srcNameList = clue.src.split(',');
        if (_.size(srcNameList) !== clueCount) {
            this.log(`WARNING! reject word count mismatch` +
                     `, expected {clueCount}, actual ${_.size(srcNameList)}, ${srcNameList}`);
        }
        this.addRejectSource(srcNameList);
    });
    return this;
};

//

ClueManager.prototype.saveRejects = function (counts) {
    if (_.isNumber(counts)) {
        counts = [ counts ];
    }
    Expect(counts).is.an.Array();
    counts.forEach(count => {
        this.rejectListArray[count].save(this.getRejectFilename(count));
    });
};

//

ClueManager.prototype.addReject = function (srcNameList, save = false) {
    if (_.isString(srcNameList)) {
        srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).is.an.Array();
    let count = _.size(srcNameList);
    Expect(count).is.above(1); // at.least(2);
    if (this.addRejectSource(srcNameList)) {
        this.rejectListArray[count].push({
            src: _.toString(srcNameList)
        });
        if (save) {
            this.rejectListArray[count].save(this.getRejectFilename(count));
        }
        return true;
    }
    return false;
};

//

ClueManager.prototype.addRejectSource = function (srcNameList) {
    if (_.isString(srcNameList)) {
        srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).is.an.Array().and.not.empty();
    srcNameList.sort();
    this.log('addRejectSource: ' + srcNameList);

    if (this.isKnownSource(srcNameList.toString())) {
        console.log('WARNING! not rejecting known source, ' + srcNameList);
        return false;
    }
    if (this.isRejectSource(srcNameList)) {
        console.log('WARNING! duplicate reject source, ' + srcNameList);
        // i had this return false commented out for some reason,
        // but it should be here.
        return false;
    }
    this.rejectSourceMap[srcNameList.toString()] = true;
    return true;
};

// source is string containing sorted, comma-separated clues

ClueManager.prototype.isKnownSource = function (source, count = 0) {
    Expect(source).is.a.String();
    Expect(count).is.a.Number();

    // check for supplied count
    if (count > 0) {
        return _.has(this.knownSourceMapArray[count], source);
    }
    // check for all counts
    return this.knownSourceMapArray.some(srcMap => _.has(srcMap, source));
};

// source: csv string or array of strings

ClueManager.prototype.isRejectSource = function (source) {
    if (!_.isString(source) && !_.isArray(source)) {
        throw new Error('bad source: ' + source);
    }
    return _.has(this.rejectSourceMap, source.toString());
};

//

ClueManager.prototype.getCountListForName = function (name) {
    let countList = [];
    for (const [index, clueMap] of this.knownClueMapArray.entries()) {
        if (_.has(clueMap, name)) {
            countList.push(index);
        }
    };
    return countList;
};

//

ClueManager.prototype.getSrcListForNc = function (nc) {
    let clueMap = this.knownClueMapArray[nc.count];
    Expect(_.has(clueMap, nc.name)).is.true();
    return clueMap[nc.name];
};

//

ClueManager.prototype.getSrcListMapForName = function (name) {
    let srcListMap = {};
    for (let index = 1; index < this.maxClues; ++index) {
        let srcList = this.knownClueMapArray[index][name];
        if (srcList) {
            srcListMap[index] = srcList;
        }
    }
    return srcListMap;
};

// TODO this is failing with ncList.length > 1

ClueManager.prototype.primaryNcListToNameSrcLists = function (ncList) {
    let log = 0 && (ncList.length > 1); // nameSrcLists.length > 1) {
    let srcLists = ncList.map(nc => this.primaryNcToSrcList(nc));
    let indexLists = srcLists.map(srcList => [...Array(srcList.length).keys()]);  // e.g. [ [ 0 ], [ 0, 1 ], [ 0 ], [ 0 ] ]
    let nameSrcLists = Peco.makeNew({
        listArray: indexLists,
        max:        999 // technically, clue-types[variety].max_clues
    })  .getCombinations()
	.map(indexList => indexList.map((value, index) => NameCount.makeNew(ncList[index].name, srcLists[index][value])));

    nameSrcLists = nameSrcLists.filter(nameSrcList => _.uniqBy(nameSrcList, NameCount.count).length === nameSrcList.length)
	.map(nameSrcList => _.sortBy(nameSrcList, NameCount.count));

    if (log) {
	console.log(`    ncList: ${ncList}`);
	console.log(`    nameSrcLists: ${nameSrcLists}`);
	console.log(`    uniq: ${_.uniqBy(nameSrcLists, _.toString)}`);
    }
    return _.uniqBy(nameSrcLists, _.toString);
}

//

ClueManager.prototype.primaryNcListToNameSrcSets = function (ncList) {
    let log = 0 && (ncList.length > 1); // nameSrcLists.length > 1) {
    let srcLists = ncList.map(nc => this.primaryNcToSrcList(nc));
    let indexLists = srcLists.map(srcList => [...Array(srcList.length).keys()]);  // e.g. [ [ 0 ], [ 0, 1 ], [ 0 ], [ 0 ] ]
    let nameSrcSets = Peco.makeNew({
        listArray: indexLists,
        max:       2 // 999 // technically, clue-types[variety].max_clues
    })  .getCombinations()
	.reduce((result, indexList) => {
	    let set = new Set();
	    //indexList.forEach((value, index) => set.add(NameCount.makeNew(ncList[index].name, srcLists[index][value])));
	    indexList.forEach((value, index) => set.add(NameCount.makeCanonicalName(ncList[index].name, srcLists[index][value])));
	    if (set.size === indexList.length) {
		result.push(set); // no duplicates
	    }
	    return result;
	}, []);

    if (log) {
	//console.log(`    ncList: ${ncList}`);
	//console.log(`    nameSrcLists: ${nameSrcLists}`);
	//console.log(`    uniq: ${_.uniqBy(nameSrcLists, _.toString)}`);
    }
    // TODO: put in some code to check for dupes, to see if we're actually seeing dupes. might slow down a little.
    
    //return _.uniqBy(nameSrcLists, _.toString);
    return nameSrcSets;
}

//

ClueManager.prototype.primaryNcToSrcList = function (nc) {
    if (nc.count !== 1) throw new Error(`nc.count must be 1 (${nc})`);
    const source = this.knownClueMapArray[1][nc.name];
    return _.isArray(source) ? source : [ source ];
}

//
/*
ClueManager.prototype.primaryNcToNameSrc = function (nc) {
    if (nc.count !== 1) throw new Error(`nc.count must be 1 (${nc})`);
    return NameCount.makeNew(nc.name, this.knownClueMapArray[1][nc.name]);
};
*/
//

ClueManager.prototype.makeSrcNameListArray = function (nc) {
    let srcNameListArray = [];
    this.getSrcListForNc(nc).forEach(src => {
        srcNameListArray.push(src.split(','));
    });
    return srcNameListArray;
};

// args:
//  sum:     args.sum,
//  max:     args.max,
//  require: args.require
//
// A "clueSourceList" is a list (array) where each element is a
// cluelist, such as [clues1,clues1,clues2].
//
// Given a sum, such as 3, generate an array of lists of addends that
// that add up to that sum, such as [ [1, 2], [2, 1] ], and return an
// array of lists of clueLists of the corresponding clue counts, such
// as [ [clues1, clues2], [clues2, clues1] ].

ClueManager.prototype.getClueSourceListArray = function (args) {
    Log.info(`++clueSrcListArray` +
             `, sum: ${args.sum}, max: ${args.max}, require: ${args.require}`);

    let clueCountListArray = Peco.makeNew({
        sum:     args.sum,
        max:     args.max,
        require: args.require
    }).getCombinations(); 

    Debug(`clueCountListArray(${clueCountListArray.length}): ${Stringify(clueCountListArray)}`);

    let clueSourceListArray = [];
    // TODO: .filter()
    clueCountListArray.forEach(clueCountList => {
        let clueSourceList = [];
        if (clueCountList.every(count => {
            // empty lists not allowed
            if (_.isEmpty(this.clueListArray[count])) {
                return false;
            }
            clueSourceList.push({ 
                list:  this.clueListArray[count],
                count: count
            });
            return true;
        }, this)) {
            clueSourceListArray.push(clueSourceList);
        }
    }, this);
    if (!clueSourceListArray.length) {
        // this happens, for example, with -c 3 -x 3 -q1,2,3
        // we require sources 1,2,3, and search for combos with
        // -max 1,2,3 of which both -max 1,2 will hit here because
        // we can't match the 3 required sources
        //console.log('WARNING! getClueSrcListArray empty!');
    }
    return clueSourceListArray;
};

//

ClueManager.prototype.filterAddends = function (addends, sizes) {
    let filtered = [];
    addends.forEach(list => {
        if (sizes.every(size => {
            return list.indexOf(Number(size)) > -1;
        })) {
            filtered.push(list);
        }
    });
    return filtered;
};

//

ClueManager.prototype.filter = function (srcCsvList, clueCount, map = {}) {
    let known = 0;
    let reject = 0;
    let duplicate = 0;
    srcCsvList.forEach(srcCsv => {
        if (this.isRejectSource(srcCsv)) {
            this.log(`isRejectSource(${clueCount}) ${srcCsv}`);
            ++reject;
        } else {
            if (this.isKnownSource(srcCsv, clueCount)) {
		this.log(`isKnownSource(${clueCount}) ${srcCsv}`);
		++known;
	    }
            if (_.has(map, srcCsv)) {
                this.log(`duplicate: ${srcCsv}`);
                ++duplicate;
            }
            map[srcCsv] = true;
        }
    });
    return {
        map:       map,
        known:     known,
        reject:    reject,
        duplicate: duplicate
    };
};

function singleEntry (nc, sources) {
    //console.log(`  singleEntry ${nc} sources: ${Stringify(sources)}`);
    return {
	results: [
	    {
		ncList: [ nc ],
		nameSrcList: [ NameCount.makeNew(nc.name, sources[0]) ]
	    }
	]
    };
};

//

ClueManager.prototype.getKnownSourceMapEntries = function (nc, andSources = false) {
    const clueMap = this.knownClueMapArray[nc.count];
    if (!clueMap) throw new Error(`No clueMap at ${nc.count}`);
    const sourcesList = clueMap[nc.name];
    if (!sourcesList) throw new Error(`No sourcesList at ${nc.name}`);
    // TODO: single entry, really? what if same primary clue name is used twice?
    if (nc.count === 1)  {
	return andSources ? [ { entry: singleEntry(nc, sourcesList) } ] : [ singleEntry(nc, sourcesList) ];
    }

    if (nc.toString() == 'washington:5') {
	console.log(`sourcesList: ${showStrList(sourcesList)}`);
    }

    return sourcesList.map(sources => sources.split(',').sort().toString()) // sort sources
	.map(sources => {
	    let entry = this.knownSourceMapArray[nc.count][sources];
	    console.log(` sources: ${sources}, entry: ${Stringify(entry)}`);//, entry2: ${Stringify(entry2)}`);
	    return andSources ? { entry, sources } : entry;
	}); 
};

//

ClueManager.prototype.getKnownClues = function (nameList) {
    if (_.isString(nameList)) {
        nameList = nameList.split(',');
    }
    Expect(nameList).is.an.Array();
    const sourceCsv = nameList.sort().toString();
    let nameClueMap = {};
    this.knownSourceMapArray.forEach(srcMap => {
        if (_.has(srcMap, sourceCsv)) {
            for (const clue of srcMap[sourceCsv].clues) {
                if (!_.has(nameClueMap, clue.name)) {
                    nameClueMap[clue.name] = [];
                }
                nameClueMap[clue.name].push(clue);
            }
        }
    });
    return nameClueMap;
};

//

ClueManager.prototype.getKnownClueNames = function (nameList) {
    return _.keys(this.getKnownClues(nameList));
};

//

ClueManager.prototype.getClueCountListArray = function (nameList) {
    Expect(nameList).is.not.empty();
    // each count list contains the clueMapArray indexes in which
    // each name appears
    let countListArray = Array(_.size(nameList)).fill().map(_ => []);
    for (let count = 1; count <= this.maxClues; ++count) {
        let map = this.knownClueMapArray[count];
        Expect(map).is.ok(); // I know this will fail when I move to synth clues
        nameList.forEach((name, index) => {
            if (_.has(map, name)) {
                countListArray[index].push(count);
            }
        });
    }
    return countListArray;
};

//

ClueManager.prototype.getValidCounts = function (nameList, countListArray) {
    if ((nameList.length === 1) || this.isRejectSource(nameList)) return [];

    let addCountSet = new Set();
    let known = false;
    let reject = false;
    Peco.makeNew({
        listArray: countListArray,
        max:       this.maxClues
    }).getCombinations().forEach(clueCountList => {
        Debug(`nameList: ${nameList}, clueCountList: ${clueCountList}`);
        let sum = clueCountList.reduce((a, b) => a + b);
        // why was I passing validateAll: true here, shouldn't a single
        // validation suffice?
        if (Validator.validateSources({
            sum:         sum,
            nameList:    nameList,
            count:       nameList.length,
            validateAll: false
        }).success) {
            addCountSet.add(sum);
        }
    });
    return Array.from(addCountSet);
};

//

ClueManager.prototype.getCountList = function (nameOrList) {
    return _.isString(nameOrList)
        ? this.getCountListForName(nameOrList)
        : this.getValidCounts(nameOrList, this.getClueCountListArray(nameOrList));
};

//

ClueManager.prototype.getPrimarySources = function () {
    let primarySources = [];
    let hash = {};
    for (const clue of this.clueListArray[1]) {
        if (_.has(hash, clue.src)) continue;
        primarySources.push(clue.src);
        hash[clue.src] = true;
    }
    return primarySources;
};

//

ClueManager.prototype.getInversePrimarySources = function (sources) {
    Expect(sources).is.an.Array();
    let inverseSources = [];
    for (const src of this.getPrimarySources()) {
        if (_.includes(sources, src)) continue;
        inverseSources.push(src);
    }
    return inverseSources;
};

//

ClueManager.prototype.addClueForCounts = function (countSet, name, src, options) {
    Expect(countSet).is.instanceof(Set);
    Expect(name).is.a.String();
    Expect(src).is.a.String();
    return Array.from(countSet).reduce((added, count) => {
        if (this.addClue(count, {
            name: name,
            src:  src
        }, options.save, true)) { // save, nothrow
            console.log(`${count}: added ${name}`);
	    added += 1;
        } else {
            console.log(`${count}: ${name} already present`);
	}
	return added;
    }, 0);
};

//

ClueManager.prototype.removeClueForCounts = function (countSet, name, src, options = {}) {
    Expect(countSet).is.instanceof(Set);
    Expect(name).is.a.String();
    Expect(src).is.a.String();
    let removed = 0;
    for (let count of countSet.keys()) {
        if (this.removeClue(count, {
            name: name,
            src:  src
        }, options.save, options.nothrow)) {
            Debug(`removed ${name}:${count}`);
            removed += 1;
        } else {
            // not sure this should ever happen. removeClue throws atm.
            Debug(`${name}:${count} not present`);
        }
    }
    return removed;
};

// each count list contains the clueMapArray indexes in which
// each name appears

ClueManager.prototype.getKnownClueIndexLists = function (nameList) {
    let countListArray = Array(_.size(nameList)).fill().map(_ => []);
    //Debug(countListArray);
    for (let count = 1; count <= this.maxClues; ++count) {
        const map = this.knownClueMapArray[count];
        if (!_.isUndefined(map)) {
            nameList.forEach((name, index) => {
                if (_.has(map, name)) {
                    countListArray[index].push(count);
                }
            });
        }
        else {
            console.log('missing known cluemap #' + count);
        }
    }
    // verify that all names were found
    nameList.forEach((name, index) => {
        Expect(countListArray[index]).is.ok();
    });
    return countListArray;
};

//
// args:
//   add=Name
//   remove=Name
//   reject
//   isKnown
//   isReject
//

ClueManager.prototype.addRemoveOrReject = function (args, nameList, countSet, options = {}) {
    let count = 0;
    if (args.add) {
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --add due to single source');
        } else if (args.isReject) {
            console.log('WARNING! cannot add known clue: already rejected, ' + nameList);
        } else {
            count = this.addClueForCounts(countSet, args.add, nameList.toString(), options);
        }
    } else if (args.remove) {
        Debug(`remove [${args.remove}] as ${nameList} from ${[...countSet.values()]}`);
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --remove due to single source');
        } else {
            let removeOptions = { save: options.save, nothrow: true };
            count = this.removeClueForCounts(countSet, args.remove, nameList.toString(), removeOptions);
        }
    } else if (args.reject) {
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --reject due to single source');
        } else if (args.isKnown) {
            console.log('WARNING! cannot add reject clue: already known, ' + nameList);
        } else if (this.addReject(nameList.toString(), true)) {
            console.log('updated');
        }
        else {
            console.log('update failed');
        }
    }
    return count;
};

ClueManager.prototype.getAllCountListCombosForNameList = function (nameList, max = this.maxClues) {
    const countListArray = this.getKnownClueIndexLists(nameList);
    Debug(countListArray);
    return Peco.makeNew({
        listArray: countListArray,
	max
    }).getCombinations();
}

ClueManager.prototype.buildNcListFromNameListAndCountList = function (nameList, countList) {
    return countList.map((count, index) => NameCount.makeNew(nameList[index], count));
}

ClueManager.prototype.buildNcListsFromNameListAndCountLists = function (nameList, countLists) {
    let ncLists = [];
    for (const countList of countLists) {
	ncLists.push(this.buildNcListFromNameListAndCountList(nameList, countList));
    }
    return ncLists;
}

ClueManager.prototype.buildNcListsFromNameList = function (nameList) {
    const countLists = this.getAllCountListCombosForNameList(nameList);
    if (_.isEmpty(countLists)) {
        return countLists;
    }
    return this.buildNcListsFromNameListAndCountLists(nameList, countLists);
}

ClueManager.prototype.getListOfPrimaryNameSrcLists = function (ncList) {
    // TODO: can use reduce() here too 
    let listOfPrimaryNameSrcLists = [];
    //console.log(`ncList: ${ncList}`);
    for (const nc of ncList) {
	console.log(`  nc: ${nc}`);
	let added = false;
	let entries;
	for (;;) {
	    entries = this.getKnownSourceMapEntries(nc, true);
	    if (!_.isArray(entries) || _.isEmpty(entries)) {
		console.log(`  explosion, nc: ${nc}, entries: ${Stringify(entries)}`);
		process.exit(-1);
	    }
	    if (added || entries[0].entry || nc.count === 1) break;
	    const sources = entries[0].sources;
	    
	    console.log(`adding nc: ${nc}, sources ${sources}, entries: ${Stringify(entries)}`);

	    const clue = { name: nc.name, src: sources };
	    if (nc.count > 1) {
		this.addCompoundClue(clue, nc.count, true);
	    } else {
		this.addPrimaryClueToMap(clue);
	    }
	    //
	    // TODO
	    //
	    // call addClue here too
	    //this.addClue(clue, nc.count)
	    added = true;
	}

	// verify that no other entries are undefined

	const primaryNameSrcLists = _.flatten(entries.map((item, index) => {
	    const entry = item.entry;
	    if (!entry || !entry.results || !_.isArray(entry.results) || _.isEmpty(entry.results)) {
		 // || _.isEmpty(item.sources)) {

		console.log(`  explosion2, nc: ${nc}, sources: ${item.sources}, ` +
			    `entries: ${Stringify(entries)}, entry: ${Stringify(entry)}`);
		if (!entry) console.log('entry null');
		else if (!entry.results) console.log('entry.results null');
		else if (!_.isArray(entry.results)) console.log('entry.results not array');
		else if (_.isEmpty(entry.results)) console.log('entry.results empty');
		process.exit(-1);
	    }
	    return entry.results.map(result => result.nameSrcList);
	}));
	primaryNameSrcLists.forEach(nameSrcList => console.log(`    nameSrcList: ${nameSrcList}`));
	listOfPrimaryNameSrcLists.push(primaryNameSrcLists);
    }
    return listOfPrimaryNameSrcLists;
}

ClueManager.prototype.origbuildListsOfPrimaryNameSrcLists = function (ncLists) {
    return ncLists.map(ncList => this.getListOfPrimaryNameSrcLists(ncList));
}

ClueManager.prototype.buildListsOfPrimaryNameSrcLists = function (ncLists) {
    return ncLists.map(ncList => {
	let result = this.getListOfPrimaryNameSrcLists(ncList);
	if (!result[0][0]) {
	    console.log(`ncList ${ncList}`);
	}
	return result;
    });
}

function getCompatiblePrimaryNameSrcList (listOfListOfPrimaryNameSrcLists) {
    //console.log(`${Stringify(listOfListOfPrimaryNameSrcLists)}`);
    const listArray = listOfListOfPrimaryNameSrcLists.map(listOfNameSrcLists => [...Array(listOfNameSrcLists.length).keys()]); // 0..nameSrcList.length
    let comboLists = Peco.makeNew({
        listArray,
        max: listOfListOfPrimaryNameSrcLists.reduce((sum, listOfNameSrcLists) => sum + listOfNameSrcLists.length, 0)
    }).getCombinations();

    return comboLists.some(comboList => {
	const nameSrcList = comboList.reduce((nameSrcList, comboListValue, comboListIndex) => {
	    let nsList = listOfListOfPrimaryNameSrcLists[comboListIndex][comboListValue];
	    //console.log(`nameSrcList: ${nameSrcList}, clValue ${comboListValue}, clIndex ${comboListIndex}, nsList: ${nsList}`);
	    if (nsList) {
		nameSrcList.push(...nsList);
	    } else {
		console.log(`no nsList for index: ${comboListIndex}, value: ${comboListValue}`);
	    }
	    return nameSrcList;
	}, []);
	const uniqLen = _.uniqBy(nameSrcList, NameCount.count).length;
	return (uniqLen === nameSrcList.length) ? nameSrcList : undefined;
    });
}

//

ClueManager.prototype.fast_getCountListArrays = function (nameCsv, options) {
    const nameList = nameCsv.split(',').sort();
    Debug(`fast_getCountListArrays for ${nameList}`);

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    const ncLists = this.buildNcListsFromNameList(nameList);
    if (_.isEmpty(ncLists)) {
	console.log(`No ncLists for ${nameList}`);
	return [];
    }
    return this.buildListsOfPrimaryNameSrcLists(ncLists).reduce((compatibleNcLists, listOfListOfPrimaryNameSrcLists, index) => {
	const compatibleNameSrcList = getCompatiblePrimaryNameSrcList(listOfListOfPrimaryNameSrcLists);
	console.log(`${ncLists[index]}  ${compatibleNameSrcList ? 'VALID' : 'invalid'}`);
	if (compatibleNameSrcList) {
	    compatibleNcLists.push(ncLists[index]);
	}
	return compatibleNcLists;
    }, []);
};

// Probably not the most unique function name possible.

let invalidHash = {};

ClueManager.prototype.getCountListArrays = function (nameCsv, options) {
    const validateAll = options.any ? false : true;
    const nameList = nameCsv.split(',').sort();
    Debug(`getValidCountLists for ${nameList}`);

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    const resultList = this.getAllCountListCombosForNameList(nameList);
    if (_.isEmpty(resultList)) {
        console.log(`No matches for ${nameList}`);
        return null;
    }

    let addRemoveSet;
    if (options.add || options.remove) {
        addRemoveSet = new Set();
    }
    let valid = [];
    let known = [];
    let rejects = [];
    let clues = [];
    let invalid = [];

    //console.log(`size ${resultList.length}`);
    let totalElapsed = 0;

    for (const clueCountList of resultList) {
        const sum = clueCountList.reduce((a, b) => a + b);
	//console.log(`sum: ${sum}, list: ${clueCountList}`);
	const start = new Date();
	//console.log(`${nameList}`);

	let x =_.uniqBy(clueCountList, _.toNumber);
	let ncListStr = clueCountList.map((count, index) => NameCount.makeNew(nameList[index], count)).toString();
	let result;
	result = invalidHash[ncListStr];
	if (!result) {
            result = Validator.validateSources({
		sum:         sum,
		nameList:    nameList,
		count:       nameList.length,
		require:     x, //clueCountList
		validateAll
            });
	}
	const elapsed = new Duration(start, new Date()).milliseconds;
	totalElapsed += elapsed;
	invalidHash[ncListStr] = result;

	//console.log(`validate: ${PrettyMs(elapsed)}, total: ${PrettyMs(totalElapsed)}`);
        
        if (!result.success) {
	    //console.log(`invalid: ${nameList}  CL ${clueCountList}  x ${x} sum ${sum}  validateAll=${validateAll}`);
            invalid.push(clueCountList);
        } else if (this.isRejectSource(nameList)) {
            rejects.push(clueCountList);
        } else if (nameList.length === 1) {
	    //console.log('hereLen1');
            let name = nameList[0];
            let nameSrcList = this.clueListArray[sum]
                    .filter(clue => clue.name === name)
                    .map(clue => clue.src);
            if (nameSrcList.length > 0) {
                //let clueNameList = this.clueListArray[sum].map(clue => clue.name);
                //if (clueNameList.includes(name)) {
                //
                
                /*
                 this.clueListArray[sum].forEach(clue => {
                 if (clue.name === name) {
                 clueSrcList.push(`"${clue.src}"`);
                 }
                 });
                 */
                clues.push({ countList: clueCountList, nameList: nameSrcList });
            }
        } else {
            let any = this.knownSourceMapArray[sum][nameList];
            if (any) {
                known.push({ countList: clueCountList, nameList: any.clues.map(clue => clue.name) });
            } else {
                valid.push(clueCountList);
            }
	    //console.log('hereValid1');
            if (options.add || options.remove) {
		//console.log('hereAdd1');
                addRemoveSet.add(sum);
            }
        }
    }
    //console.log('--getCountList');
    return { valid, known, rejects, invalid, clues, addRemoveSet };
};

