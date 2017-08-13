//
// clue-manager.js
//

'use strict';

// export a singleton

module.exports = exports = new ClueManager();

const _              = require('lodash');
const ClueList       = require('./clue-list');
const Clues          = require('./clue-types');
const Expect         = require('chai').expect;
const NameCount      = require('./name-count');
const Path           = require('path');
const Peco           = require('./peco');
const Validator      = require('./validator');

//

const DATA_DIR              =  Path.normalize(`${Path.dirname(module.filename)}/../data/`);
const REJECTS_DIR           = 'rejects';

//
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

    this.logging = false;
//    this.logging = true;

    this.logLevel = 0;
}

//
//

ClueManager.prototype.log = function (text) {
    let pad = '';
    let index;
    if (this.logging) {
	for (let index = 0; index < this.logLevel; index += 1) { pad += ' '; }
	console.log(pad + text);
    }
}

//

ClueManager.prototype.loadClueList = function (count, options = {}) {
    return ClueList.makeFrom({
	filename : this.getKnownFilename(count, options.dir)
    });
}

//

ClueManager.prototype.saveClueList = function (list, count, options = {}) {
    list.save(this.getKnownFilename(count, options.dir));
}

// args:
//  baseDir:  base directory (meta, synth)
//  ignoreErrors:
//  validateAll:
//
ClueManager.prototype.loadAllClues = function (args) {
    this.dir = `${DATA_DIR}${args.clues.baseDir}`;
    if (args.ignoreErrors) {
	this.ignoreLoadErrors = true;
    }
    this.maxClues = args.clues.MAX_CLUE_COUNT;

    for (let count = 1; count <= this.maxClues; ++count) {
	let knownClueList = this.loadClueList(count);
	this.clueListArray[count] = knownClueList;
	if (count === 1) {
	    this.addKnownPrimaryClues(knownClueList);
	}
	else {
	    this.addKnownCompoundClues(knownClueList, count, args.validateAll);
	}
    }

    for (let count = 2; count <= this.maxClues; ++count) {
	let rejectClueList = null;
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
}

//

ClueManager.prototype.addKnownPrimaryClues = function (clueList) {
    const count = 1;
    let clueMap = this.knownClueMapArray[count] = {};
    clueList.forEach(clue => {
	if (clue.ignore) {
	    return; // continue
	}
	if (!_.has(clueMap, clue.name)) {
	    clueMap[clue.name] = [];
	}
	clueMap[clue.name].push(clue.src);
    });
    return this;
}

//

ClueManager.prototype.getKnownFilename = function (count, dir = undefined) {
    return Path.format({
	dir:  _.isUndefined(dir) ? this.dir : `${DATA_DIR}${dir}`,
	base: `clues${count}.json`
    });
}

//

ClueManager.prototype.getRejectFilename = function (count) {
    return Path.format({
	dir:  `${DATA_DIR}${REJECTS_DIR}`,
	base: `rejects${count}.json`
    });
}

//
//

ClueManager.prototype.addKnownCompoundClues = function (clueList, clueCount, validateAll) {
    // so this is currently only callable once per clueCount.
    Expect(this.knownClueMapArray[clueCount]).to.be.undefined;
    this.knownClueMapArray[clueCount] = {};
    if (clueCount > 1) {
	Expect(this.knownSourceMapArray[clueCount]).to.be.undefined;
	this.knownSourceMapArray[clueCount] = {};
    }
    let srcMap = this.knownSourceMapArray[clueCount];
    clueList.forEach(clue => {
	if (clue.ignore) {
	    return; // continue
	}
	let srcNameList = clue.src.split(',').sort();
	let srcKey = srcNameList.toString();
	if (clueCount > 1) {
	    // new sources need to be validated
	    if (!_.has(srcMap, srcKey)) {
		if (this.logging) {
		    this.log('############ validating Known Combo: ' + srcKey);
		}
		let vsResult = Validator.validateSources({
		    sum:         clueCount,
		    nameList:    srcNameList,
		    count:       srcNameList.length,
		    validateAll: validateAll
		});
		if (!this.ignoreLoadErrors) {
		    Expect(vsResult.success, 'vsResult').to.be.true;
		}
		srcMap[srcKey] = [];
	    }
	    srcMap[srcKey].push(clue);
	}
	this.addKnownClue(clueCount, clue.name, srcKey);
    }, this);
    return this;
}

//
//

ClueManager.prototype.addKnownClue = function (count, name, source, nothrow) {
    Expect(count).to.be.a('number');
    Expect(name).to.be.a('string');
    Expect(source).to.be.a('string');
    let clueMap = this.knownClueMapArray[count];
    if (!_.has(clueMap, name)) {
	clueMap[name] = [ source ];
    } else if (!clueMap[name].includes(source)) {
	if (this.logging) {
	    this.log(`clueMap[${name}] = ${clueMap[name]}`);
	    this.log('addKnownClue(' + count + ') ' +
		     name + ' : ' + source);
	}
	clueMap[name].push(source);
    } else {
	if (nothrow) return false;
	throw new Error('duplicate clue name/source' + 
			'(' + count + ') ' +
			name + ' : ' + source);
    }
    return true;
}

//
//

ClueManager.prototype.saveClues = function (counts) {
    if (_.isNumber(counts)) {
	counts = [ counts ];
    }
    Expect(counts).to.be.an('array');
    for (const count of counts) {
	this.saveClueList(this.clueListArray[count], count);
    }
}

//
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
}

//

ClueManager.prototype.addMaybe = function (name, srcNameList, note, save = false) {
    return false;
    if (_.isString(srcNameList)) {
	srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).to.be.an('array');
    let count = _.size(srcNameList);
    Expect(count).to.be.at.least(2);
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
}


//
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
}

//
//

ClueManager.prototype.saveRejects = function (counts) {
    if (_.isNumber(counts)) {
	counts = [ counts ];
    }
    Expect(counts).to.be.an('array');
    counts.forEach(count => {
	this.rejectListArray[count].save(this.getRejectFilename(count));
    });
}

//

ClueManager.prototype.addReject = function (srcNameList, save = false) {
    if (_.isString(srcNameList)) {
	srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).to.be.an('array');
    let count = _.size(srcNameList);
    Expect(count).to.be.at.least(2);
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
}

//

ClueManager.prototype.addRejectSource = function (srcNameList) {
    if (_.isString(srcNameList)) {
	srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).to.be.an('array').that.is.not.empty;
    srcNameList.sort();
    if (this.logging) {
	this.log('addRejectSource: ' + srcNameList);
    }

    if (this.isKnownSource(srcNameList.toString())) {
	console.log('WARNING! not rejecting known source, ' + srcNameList);
	return false;
    }
    if (this.isRejectSource(srcNameList)) {
	console.log('WARNING! duplicate reject source, ' + srcNameList);
	//return false;
    }
    this.rejectSourceMap[srcNameList.toString()] = true;
    return true;
}

// source is string containing sorted, comma-separated clues
//
ClueManager.prototype.isKnownSource = function (source, count = 0) {
    Expect(source).to.be.a('string');
    Expect(count).to.be.a('number');

    // check for supplied count
    if (count > 0) {
	return _.has(this.knownSourceMapArray[count], source);
    }

    // check for all counts
    return this.knownSourceMapArray.some(srcMap => _.has(srcMap, source));
}

// source: csv string or array of strings
//
ClueManager.prototype.isRejectSource = function (source) {
    if (!_.isString(source) && !_.isArray(source)) {
	throw new Error('bad source: ' + source);
    }
    return _.has(this.rejectSourceMap, source.toString());
}

//
//

ClueManager.prototype.getCountListForName = function (name) {
    let countList = [];
    for (const [index, clueMap] of this.knownClueMapArray.entries()) {
	if (_.has(clueMap, name)) {
	    countList.push(index);
	}
    };
    return countList;
}

//
//

ClueManager.prototype.getSrcListForNc = function (nc) {
    let clueMap = this.knownClueMapArray[nc.count];
    Expect(clueMap, `${nc}`).to.have.property(nc.name);
    return clueMap[nc.name];
}

//
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
}

//
//

ClueManager.prototype.makeSrcNameListArray = function (nc) {
    let srcNameListArray = [];
    this.getSrcListForNc(nc).forEach(src => {
	srcNameListArray.push(src.split(','));
    });
    return srcNameListArray;
}

//
// args:
//  sum:     args.sum,
//  max:     args.max,
//  require: args.require
//
// A "clueSourceList" is a list (array) where each element is a
// cluelist, such as [clues1,clues1,clues2].
//
// Given a sum, such as 3, retrieve the array of lists of addends
// that add up to that sum, such as [1,2], and return an array of lists
// of clueLists of the specified clue counts, such as [clues1,clues2].
//
ClueManager.prototype.getClueSourceListArray = function (args) {
    let clueSourceListArray = [];

    if (this.logging) {
	this.log(`++clueSrcListArray` +
		 `, sum: ${args.sum}, max: ${args.max}, require: ${args.require}`);
    }
    Peco.makeNew({
	sum:     args.sum,
	max:     args.max,
	require: args.require
    }).getCombinations().forEach(clueCountList => {
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
	// we can't match 3 numbers
	console.log('WARNING! getClueSrcListArray empty!');
    }
    return clueSourceListArray;
}

//
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
}

//
//

ClueManager.prototype.filter = function (srcCsvList, clueCount, map = {}) {
    let known = 0;
    let reject = 0;
    let duplicate = 0;
    srcCsvList.forEach(srcCsv => {
	if (this.isKnownSource(srcCsv, clueCount)) {
	    if (this.logging) this.log(`isKnownSource(${clueCount}) ${srcCsv}`);
	    ++known;
	} else if (this.isRejectSource(srcCsv)) {
	    if (this.logging) this.log(`isRejectSource(${clueCount}) ${srcCsv}`);
	    ++reject;
	} else {
	    if (_.has(map, srcCsv)) {
		if (this.logging) this.log(`duplicate: ${srcCsv}`);
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
}

//
//

ClueManager.prototype.getKnownClues = function (wordList) {
    if (_.isString(wordList)) {
	wordList = wordList.split(',');
    }
    Expect(wordList).to.be.an('array');
    wordList = wordList.sort().toString();
    let resultList = [];
    this.knownSourceMapArray.forEach(srcMap => {
	if (_.has(srcMap, wordList)) {
	    // srcMap[wordList] is a clueList
	    resultList.push(...srcMap[wordList].map(clue => clue.name));
	}
    });
    return resultList;
}

//

ClueManager.prototype.getClueCountListArray = function (nameList) {
    Expect(nameList.length).to.be.not.empty;
    // each count list contains the clueMapArray indexes in which
    // each name appears
    let countListArray = Array(_.size(nameList)).fill().map(() => []);
    for (let count = 1; count <= this.maxClues; ++count) {
	let map = this.knownClueMapArray[count];
	Expect(map).to.exist; // I know this will fail when I move to synth clues
	nameList.forEach((name, index) => {
	    if (_.has(map, name)) {
		countListArray[index].push(count);
	    }
	});
    }
    return countListArray;
}

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
	let sum = clueCountList.reduce((a, b) => a + b);
	if (Validator.validateSources({
	    sum:      sum,
	    nameList: nameList,
	    count:    nameList.length,
	    validateAll: true
	}).success) {
	    addCountSet.add(sum);
	}
    });
    return Array.from(addCountSet);
}

//

ClueManager.prototype.getCountList = function (nameOrList) {
    return _.isString(nameOrList)
	? this.getCountListForName(nameOrList)
	: this.getValidCounts(nameOrList, this.getClueCountListArray(nameOrList));
}

