//
// CLUE_MANAGER.JS
//

'use strict';

// export a singleton

module.exports = exports = new ClueManager();

const _              = require('lodash');
const Path           = require('path');
const expect         = require('chai').expect;

const ClueList       = require('./clue_list');
const Validator      = require('./validator');
const NameCount      = require('./name_count');
const Peco           = require('./peco');

//

const DATA_DIR              =  Path.dirname(module.filename) + '/../data/';

const MAX_SYNTH_CLUE_COUNT  = 25;
const REQ_SYNTH_CLUE_COUNT  = 11; // should make this 12 soon
const MAX_META_CLUE_COUNT   = 9;
const REQ_META_CLUE_COUNT   = 9;

//
//

function ClueManager() {
    this.clueListArray = [];         // the JSON clue files in an array
    this.rejectListArray = [];       // the JSON reject files in an array
    this.knownClueMapArray = [];     // map clue name to clue src
    this.knownSourceMapArray = [];   // map known source to list of clues
    this.rejectSourceMap = {};       // map reject source to true/false (currently)

    this.rvsSuccessSeconds = 0;
    this.rvsFailDuration = 0;

    this.loaded = false;
    this.maxClues = 0;

    this.logging = false;
    this.logLevel = 0;
}

//
//

ClueManager.prototype.log = function(text) {
    var pad = '';
    var index;
    if (this.logging) {
	for (index=0; index<this.logLevel; ++index) { pad += ' '; }
	console.log(pad + text);
    }
}

// args:
//  baseDir:  base directory (meta, synth)
//  ignoreErrors:
//  validateAll:
//
ClueManager.prototype.loadAllClues = function(args) {
    var count;
    var optional = false;
    var knownClueList;
    //var max;

    this.dir = DATA_DIR + args.baseDir;

    if (args.ignoreErrors) {
	this.ignoreLoadErrors = true;
    }
    //console.log('module.filename: ' + Path.dirname(module.filename));

    let result = this.getMaxRequired(args.baseDir);
    this.maxClues = result.max; // need to set this before loading, used by validator

    for (count = 1; count <= result.max; ++count) {
	optional = count > result.required;
	knownClueList = ClueList.makeFrom(
	    { 'filename' : this.getKnownFilename(count),
	      'optional' : optional
	    }
	);
	knownClueList.clueCount = count;
	this.clueListArray[count] = knownClueList;

	if (count === 1) {
	    this.addKnownPrimaryClues(knownClueList);
	}
	else {
	    this.addKnownCompoundClues(knownClueList, count, args.validateAll);
	}
    }

    for (count = 2; count <= result.max; ++count) {
	let rejectClueList = null;
	try {
	    rejectClueList = ClueList.makeFrom({
		'filename' : this.getRejectFilename(count),
		'optional' : optional
	    });
	}
	catch (e) {
	    this.log('WARNING! missing reject file: ' + 
		     this.getRejectFilename(count));
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
//
ClueManager.prototype.getMaxRequired = function(baseDir) {
    var max =      MAX_META_CLUE_COUNT;
    var required = REQ_META_CLUE_COUNT;
    
    switch (baseDir) {
    case 'synth':
	max =      MAX_SYNTH_CLUE_COUNT;
	required = REQ_SYNTH_CLUE_COUNT;
	break;
    case 'meta':
	break;
    default:
	// default to meta
	this.log('defaulting to --meta clue counts for baseDir ' + baseDir);
	break;
    }
    return {
	max : max,
	required: required
    };
}

//
//
ClueManager.prototype.addKnownPrimaryClues = function(clueList) {
    var clueCount = 1;
    var clueMap;
    var sourceArray;
    
    clueMap = this.knownClueMapArray[clueCount] = {};
    clueList.forEach(clue => {
	if (clue.ignore) {
	    return; // continue
	}
	if (!clueMap[clue.name]) {
	    clueMap[clue.name] = [ clue.src ];
	}
	else {
	    clueMap[clue.name].push(clue.src);
	}
    }, this);

    return this;
}

//

ClueManager.prototype.getKnownFilename = function(count) {
    return this.dir + '/clues' + count + '.json';
}

//

ClueManager.prototype.getRejectFilename = function(count) {
    return this.dir + '/rejects' + count + '.json';
}

//
//

ClueManager.prototype.addKnownCompoundClues = function(clueList, clueCount, validateAll) {
    var srcMap;

    if (!this.knownClueMapArray[clueCount]) {
	this.knownClueMapArray[clueCount] = {};
    }
    else {
	// TODO: remove this and "if" check block above
	throw new Error('really? how odd');
    }
	
    if ((clueCount > 1) && !this.knownSourceMapArray[clueCount]) {
	this.knownSourceMapArray[clueCount] = {};
    }
    else {
	// TODO: remove this and "if" check block above
	throw new Error('really? how odd #2');
    }
	
    srcMap = this.knownSourceMapArray[clueCount];
    clueList.forEach(clue => {
	var srcNameList;
	var srcKey;

	if (clue.ignore) {
	    return; // continue
	}

	srcNameList = clue.src.split(',');
	srcNameList.sort();
	srcKey = srcNameList.toString();

	if (clueCount > 1) {
	    // new sources need to be validated
	    if (!_.has(srcMap, srcKey)) {
		if (this.logging) {
		    this.log('############ validating Known Combo: ' + srcKey);
		}
		
		if (!Validator.validateSources({
		    sum:         clueCount,
		    nameList:    srcNameList,
		    count:       srcNameList.length,
		    validateAll: validateAll
		}).success) {
		    if (!this.ignoreLoadErrors) {
			throw new Error('Known validate sources failed' +
					', count(' + clueCount + ') ' +
					clue.name + ' : ' + srcKey);
		    }
		}
		srcMap[srcKey] = [ clue ];
	    }
	    else {
		srcMap[srcKey].push(clue);
	    }
	}
	this.addKnownClue(clueCount, clue.name, srcKey);
    }, this);

    return this;
}

//
//

ClueManager.prototype.addClue = function(count, clue, save) {
    if (this.addKnownClue(count, clue.name, clue.src, true)) {
	this.clueListArray[count].push(clue);
	if (save) {
	    this.clueListArray[count].save(this.getKnownFilename(count));
	}
	return true;
    }
    return false;
}

//
//

ClueManager.prototype.addKnownClue = function(count, name, source, noThrow) {
    expect(count).to.be.a('number');
    expect(name).to.be.a('string');
    expect(source).to.be.a('string');

    let clueMap = this.knownClueMapArray[count];
    if (!clueMap[name]) {
	clueMap[name] = [ source ];
    }
    else if (!clueMap[name].includes(source)) {
	if (this.logging) {
	    this.log('addKnownClue(' + count + ') ' +
		     name + ' : ' + source);
	}
	clueMap[name].push(source);
    }
    else if (!noThrow) {
	throw new Error('duplicate clue name/source' + 
			'(' + count + ') ' +
			name + ' : ' + source);
    }
    else {
	return false;
    }
    return true;
}

//
//

ClueManager.prototype.addRejectCombos = function(clueList, clueCount) {
    clueList.forEach(clue => {
	var srcNameList;
	srcNameList = clue.src.split(',');

	if (_.size(srcNameList) !== clueCount) {
	    this.log('WARNING! word count mismatch' +
		     ', expected ' + clueCount +
		     ', actual ' + _.size(srcNameList) +
		     ', ' + srcNameList);
	}
	this.addRejectSource(srcNameList);
    });
    return this;
}

//

ClueManager.prototype.addReject = function(srcNameList, save) {
    let count = _.size(srcNameList);
    expect(count).to.be.at.least(2);
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

ClueManager.prototype.addRejectSource = function(srcNameList) {
    expect(srcNameList).to.be.an('array');

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
	return false;
    }
    this.rejectSourceMap[srcNameList.toString()] = true;
    return true;
}

// source is string containing sorted, comma-separated clues
//
ClueManager.prototype.isKnownSource = function(source, count = 0) {
    expect(source).to.be.a('string');
    expect(count).to.be.a('number');

    // check for supplied count
    if (count > 0) {
	return !_.isUndefined(this.knownSourceMapArray[count][source]);
    }

    // check for all counts
    return this.knownSourceMapArray.some(srcMap => {
	return !_.isUndefined(srcMap[source]);
    });
}

// source: csv
// NOTE: works with string or array of strings
//
ClueManager.prototype.isRejectSource = function(source) {
    if (!_.isString(source) && !_.isArray(source)) {
	throw new Error('bad source: ' + source);
    }
    return this.rejectSourceMap[source.toString()];
}

//
//

ClueManager.prototype.getCountListForName = function(name) {
    var countList = [];
    this.knownClueMapArray.forEach(clueMap => {
	if (_.has(clueMap, name)) {
	    countList.push(count);
	}
    });
    return countList;
}

//
//

ClueManager.prototype.getSrcListForNc = function(nc) {
    let srcList = this.knownClueMapArray[nc.count][nc.name];
    if (!srcList) {
	throw new Error('specified clue does not exist, ' + nc);
    }
    return srcList;
}

//
//

ClueManager.prototype.getSrcListMapForName = function(name) {
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

ClueManager.prototype.makeSrcNameListArray = function(nc) {
    let srcNameListArray = [];

//    console.log(nc.toJSON());
//    console.log(this.getSrcListForNc(nc));
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

ClueManager.prototype.getClueSourceListArray = function(args) {
    let clueSourceListArray = [];

    if (this.logging) {
	this.log('++clueSrcListArray' +
		 ', sum: ' + args.sum +
		 ', max: ' + args.max +
		 ', require: ' + args.require);
    }
    Peco.makeNew({
	sum:     args.sum,
	max:     args.max,
	require: args.require
    }).getCombinations().forEach(clueCountList => {
	let clueSourceList = [];

	if (clueCountList.every(count => {
	    // empty lists not allowed
	    if (!this.clueListArray[count].length) {
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

ClueManager.prototype.filterAddends = function(addends, sizes) {
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

ClueManager.prototype.filter = function(srcCsvList, clueCount) {
    let known = 0;
    let reject = 0;
    let duplicate = 0;
    let map = {};

    // TODO: rather than deleting in array, build a new one?
    // TODO: clueListArray.
    srcCsvList.forEach(srcCsv => {
	if (this.isKnownSource(srcCsv, clueCount)) {
	    if (this.logging) {
		this.log('isKnownSource(' + clueCount + ') ' + srcCsv);
	    }
	    ++known;
	    //delete clueListArray[index];
	}
	else if (this.isRejectSource(srcCsv)) {
	    if (this.logging) {
		this.log('isRejectSource(' + clueCount + ') ' + srcCsv);
	    }
	    ++reject;
	    //delete clueListArray[index];
	}
	else {
	    if (map[srcCsv]) {
		console.log(`duplicate: ${srcCsv}`);
		++duplicate;
	    }
	    map[srcCsv] = true;
	}
    });
    return {
	set:       map,
	known:     known,
	reject:    reject,
	duplicate: duplicate
    };
}

//
//

ClueManager.prototype.getKnownClues = function(wordList) {
    let resultList = [];

    if (_.isArray(wordList)) {
	wordList = _.toString(wordList);
    }
    else if (!_.isString(wordList)) {
	throw new Error('invalid wordlist type, ' + (typeof wordList));
    }
    this.knownSourceMapArray.forEach(srcMap => {
	var clueList = srcMap[wordList];
	if (clueList) {
	    // TODO: something like: _.concat(resultList, _.extract(clueList, 'name'));
	    clueList.forEach(clue => {
		resultList.push(clue.name);
	    });
	}
    });
    return resultList;
}

