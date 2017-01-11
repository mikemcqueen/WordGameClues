//
// CLUE_MANAGER.JS
//

'use strict';

// export a singleton

module.exports = exports = new ClueManager();

var Fs             = require('fs');

var ClueList       = require('./clue_list');
var Validator      = require('./validator');
var NameCount      = require('./name_count');
var Peco           = require('./peco');

//
//

function ClueManager() {
    this.clueListArray = [];
    this.rejectListArray = [];
    this.knownClueMapArray = [];     // map clue name to clue src
    this.knownSourceMapArray = [];   // map known source (incl. combo string) to true/false (currently)
    this.rejectSourceMapArray = []; // may need to make this a MapArray eventually

    this.rvsSuccessSeconds = 0;
    this.rvsFailDuration = 0;

    this.loaded = false;
    this.maxClues = 0;

    this.logging = true;
    this.logLevel = 0;
}

//
//

ClueManager.prototype.log = function(text) {
    var pad = '';
    var index;
    for (var index=0; index<this.logLevel; ++index) { pad += ' '; }
    console.log(pad + text);
}

// args:
//  known:    known filename base
//  reject:   reject filename base
//  max:      max clue file# to load
//  required: required clue file# to load
//

ClueManager.prototype.loadAllClues = function(args) {
    var count;
    var optional = false;
    var knownClueList;
    var rejectClueList;

    for (count = 1; count <= args.max; ++count) {
	optional = count > args.required;
	knownClueList = ClueList.makeFrom(
	    { 'filename' : args.known + count + '.json',
	      'optional' : optional
	    }
	);
	knownClueList.clueCount = count;
	this.clueListArray[count] = knownClueList;

	// this is stupid.
	if (count === 1) {
	    this.addKnownPrimaryClues(knownClueList);
	}
	else {
	    this.addKnownCompoundClues(knownClueList, count);
	    rejectClueList = null;
	    try {
		rejectClueList = ClueList.makeFrom({
		    'filename' : args.reject + count + '.json',
		    'optional' : optional
		});
	    }
	    catch (e) {
		if (this.logging) {
		    this.log('missing reject file: ' + 
			     args.reject + count + '.json');
		}
	    }
	    this.rejectSourceMapArray[count] = {};
	    if (rejectClueList ) {
		this.rejectListArray[count] = rejectClueList;
		this.addRejectCombos(rejectClueList, count);
	    }
	}
    }

    this.loaded = true;
    this.maxClues = args.max;

    return this;
}

//
//

ClueManager.prototype.loadIgnoredClues = function() {
    var ignoredList;

    try {
	ignoredList = JSON.parse(Fs.readFileSync("ignored.json", 'utf8'));
    }
    catch (e) {
	return;
    }

    ignoredList.forEach(function(clue) {
	this.ignoredClueMap[clue] = true;
    }, this);

    return this;
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
//

ClueManager.prototype.addKnownCompoundClues = function(clueList, clueCount) {
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

	if ((clueCount > 1) && !srcMap[srcKey]) { // or .contains(clue);
	    if (this.logging) {
		this.log('############ validating Known Combo: ' + srcKey);
	    }
	    
	    if (!Validator.validateSources({
		sum:      clueCount,
		nameList: srcNameList,
		count:    srcNameList.length
	    })) {
		throw new Error('Known validate sources failed' +
				', count(' + clueCount + ') ' +
				clue.name + ' : ' + srcKey);
	    }
	    srcMap[srcKey] = true; // or .push(clue);
	}
	this.addKnownClue(clueCount, clue.name, srcKey);
    }, this);

    return this;
}

//
//

ClueManager.prototype.addKnownClue = function(count, name, source) {
    var clueMap = this.knownClueMapArray[count];

    if (!count || !name || !source) {
	throw new Error('missing argument, count: ' + count +
			', name: ' + name +
			', source: '+ source);
    }

    if (!clueMap[name]) {
	clueMap[name] = [ source ];
    }
    else if (clueMap[name].indexOf(source) == -1) {

	if (this.logging) {
	    this.log('addKnownClue(' + count + ') ' +
		     name + ' : ' + source);
	}

	clueMap[name].push(source);
    }
    else {
	throw new Error('duplicate clue name/source' + 
			'(' + count + ') ' +
			name + ' : ' + source);
    }
    return this;
}

//
//

ClueManager.prototype.addRejectCombos = function(clueList, clueCount) {
    clueList.forEach(clue => {
	var srcNameList;

	if (clue.ignore) {
	    return; // continue;
	}

	srcNameList = clue.src.split(',');
	srcNameList.sort();

	if (this.isRejectSource(srcNameList.toString(), clueCount)) {
	    throw new Error('Duplicate reject source, count(' + clueCount + ')' +
			    ' srcNames: ' + srcNameList);
	}
	else {
	    if (this.logging) {
		this.log('############ validating Reject Combo : ' + srcNameList);
	    }

	    if (!Validator.validateSources({
		sum:      clueCount,
		nameList: srcNameList,
		count:    srcNameList.length
	    })) {
		throw new Error('Reject validate sources failed, count(' + clueCount + ')' +
				' srcNames: ' + srcNameList);
	    }
	}

	if (this.logging) {
	    this.log('addRejectCombo: ' + srcNameList);
	}

	this.rejectSourceMapArray[clueCount][srcNameList] = true;
    }, this);

    return this;
}

// source is string containing sorted, comma-separated clues
//

ClueManager.prototype.isKnownSource = function(source, count) {
    if (!source || !count) {
	throw new Error('missing args, source: ' + source +
			', count: ' + count);
    }
    return this.knownSourceMapArray[count][source];
}

// source is string containing sorted, comma-separated clues
//

ClueManager.prototype.isRejectSource = function(source, count) {
    if (!source || !count) {
	throw new Error('missing args, source: ' + source +
			', count: ' + count);
    }
    return this.rejectSourceMapArray[count][source];
}

//
//

ClueManager.prototype.isIgnoredClue = function(clue) {
    return this.ignoredClueMap[clue];
}


//
//

ClueManager.prototype.getCountListForName = function(name) {
    var countList = [];
    var count;
    
    for (count in this.knownClueMapArray) {
	if (this.knownClueMapArray[count][name]) {
	    countList.push(count);
	}
    }
    return countList;
}

//
//

ClueManager.prototype.getSrcListForNc = function(nc) {
    var srcList;
    srcList = this.knownClueMapArray[nc.count][nc.name];
    if (!srcList) {
	throw new Error('specified clue does not exist, ' + nc);
    }
    return srcList;
}

//
//

ClueManager.prototype.getSrcListMapForName = function(name) {
    var srcListMap;
    var srclist;
    var index;

    srcListMap = {};
    for (index = 1; index < this.maxClues; ++index) {
	srcList = this.knownClueMapArray[index][name];
	if (srcList) {
	    srcListMap[index] = srcList;
	}
    }
    return srcListMap;
}

//
//

ClueManager.prototype.makeSrcNameListArray = function(nc) {
    var srcNameListArray = [];

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
// Given a count, such as 3, retrieve the array of lists of addends
// that sum to that count, such as [1,2], and return an array of lists
// of clueLists of the specified clue counts, such as [clues1,clues2].

ClueManager.prototype.getClueSourceListArray = function(args) {
    var clueSourceListArray = [];

    (new Peco({
	sum:     args.sum,
	max:     args.max,
	require: args.require
    })).getCombinations().forEach(clueCountList => {
	var clueSourceList = [];

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
	throw new Error('clueSourceListArray empty!');
    }

    return clueSourceListArray;
}

//
//

ClueManager.prototype.filterAddends = function(addends, sizes) {
    var filtered = [];

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

ClueManager.prototype.filter = function(clueListArray, clueCount) {
    var index;
    var source;
    var known = 0;
    var reject = 0;

    for (index = 0; index < clueListArray.length; ++index) {
	source = clueListArray[index].makeKey();
	if (this.isKnownSource(source, clueCount)) {
	    if (this.logging) {
		this.log('isKnownSource(' + clueCount + ') ' + source);
	    }
	    ++known;
	    delete clueListArray[index];
	}
	else if (this.isRejectSource(source, clueCount)) {
	    if (this.logging) {
		this.log('isRejectSource(' + clueCount + ') ' + source);
	    }
	    ++reject;
	    delete clueListArray[index];
	}
    }
    return {
	'array' : clueListArray,
	'known' : known,
	'reject': reject
    };
}

