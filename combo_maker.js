//
// COMBO_MAKER.JS
//

'use strict';

// export a singleton

module.exports = exports = new ComboMaker();

var Duration    = require('duration');

//

var ClueManager = require('./clue_manager');
var ClueList    = require('./clue_list');
var Validator   = require('./validator');
var NameCount   = require('./name_count');
var Peco        = require('./peco'); // use this at some point

//
//

function ComboMaker() {
    this.hash = {};

    this.logging = false;
}

//
// count:   # of primary clues to combine
// max:     max # of sources to use
// require: required clue counts, e.g. [3,5,8]
// use:     list of clue names, e.g. ['john:1','bob:5']
//
// A "clueSourceList" is a list (array) where each element is a
// cluelist, such as [clues1,clues1,clues2].
//

ComboMaker.prototype.makeCombos = function(args) {
    var clueSourceListArray;
//    var clueCount = args.count;
    var clueListArray = [];
    var successDuration = 0;
    var failDuration = 0;
    var nextDuration = 0;
    var useNcList;
    var require;

    this.nextSort = 0;
    this.nextDupeClue = 0;
    this.nextDupeSrc = 0;
    this.nextDupeCombo = 0;

    // TODO USE "validateArgs" 

    /*
    this.log('CM: sum=' + args.sum + ' max=' + args.max);
    */

    if (args.use) {
	require = args.require ? args.require : [];
	useNcList = this.buildUseNcList(args.use, require);
    }

    clueSourceListArray = ClueManager.getClueSourceListArray({
	sum:       args.sum,
	max:       args.max,
	require:   require
    });

    clueSourceListArray.forEach(clueSourceList => {
	var sourceIndexes = [];
	var clueNameList;
	var clueList;
//	var index;
	var clue;
	var result;
	var validateResult;
	var start;
	var duration;
	var ncMap;

	result = this.first(clueSourceList, sourceIndexes);
	if (!result) {
	    throw new Error('no valid combos');
	}

	for (; result; (function(thisArg) {
	    start = new Date();
	    result = thisArg.next(clueSourceList, sourceIndexes);
	    duration = new Duration(start, new Date());
	    nextDuration += duration.milliseconds;
	})(this)) {
	    ncMap = {}
	    clueNameList = [];
	    clueSourceList.forEach((clueSource, index) => {
		clue = clueSource.list[sourceIndexes[index]];
		clueNameList.push(clue.name);
		// TODO: I'm not sure what I'm doing here, but I don't think it's working
		if (useNcList) {
		    ncMap[clue.name] = NameCount.makeNew(clue.name, clueSource.count);
		}
	    });

	    if (useNcList) {
		// every name in useNcList must exist as a key in ncMap
		if (!useNcList.every(nc => {
		    return ncMap[nc.name];
		})) {
		    continue; // invalid combo
		}
	    }

	    start = new Date();
	    validateResult = Validator.validateSources({
		sum:       args.sum,
		count:     clueNameList.length,
		nameList:  clueNameList,
		require:   require
	    });
	    duration = new Duration(start, new Date());

	    if (validateResult) {
		successDuration += duration.milliseconds;

		clueList = ClueList.makeNew();
		clueSourceList.forEach((clueSource, index) => {
		    clue = clueSource.list[sourceIndexes[index]];
		    // this is a bid odd
		    clue.count = clueSource.count;
		    if (isNaN(clue.count)) {
			throw new Error('bad clue count');
		    }
		    clueList.push(clue);
		});
		clueListArray.push(clueList);
	    }
	    else {
		failDuration += duration.milliseconds;
	    }

	}
    }, this);

    console.log('success: ' + successDuration + 'ms' +
		', fail: ' + failDuration + 'ms' +
		', next: ' + nextDuration + 'ms' +
		', nextSort: ' + this.nextSort + 'ms');
    console.log('dupeClue(' + this.nextDupeClue + ') ' +
		'dupeSrc(' + this.nextDupeSrc + ') ' +
		'dupeCombo(' + this.nextDupeCombo + ') ');

    if (0) { this.displayCombos(clueListArray); }

    return clueListArray;
}

//
//

ComboMaker.prototype.buildUseNcList = function(nameList, requireList) {
    var ncList;

    ncList = [];
    nameList.forEach(name =>  {
	var nc = NameCount.makeNew(name);
	if (nc.count) {
	    if (!ClueManager.knownClueMapArray[nc.count][nc.name]) {
		throw new Error('specified clue does not exist, ' + nc);
	    }
	    if (requireList.indexOf(nc.count) == -1) {
		requireList.push(nc.count);
	    }
	}
	ncList.push(nc);
    });
    return ncList;
}

//
//

ComboMaker.prototype.hasUniqueClues = function(clueList) {
    var sourceMap = {};
    clueList.forEach(function(clue) {
	if (isNaN(clue.count)) {
	    throw new Error('bad clue count');
	}
	else if (clue.count > 1) {

	}
	else if (!this.testSetKey(sourceMap, clue.src)) {
	    return false;
	}
    }, this);

    return true;
}

//
//

ComboMaker.prototype.testSetKey = function(map, key) {
    if (!map[key]) {
	map[key] = true;
	return true;
    }
    return false;
}

//
//

ComboMaker.prototype.displaySourceListArray = function(sourceListArray) {
    console.log('-----\n');
    sourceListArray.forEach(function(sourceList) {
	sourceList.forEach(function(source) {
	    source.display();
	    console.log('');
	});
	console.log('-----\n');
    });
}

//
//

ComboMaker.prototype.first = function(clueSourceList, sourceIndexes) {
    var index;

    this.hash = {};
    for (index = 0; index < clueSourceList.length; ++index) {
	sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;

    return this.next(clueSourceList, sourceIndexes);
}

//
//

ComboMaker.prototype.next = function(clueSourceList, sourceIndexes) {
    var hash;
    var listIndex;
    var comboStr;
    var clue;
    var start;
    var count;
    var name;
    var src;

    for (;;) {
	if (!this.nextIndex(clueSourceList, sourceIndexes)) {
	    return false;
	}

	var comboList = [];
	if (!clueSourceList.every((clueSource, index) => {
	    clue = clueSource.list[sourceIndexes[index]];
	    if (clue.ignore || clue.skip) {
		return false;
	    }
	    comboList.push(clue.name + ':' + clue.src);
	    return true;
	})) {
	    continue;
	}
	comboStr = comboList.toString();

	hash = {};
	// skip combinations that have duplicate clues or sources
	if (!clueSourceList.every((clueSource, index) => {

	    // check for duplicate clues
	    clue = clueSource.list[sourceIndexes[index]];
	    count = clueSource.count;
	    name = NameCount.makeCanonicalName(clue.name, count);
	    if (name in hash) {
		this.nextDupeClue++;
		if (this.logging) {
		    this.log('skipping duplicate clue: ' + comboStr);
		}
		return false;
	    }
	    hash[name] = true;

	    // questionable whether this is "correct", but it's probably OK.
	    src =  NameCount.makeCanonicalName(clue.src, count);
	    if (src in hash) {
		this.nextDupeSrc++;
		if (this.logging) {
		    this.log('skipping duplicate source: ' + comboStr);
		}
		return false;
	    }
	    hash[src] = true;

	    return true;
	})) {
	    continue;
	}

	// skip duplicate combinations
	start = new Date();
	comboList.sort();
	this.nextSort += (new Duration(start, new Date())).milliseconds;
	comboStr = comboList.toString();

	if (comboStr in this.hash) {
	    this.nextDupeCombo++;
	    if (this.logging) {
		this.log('skipping duplicate combo: ' + comboStr);
	    }
	    continue;
	}
	this.hash[comboStr] = true;

	return true; // success
    }
}

//
//

ComboMaker.prototype.nextIndex = function(clueSourceList, sourceIndexes) {
    var index = sourceIndexes.length - 1;

    // increment last index
    ++sourceIndexes[index];

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    while (sourceIndexes[index] == clueSourceList[index].list.length) {
	sourceIndexes[index] = 0;
	--index;
	if (index < 0) {
	    return false;
	}
	++sourceIndexes[index];
    }
    return true;
}

//
//

ComboMaker.prototype.displayCombos = function(clueListArray) {
    console.log('\n-----\n');
    var count = 0;
    clueListArray.forEach(function(clueList) {
	clueList.display();
	++count;
    });
    console.log('total = ' + count);
}

//
//

ComboMaker.prototype.clueListToString = function(clueList) {
    var str = '';

    clueList.forEach(function(clue) {
	if (str.length > 0) {
	    str += ' ';
	}
	str += clue.name;
	if (clue.src) {
	    str += ':' + clue.src;
	}
    });

    return str;
}

//
//

ComboMaker.prototype.log = function(text) {
    console.log(text);
}
