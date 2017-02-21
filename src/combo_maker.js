//
// COMBO_MAKER.JS
//

'use strict';

// export a singleton

module.exports = exports = new ComboMaker();

//

var _           = require('lodash');
var Duration    = require('duration');
var expect      = require('chai').expect;

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
//

ComboMaker.prototype.log = function(text) {
    if (this.logging) {
	console.log(text);
    }
}

//
// args:
//  count:   # of primary clues to combine
//  max:     max # of sources to use
//  require: required clue counts, e.g. [3,5,8]
//  sources: limit to these primary sources, e.g. [1,9,14]
//  use:     list of clue name-counts, e.g. ['john:1','bob:5']
//
// A "clueSourceList" is a list (array) where each element is a
// object that contains a list (cluelist) and a count, such as
// [ { list:clues1, count:1 },{ list:clues2, count:2 }].
//
ComboMaker.prototype.makeCombos = function(args) {
    var nameCsvList = [];
    var successDuration = 0;
    var failDuration = 0;
    var nextDuration = 0;
    var useNcList;
    var require;

    this.nextSort = 0;
    this.nextDupeClue = 0;
    this.nextDupeSrc = 0;
    this.nextDupeCombo = 0;

    if (_.isUndefined(args.maxResults)) {
	args.maxResults = 50000;
    }

    // TODO USE "validateArgs" 

    if (!_.isUndefined(args.use)) {
	useNcList = this.buildUseNcList(args.use); // , require);
    }
    let validateAll = false;
    if (!_.isUndefined(args.sources)) {
	console.log('Valididating sources: ' + args.sources);
	validateAll = true;
    }

    let totalCount = 0;
    let skipCount = 0;
    // for each sourceList in sourceListArray
    ClueManager.getClueSourceListArray({
	sum:     args.sum,
	max:     args.max,
	require: args.require,
    }).forEach(clueSourceList => {
	let sourceIndexes = [];

	let result = this.first(clueSourceList, sourceIndexes);
	if (result.done) {
	    throw new Error('no valid combos');
	}

	// this is effectively Peco.getCombinations().forEach()
	let first = true;
	while (!result.done) {
	    if (!first) {
		let start = new Date();
		result = this.next(clueSourceList, sourceIndexes);
		nextDuration += (new Duration(start, new Date())).milliseconds;
		if (result.done) {
		    break;
		}
	    }
	    else {
		first = false;
	    }

	    /*
	    // build list of clue names from list of clue sources and sourceIndex array
	    let clueNameList = clueSourceList.map(
		(clueSource, index) => clueSource.list[sourceIndexes[index]].name);

	    // DUBIOUS! filter out clue lists with duplicate clue names.
	    if (_.uniq(clueNameList).length !== clueNameList.length) {
		expect(true).to.be.false; // because we filter these out in next()
		continue;
	    }
	    */

	    // if useNcList, all nc must exist in current combo's nc list
	    if (!_.isUndefined(useNcList)) {
		if (_.intersection(useNcList, result.ncList).length !== useNcList.length) {
		    ++skipCount;
		    continue;
		}
	    }

	    let start = new Date();
	    let validateResult = Validator.validateSources({
		sum:         args.sum,
		nameList:    result.nameList,
		count:       result.nameList.length,
		require:     args.require, // ??
		validateAll: validateAll
	    });
	    let duration = new Duration(start, new Date());

	    if (validateResult.success) {
		successDuration += duration.milliseconds;

		if (validateAll &&
		    !this.checkPrimarySources(validateResult.ncListArray, args.sources)) {
		    continue;
		}			
		if (nameCsvList.length < args.maxResults) {
		    nameCsvList.push(result.nameList.toString());
		}
		if ((++totalCount % 10000) === 0) {
		    console.log(`total(${totalCount}), hash(${_.size(this.hash)}), list(${nameCsvList.length})`);
		}
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
    console.log(`total(${totalCount})` +
		', dupeClue(' + this.nextDupeClue + ')' +
		', dupeSrc(' + this.nextDupeSrc + ')' +
		', dupeCombo(' + this.nextDupeCombo + ')' +
		`, skip(${skipCount})`);

    if (0) { this.displayCombos(clueListArray); }

    return nameCsvList;
}

// As long as one final result has only primary sources from 'sources'
// array, we're good.
//
ComboMaker.prototype.checkPrimarySources = function(resultList, sources) {
    return resultList.some(result => {
	return NameCount.makeCountList(result.nameSrcList).
	    every(source => {
		return _.includes(sources, source);
	    });
    });
}

//
//

ComboMaker.prototype.buildUseNcList = function(nameList, countList) {
    let ncList = [];
    nameList.forEach(name =>  {
	let nc = NameCount.makeNew(name);
	if (nc.count) {
	    if (!ClueManager.knownClueMapArray[nc.count][nc.name]) {
		throw new Error('specified clue does not exist, ' + nc);
	    }
	    /*
	    if (!_.includes(countList, nc.count)) {
		countList.push(nc.count);
	    }
	    */
	}
	ncList.push(nc);
    });
    return ncList;
}

//
//

ComboMaker.prototype.hasUniqueClues = function(clueList) {
    let sourceMap = {};
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
    let index;

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
    for (;;) {
	if (!this.nextIndex(clueSourceList, sourceIndexes)) {
	    return { done: true };
	}
	let ncList = [];          // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
	let nameList = [];        // e.g. [ "pollock", "jackson" ]
	let srcCountStrList = []  // e.g. [ "white,fish:2", "moon,walker:4" ]
	if (!clueSourceList.every((clueSource, index) => {
	    let clue = clueSource.list[sourceIndexes[index]];
	    if (clue.ignore || clue.skip) {
		return false; // every.exit
	    }
	    nameList.push(clue.name);
	    // I think this is right
	    ncList.push(NameCount.makeNew(clue.name, clueSource.count)); //clue.src));
	    srcCountStrList.push(NameCount.makeCanonicalName(clue.src, clueSource.count));
	    return true; // every.continue;
	})) {
	    continue;
	}

	// skip combinations that have duplicate source:count
	if (_.uniq(srcCountStrList).length !== srcCountStrList.length) {
	    if (this.logging) {
		this.log('skipping duplicate clue src: ' + strCountStrList);
	    }
	    ++this.nextDupeSrc;
	    continue;
	}

	// skip combinations that have duplicate names
	nameList.sort();
	if (_.sortedUniq(nameList).length !== nameList.length) {
	    if (this.logging) {
		this.log('skipping duplicate clue name: ' + nameList);
	    }
	    ++this.nextDupeClue; // TODO: DupeName
	    continue;
	}

	// skip duplicate combinations
	ncList.sort();
	let ncCsv = ncList.toString();
	if (ncCsv in this.hash) {
	    if (this.logging) {
		this.log('skipping duplicate combo: ' + ncCsv);
	    }
	    ++this.nextDupeCombo;
	    continue;
	}
	this.hash[ncCsv] = true;

	return {
	    done:     false,
	    ncList:   ncList,
	    nameList: nameList
	};
    }
}

//
//

ComboMaker.prototype.nextIndex = function(clueSourceList, sourceIndexes) {
    let index = sourceIndexes.length - 1;

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
    let count = 0;
    clueListArray.forEach(function(clueList) {
	clueList.display();
	++count;
    });
    console.log('total = ' + count);
}

//
//

ComboMaker.prototype.clueListToString = function(clueList) {
    let str = '';

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

