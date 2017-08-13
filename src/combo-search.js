//
// combo-search.js
//

'use strict';

// export a singleton

module.exports = exports = new ComboSearch();


var _              = require('lodash');

var ClueManager    = require('./clue-manager');
var Validator      = require('./validator');
var ClueList       = require('./clue-list');
var NameCount      = require('./name-count');
var Peco           = require('./peco');

//

function ComboSearch() {
    this.logging = true;
    this.logLevel = 0;
}

//

ComboSearch.prototype.log = function(text) {
    var pad;
    var index;
    if (this.logging) {
	pad = '';
	for (var index=0; index<this.logLevel; ++index) {
	    pad += ' ';
	}
	console.log(pad + text);
    }
}

//
//

ComboSearch.prototype.findAlternateSourcesForName = function(name, count) {
    var nc;
    var srcNameListArray;
    var resultNcListArray = [];
    var peco;

    nc = NameCount.makeNew(name);
    srcNameListArray = ClueManager.makeSrcNameListArray(nc);
    srcNameListArray.forEach(srcNameList => {
	var curCount;
	var maxCount;
	var countListArray;
	var matchCountListArray = [];

	if (this.logging) {
	    this.log('looking for source list ' + srcNameList);
	}

	countListArray = [];
	srcNameList.forEach(name => {
	    countListArray.push(ClueManager.getCountListForName(name));
	});
	
	if (this.logging) {
	    this.log('count list:');
	    countListArray.forEach(countList => {
		this.log(' ' + countList);
	    });
	}
	    
	if (count) {
	    curCount = maxCount = count;
	}
	else {
	    curCount = srcNameList.length;
	    maxCount = ClueManager.maxClues;
	}
	for (; curCount <= maxCount; ++curCount) {
	    if (curCount == nc.count) {
		continue;
	    }

	    if (this.logging) {
		this.log('  for count ' + curCount);
	    }

	    Peco.makeNew({
		sum:   curCount,
		count: srcNameList.length
	    }).getPermutations().forEach(countList => {
		if (this.logging) {
		    this.log('   in ' + countList);
		}

		if (this.findCountListInCountListArray(countList, countListArray)) {
		    if (!matchCountListArray[curCount]) {
			matchCountListArray[curCount] = [ countList ];
		    }
		    else {
			matchCountListArray[curCount].push(countList);
		    }
		    if (this.logging) {
			this.log('    found! length=' + countList.length);
		    }
		}
		else {
		    if (this.logging) {
			this.log('    failed');
		    }
		}
	    }, this);
	}

	// really: countListArrayArray
	matchCountListArray.forEach((countListArray, claaIndex) => {
	    var ncListArray = [];
	    countListArray.forEach((countList, claIndex) => {
		var ncList = [];
		var sum = 0;
		var result;
		countList.forEach((count, clIndex) => {
		    sum += count;
		    ncList.push(NameCount.makeNew(srcNameList[clIndex], count));
		});
		if (sum != claaIndex ) {
		    throw new Error('something i dont understand here obviously');
		}
		result = Validator.validateSources({
		    sum:      sum,
		    nameList: srcNameList,
		    count:    srcNameList.length
		});
		if (result.success) {
		    ncListArray.push(ncList);
		}
	    });
	    resultNcListArray[claaIndex] = ncListArray;
	});
    });
    return resultNcListArray;
}

// find [1, 2] in { [1,4],[2,5] }
//

ComboSearch.prototype.findCountListInCountListArray =
    function(countList, countListArray)
{
    var indexLengthList;
    var index;
    var resultCountList;

    if (countList.length != countListArray.length) {
	throw new Error('mismatched lengths');
    }

    indexLengthList = [];
    countListArray.forEach(cl => {
	indexLengthList.push({
	    index:  0,
	    length: cl.length
	});
    });

    do {
	resultCountList = [];
	if (countList.every((count, index) => {
	    if (count != countListArray[index][indexLengthList[index].index]) {
		return false;
	    }
	    return true; // every.continue
	})) {
	    return true; // function.exit
	}
    } while (this.nextIndexLength(indexLengthList));

    return null;
}

//
//

ComboSearch.prototype.findNameListInCountList =
    function(nameList, countList)
{
    var ncList;
    var countListArray = [];
    var countList;
    var count;

    if (nameList.length != countList.length) {
	throw new Error('mismatched list lengths');
    }


    return ncList;
}

//
//TODO: 

ComboSearch.prototype.first =
    function(clueSourceList, sourceIndexes)
{
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

ComboSearch.prototype.nextIndexLength =
    function(indexLengthList)
{
    var index = indexLengthList.length - 1;

    // increment last index
    ++indexLengthList[index].index;

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    while (indexLengthList[index].index >= indexLengthList[index].length) {
	indexLengthList[index].index = 0;
	--index;
	if (index < 0) {
	    return false;
	}
	++indexLengthList[index].index;
    }
    return true;
}

