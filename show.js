'use strict';

var showExports = {
    compatibleKnownClues : compatibleKnownClues
};

module.exports = showExports;

//

var ClueManager            = require('./clue_manager');
var Validator              = require('./validator');
var NameCount              = require('./name_count');

var LOGGING = true;

//
//

function log(text) {
    if (LOGGING) {
	console.log(text);
    }
}


// 1. 1st pass, require  name:count on all useNames, 
//    later I can loop through all possible nc combos
// 2. start with total maxclues - total primary clue count,
//    later I can loop from cluecount + 1 to maxclues
// 3. iterate through clue lists that are equal or less
//    than remainder. validate & add to map
// 4. dump map (array) contents
// 5. for -o:
//     will need to integrate in show-sources data 
//     will need some sort by sources, then alpha
// 6. note that validateSources' refusal to validate
//    NCs will likely yield some funky results, until
//    I get that implemented
//
function compatibleKnownClues(useNameList, max) {
    var ncList;
    var remain;
    var totalCount;
    var count;
    var matchNameMapArray; // should probably be SrcMap[src] = [ { name: name, srcCounts: [1,2,3] }, ... ]
    var clueList;

    totalCount = 0;
    ncList = [];
    useNameList.forEach(name => {
	var nc = new NameCount(name);
	if (!nc.count) {
	    throw new Error('All -u names require a count (for now)');
	}
	totalCount += nc.count;
	ncList.push(nc);
    });
    
    log('Show.compatibleKnownClues, ' + ncList +
	', total = ' + totalCount);

    if (!max) {
	max = ClueManager.maxClues;
    }
    remain = max - totalCount;
    if (remain < 1) {
	console.log('Nothing to do.');
	return;
    }

    matchNameMapArray = [];
    for (count = 1; count <= remain; ++count) {
	clueList = ClueManager.clueListArray[count];
	
	clueList.forEach(clue => {
	    if (isCompatibleClue(ncList, clue, totalCount + count)) {
		addToCompatibleMap(clue, matchNameMapArray, count);
	    }
	});
    }
    dumpCompatibleClues(matchNameMapArray);
}

//
//

function dumpCompatibleClues(nameMapArray) {
    var count;
    var map;
    var list;
    var key;

    for (count = 1; count < ClueManager.maxClues; ++count) {
	map = nameMapArray[count];

	// copy all map entries to array and sort by source length,
	// source #s, alpha name

	list = [];
	for (key in map) {
	    list.push({
		name:    key,
		srcList: map[key] // { srcListArray: [[x,y],[y,z]], countListArray[[1,2],[2,3]]
	    });
	}

	//list.sort(compatibleClueSort);

	list.forEach(elem => {
	    dumpCompatibleData(elem);
	});
    }
}

//

function dumpCompatibleData(data) {
    data.srcList.forEach((src, index) => {
	var srcCountStr = '';
	var srcStr = '';

	srcCountStr = '1';
	srcStr = src;
	
	console.log(srcCountStr + format2(srcCountStr, 10) +
		    ' ' + srcStr + format2(srcStr, 35) + 
		    ' ' + data.name);
    });
}

//

function format2(text, span) {
    var result = "";
    for (var len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}


//
//

function isCompatibleClue(ncList, clue, sum) {
    var nameList;
    var valid;

    nameList = [];
    ncList.forEach(nc => {
	nameList.push(nc.name);
    });
    nameList.push(clue.name);
    nameList.sort(); // necessary?

    log('Validating ' + nameList + ' (' + sum + ')');

    valid = Validator.validateSources({
	sum:      sum,
	nameList: nameList,
	count:    nameList.length
	//,showAll:  true
    });

    log('isCompatible: ' + nameList + ', ' + valid);
    return valid;
}

//
//

function addToCompatibleMap(clue, matchNameMapArray, index) {
    var map;
    var src;
    
    if (!matchNameMapArray[index]) {
	map = matchNameMapArray[index] = {};
    }
    else {
	map = matchNameMapArray[index];
    }
	
    src = clue.src;

    if (!map[clue.name]) {
	map[clue.name] = [ src ];
    }
    else {
	map[clue.name].push(src);
    }
}
