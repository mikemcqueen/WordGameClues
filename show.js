'use strict';

var showExports = {
    compatibleKnownClues : compatibleKnownClues
};

module.exports = showExports;

//

var ClueManager            = require('./clue_manager');
var Validator              = require('./validator');
var NameCount              = require('./name_count');

var FIRST_COLUMN_WIDTH    = 15;
var SECOND_COLUMN_WIDTH   = 35;

//
//

var LOGGING = false;

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
function compatibleKnownClues(args) {
    var ncList;
    var nameSourceList;
    var remain;
    var totalCount;
    var count;
    var matchNameMapArray; // should probably be SrcMap[src] = [ { name: name, srcCounts: [1,2,3] }, ... ]
    var clueList;
    var result;

    if (!args.nameList || !args.max) { // || !args.ncMap) {
	throw new Error('missing argument, nameList: ' + args.nameList +
			', max: ' + args.max ); //+
	//', ncMap: ' + args.ncMap);
    }

    totalCount = 0;
    ncList = [];
    args.nameList.forEach(name => {
	var nc = NameCount.makeNew(name);
	if (!nc.count) {
	    throw new Error('All -u names require a count (for now)');
	}
	totalCount += nc.count;
	ncList.push(nc);
    });
    
    console.log('Show.compatibleKnownClues, ' + args.nameList + 
	', total = ' + totalCount);

    remain = args.max - totalCount;
    if (remain < 1) {
	console.log('The total count of specified clues (' + totalCount + ')' +
		    ' equals or exceeds the maximum clue count (' + args.max + ')');
	return;
    }

    // first, make sure the supplied name list by itself
    // is a valid clue combination, and find out how many
    // primary-clue variations there are in which the clue
    // names in useNameList can result.

    result = Validator.validateSources({
	sum:         totalCount,
	nameList:    NameCount.makeNameList(NameCount.makeListFromNameList(args.nameList)),
	count:       args.nameList.length,
	validateAll: true
    });
    if (!result) {
	console.log('The nameList [ ' + args.nameList + ' ] is not a valid clue combination');
	return;
    }

    /*
    // set all sources to primary
    ncList.forEach(nc => {
	nc.count = 1;
    });
    */

    // for each result from validateResults
    result[Validator.PRIMARY_KEY][Validator.FINAL_KEY].forEach(ncCsv => {
	// ncCsv is actually a name:primary_source (not name:count)
	var ncList = NameCount.makeListFromCsv(ncCsv);
	var primarySrcList = NameCount.makeCountList(ncList);
	var result;

	// make ncList a name:count
	ncList.forEach(nc => {
	    nc.count = 1;
	});

	matchNameMapArray = [];
	for (count = 1; count <= remain; ++count) {
	    if (totalCount + count > args.max) {
		break;
	    }
	    clueList = ClueManager.clueListArray[count];
	    clueList.forEach(clue => {
		if (result = isCompatibleClue(ncList, clue, totalCount + count)) {
		    addToCompatibleMap({
			clue:     clue,
			result:   result,
			removeSrcList : primarySrcList,
			mapArray: matchNameMapArray,
			index:    count
		    });
		}
	    });
	}
	dumpCompatibleClues({
	    nameList:     args.nameList,
	    srcList:      primarySrcList,
	    nameMapArray: matchNameMapArray
	});
    });
}

//
//

function dumpCompatibleClues(args) {
    var count;
    var map;
    var list;
    var key;
    var countList = [ 1 ];

    console.log(args.srcList + format2(args.srcList, FIRST_COLUMN_WIDTH) +
		' ' + args.nameList);

    for (count = 1; count < ClueManager.maxClues; ++count) {
	map = args.nameMapArray[count];

	// copy all map entries to array and sort by source length,
	// source #s, alpha name

	list = [];
	for (key in map) {
	    list.push({
		countList: countList,
		srcList:   map[key], // { srcListArray: [[x,y],[y,z]], countListArray[[1,2],[2,3]]
		name:      key
	    });
	}

	//list.sort(compatibleClueSort);

	list.forEach(elem => {
	    dumpCompatibleData(elem);
	});
    }
}

//

function dumpCompatibleData(args) {
    args.srcList.forEach((src, index) => {
	console.log(args.countList + format2(args.countList, FIRST_COLUMN_WIDTH) +
		    ' ' + src + format2(src, SECOND_COLUMN_WIDTH) + 
		    ' ' + args.name);
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
    var result;
    var nameList;

    nameList = [];
    ncList.forEach(nc => {
	nameList.push(nc.name);
    });
    nameList.push(clue.name);
    nameList.sort(); // necessary?

    log('Validating ' + nameList + ' (' + sum + ')');

    result = Validator.validateSources({
	sum:         sum,
	nameList:    nameList,
	count:       nameList.length,
	wantResults: true
    });

    log('isCompatible: ' + nameList + ', ' + Boolean(result));

    return result;
}

//clue,
//result,
//mapArray
//index
//

function addToCompatibleMap(args) {
    var map;
    var name;
    var src;
    
    if (!args.mapArray[args.index]) {
	map = args.mapArray[args.index] = {};
    }
    else {
	map = args.mapArray[args.index];
    }
	
    name = args.clue.name;
    src = args.clue.src;

    if (!map[name]) {
	map[name] = [ src ];
    }
    else {
	map[name].push(src);
    }
}
