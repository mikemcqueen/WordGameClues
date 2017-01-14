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
var SECOND_COLUMN_WIDTH   = 25;

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

    if (!args.nameList || !args.max) { 
	throw new Error('missing argument, nameList: ' + args.nameList +
			', max: ' + args.max ); 
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

    remain = args.max - totalCount;
    if (remain < 1) {
	console.log('The total count of specified clues (' + totalCount + ')' +
		    ' equals or exceeds the maximum clue count (' + args.max + ')');
	return;
    }
    
    console.log('Show.compatibleKnownClues, ' + args.nameList + 
		', total = ' + totalCount +
		', remain = ' + remain);

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

    // for each result from validateResults
    result[Validator.PRIMARY_KEY][Validator.FINAL_KEY].forEach(ncCsv => {
	// ncCsv is actually a name:primary_source (not name:count)
	var ncList = NameCount.makeListFromCsv(ncCsv);
	var primarySrcList = NameCount.makeCountList(ncList);
	var result;

	matchNameMapArray = [];
	for (count = 1; count <= remain; ++count) {
	    clueList = ClueManager.clueListArray[count];
	    clueList.forEach(clue => {
		var resultList;
		if (resultList = isCompatibleClue({
		    sum:     count,
		    name:    clue.name,
		    exclude: primarySrcList
		})) {
		    addToCompatibleMap({
			clue:       clue,
			resultList: resultList,
			mapArray:   matchNameMapArray,
			index:      count
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
//  sum:     count,
//  name:    clue.name,
//  exclude: primarySrcList
//
function isCompatibleClue(args) {
    var result;
    var resultList;
    var primarySrc;

    //TODO: sort

    /*
    var nameList;
    nameList = [ clue.name ];
    ncList.forEach(nc => {
	nameList.push(nc.name);
    });
    nameList.push(clue.name);
    nameList.sort(); // necessary?
    */

    log('Validating ' + args.name + ' (' + args.sum + ')');

    result = Validator.validateSources({
	sum:            args.sum,
	nameList:       [ args.name ],
	count:          1,
	excludeSrcList: args.exclude,
	validateAll:    true,
	wantResults:    true
	//,quiet:          true
    });

    log('isCompatible: ' + args.name + '(' + args.sum + '), ' + Boolean(result));

    if (result) {
	resultList = Validator.getFinalResultList(result);
	if (!resultList) {
	    console.log('does this even happen anymore');
	    if (sum == 1) {
		primarySrc = ClueManager.clueNameMapArray[1][args.name];
		if (!primarySrc) {
		    throw new Error('missing primary clue: ' + args.name);
		}
		else {
		    console.log('using primary clue lookup, ' + args.name);
		    resultList = [ primarySrc ];
		}
	    }
	    else {
		throw new Error('Empty result list, name: ' + args.name + ', sum: ' + sum );
	    }
	}
    }
    return result ? resultList : false;
}

// args:
//  clue,
//  resultList,
//  mapArray
//  index
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

    args.resultList.forEach(ncCsv => {
	var primarySrcList;
	var key;
	primarySrcList = NameCount.makeCountList(NameCount.makeListFromCsv(ncCsv));
	key = name + ':' + primarySrcList;
	if (!map[key]) {
	    map[key] = [{
		src:            src,
		primarySrcList: primarySrcList
	    }];
	}
	else {
	    map[key].push({
		src:            src,
		primarySrcList: primarySrcList
	    });
	}
	log('adding: ' + primarySrcList + ' - ' + src);
    });
}

// args:
//  nameList:    
//  srcList:     
//  nameMapArray:
//
function dumpCompatibleClues(args) {
    var count;
    var map;
    var list;
    var dumpList;
    var key;
    var countList = [ 1 ];
    var sources = '';
    
    args.nameList.forEach(name => {
	var nc = NameCount.makeNew(name);
	if (sources.length > 0) {
	    sources += ', ';
	}
	// NOTE: [0] looks like a bug
	sources += ClueManager.knownClueMapArray[nc.count][nc.name][0];
    });

    console.log(args.srcList + format2(args.srcList, FIRST_COLUMN_WIDTH) +
		' ' + sources + format2(sources, SECOND_COLUMN_WIDTH) +
		' ' + args.nameList +
		'\n');

    for (count = 1; count < ClueManager.maxClues; ++count) {
	map = args.nameMapArray[count];

	// copy all map entries to array and sort by source length,
	// source #s, alpha name

	dumpList = [];
	for (key in map) {
	    map[key].forEach(elem => {
		elem.name = key;
		dumpList.push(elem);
	    });
	}

	// TODO:
	// list.sort(compatibleClueSort);

	if (dumpList.length) {
	    dumpCompatibleClueList(dumpList);
	}
    }
    console.log('');
}

//

function dumpCompatibleClueList(list) {
    list.forEach(elem => {
	console.log(elem.primarySrcList + format2(elem.primarySrcList, FIRST_COLUMN_WIDTH) +
		    ' ' + elem.src + format2(elem.src, SECOND_COLUMN_WIDTH) + 
		    ' ' + elem.name);
    });
}

//

function format2(text, span) {
    if (text === undefined) {
	text = 'undefined'
    }
    var result = "";
    for (var len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}

