'use strict';

var showExports = {
    compatibleKnownClues : compatibleKnownClues
};

module.exports = showExports;

//

var Np          = require('named-parameters');
var _           = require('lodash');

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

////////////////////////////////////////////////////////////////////////////////
//
// Show.compatibleKnownClues()
//
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

    if (!_.has(args, 'nameList') ||
	!_.has(args, 'max'))
    {
	throw new Error('missing argument');
    }

    // build ncList of supplied name:counts
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

    // first, make sure the supplied nameList by itself is a valid clue
    // combination, and find out how many primary-clue variations there
    // are in which the clue names in useNameList can result.
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

    // for each final result from validateResults
    Validator.getFinalResultList(result).forEach(ncCsv => {
	// ncCsv is actually a name:primary_source (not name:count)
	var ncList = NameCount.makeListFromCsv(ncCsv);
	var primarySrcList = NameCount.makeCountList(ncList);
	var result;

	// for each clue in each clueList[count] where count is
	// less than or equal to the remaining count, add compatible
	// clues to the compatible map.
	matchNameMapArray = [];
	for (count = 1; count <= remain; ++count) {
	    clueList = ClueManager.clueListArray[count];
	    clueList.forEach(clue => {
		var resultList;
		log('checking: ' + clue.name);
		if (resultList = isCompatibleClue({
		    sum:           count,
		    clue:          clue,
		    excludeSrcList:primarySrcList
		})) {
		    log('success, adding + ' + clue.name);
		    addToCompatibleMap({
			clue:          clue,
			resultList:    resultList,
			excludeSrcList:primarySrcList,
			nameMapArray:  matchNameMapArray,
			index:         count
		    });
		}
		else {
		    log('not compatible: ' + clue.name);
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
// args:
//  sum:           count,
//  clue:          clue
//  excludeSrcList:
//
function isCompatibleClue(args) {
    var result;
    var resultList;
    var primarySrc;

    if (!_.has(args, 'sum') ||
	!_.has(args, 'clue') ||
	!_.has(args, 'excludeSrcList'))
    {
	throw new Error('missing argument');
    }

    if (_.includes(args.excludeSrcList, _.toNumber(args.clue.src))) {
	return false;
    }

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
	nameList:       [ args.clue.name ],
	count:          1,
	excludeSrcList: args.excludeSrcList,
	validateAll:    true,
	wantResults:    true
    });

    log('isCompatible: ' + args.clue.name + ':' + args.clue.src +
	'(' + args.sum + '), ' + Boolean(result));

    if (result) {
	resultList = Validator.getFinalResultList(result);
	if (!resultList) {
	    throw new Error('I donut think this happens');
	}
    }
    return result ? resultList : false;
}

// args:
//  clue,
//  resultList,
//  excludeSrcList:
//  mapArray
//  index
//
function addToCompatibleMap(args) {
    var map;
    var name;
    var src;
    var verbose;

    if (!_.has(args, 'resultList') ||
	!_.has(args, 'excludeSrcList') ||
	!_.has(args, 'nameMapArray'))
    {
	throw new Error('missing argument');
    }

    if (!args.nameMapArray[args.index]) {
	map = args.nameMapArray[args.index] = {};
    }
    else {
	map = args.nameMapArray[args.index];
    }
	
    name = args.clue.name;
    src = args.clue.src;

    verbose = false; // name == 'ace';

    args.resultList.forEach(ncCsv => {
	var primarySrcList;
	var key;
	primarySrcList = NameCount.makeCountList(NameCount.makeListFromCsv(ncCsv));
	if (verbose) {
	    console.log('primary sources for ' + name + ': ' + primarySrcList);
	}
	primarySrcList = filterExcludedSources(primarySrcList, args.excludeSrcList, verbose);
	if (!primarySrcList.length) {
	    throw new Error('should not happen');
	}
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
	if (verbose) {
	    console.log('adding: ' + primarySrcList + ' - ' + src);
	}
	log('adding: ' + primarySrcList + ' - ' + src);
    });
}

//
//

function filterExcludedSources(srcList, excludeSrcList, verboseFlag) {
    var resultList;
    resultList = [];

    srcList.forEach(src => {
	if (verboseFlag) {
	    console.log('looking for ' + src + ' in ' + excludeSrcList);
	}
	src = _.toNumber(src);
	if (excludeSrcList.indexOf(src) === -1) {
	    if (verboseFlag) {
		console.log('not found');
	    }

	    resultList.push(src);
	}
	else if (verboseFlag) {
	    console.log('found');
	}
    });
    return resultList;
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
    var name;
    var countList = [ 1 ];
    var sources = '';
    
    args.nameList.forEach(name => {
	var nc = NameCount.makeNew(name);
	if (sources.length > 0) {
	    sources += ', ';
	}
	// TODO: [0] looks like a bug
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
	for (name in map) {
	    map[name].forEach(elem => {
		elem.name = name;
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

