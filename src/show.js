'use strict';

var Np          = require('named-parameters');
var _           = require('lodash');
var expect      = require('chai').expect;

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

// TODO: take an ncList as an arg, not a nameList

function compatibleKnownClues(args) {
    if (!_.has(args, 'nameList') ||
	!_.has(args, 'max'))
    {
	throw new Error('missing argument');
    }

    // build ncList of supplied name:counts
    let totalCount = 0;
    let ncList = [];
    args.nameList.forEach(name => {
	var nc = NameCount.makeNew(name);
	if (!nc.count) {
	    throw new Error('All -u names require a count (for now)');
	}
	totalCount += nc.count;
	ncList.push(nc);
    });

    let remain = args.max - totalCount;
    if (remain < 1) {
	console.log('The total count of specified clues (' + totalCount + ')' +
		    ' equals or exceeds the maximum clue count (' + args.max + ')');
	return;
    }
    
    console.log('Show.compatibleKnownClues, ' + args.nameList + 
		', total = ' + totalCount +
		', remain = ' + remain);

    let realNameList = args.nameList.map(name => NameCount.makeNew(name)).map(nc => nc.name);
    // first, make sure the supplied nameList by itself is a valid clue
    // combination, and find out how many primary-clue variations there
    // are in which the clue names in useNameList can result.
    let vsResult = Validator.validateSources({
	sum:         totalCount,
	nameList:    realNameList,
	count:       args.nameList.length,
	validateAll: true
    });
    if (!vsResult.success) {
	console.log('The nameList [ ' + args.nameList + ' ] is not a valid clue combination');
	return;
    }

    // for each primary-clue variation from validateResults
    vsResult.ncListArray.forEach(result => {
	log('final result: ' + result.nameSrcList);
	var primarySrcList = result.nameSrcList.map(nc => _.toNumber(nc.count));
	var result;

	// for each clue in each clueListArray[count] where count is
	// less than or equal to the remaining count, add compatible
	// clues to the compatible map.

	// should probably be SrcMap[src] = [ { name: name, srcCounts: [1,2,3] }, ... ]
	let matchNameMapArray = [];
	for (let count = 1; count <= remain; ++count) {
	    // use only uniquely named clues
	    _.uniqWith(ClueManager.clueListArray[count], (c1, c2) => {
		return c1.name === c2.name;
	    }).forEach(clue => {
		log('checking: ' + clue.name);
		let resultList;
		if (resultList = isCompatibleClue({
		    sum:            count,
		    clue:           clue,
		    excludeSrcList: primarySrcList
		})) {
		    log('success, adding ' + clue.name);
		    addToCompatibleMap({
			clue:           clue,
			resultList:     resultList,
			excludeSrcList: primarySrcList,
			nameMapArray:   matchNameMapArray,
			index:          count
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
	    nameMapArray: matchNameMapArray,
	    asCsv:        args.asCsv
	});
    });
}

// args:
//  sum:           
//  clue:          
//  excludeSrcList:
//
function isCompatibleClue(args) {
    expect(args).to.have.property('sum')
	.that.is.a('number');
    expect(args).to.have.property('clue')
	.that.is.an('object');
    expect(args).to.have.property('excludeSrcList')
	.that.is.an('array');

    if (_.includes(args.excludeSrcList, _.toNumber(args.clue.src))) {
	return false;
    }

    log('Validating ' + args.name + ' (' + args.sum + ')');

    let result = Validator.validateSources({
	sum:            args.sum,
	nameList:       [ args.clue.name ],
	count:          1,
	excludeSrcList: args.excludeSrcList,
	validateAll:    true,
	wantResults:    true
    });

    log('isCompatible: ' + args.clue.name + ':' + args.clue.src +
	'(' + args.sum + '), ' + Boolean(result));

    return result.success ? result.ncListArray : false;
}

// args:
//  clue,
//  resultList,
//  excludeSrcList:
//  mapArray
//  index
//
function addToCompatibleMap(args) {
    expect(args).to.have.property('resultList')
	.that.is.an('array').and.is.not.empty;
    expect(args).to.have.property('excludeSrcList')
	.that.is.an('array');//.and.is.not.empty;
    expect(args).to.have.property('nameMapArray')
	.that.is.an('array');//.and.is.not.empty;

    if (LOGGING) {
	log('++addToCompatibleMap' +
	    ', clue: ' + args.clue +
	    ', resultList: ' + args.resultList);
    }

    let map;
    if (!args.nameMapArray[args.index]) {
	map = args.nameMapArray[args.index] = {};
    }
    else {
	map = args.nameMapArray[args.index];
    }
	
    let name = args.clue.name;
    let src = args.clue.src;

    args.resultList.forEach(result => {
	let primarySrcList = result.nameSrcList.map(nc => _.toNumber(nc.count));
	primarySrcList = filterExcludedSources(primarySrcList, args.excludeSrcList);
	if (!primarySrcList.length) {
	    throw new Error('should not happen');
	}
	let key = name + ':' + primarySrcList;
	if (_.isUndefined(map[key])) {
	    map[key] = [];
	}
	map[key].push({
	    src:            src,
	    primarySrcList: primarySrcList
	});
	log('adding: ' + primarySrcList + ' - ' + src);
    });
}

//
//

function filterExcludedSources(srcList, excludeSrcList, verboseFlag) {
    let resultList = [];
    srcList.forEach(src => {
	if (!excludeSrcList.includes(_.toNumber(src))) {
	    resultList.push(src);
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
    var list;
    var name;
    var countList = [ 1 ];

    let sources = '';
    args.nameList.forEach(name => {
	var nc = NameCount.makeNew(name);
	if (sources.length > 0) {
	    sources += ', ';
	}
	// TODO: [0] looks like a bug
	// sortof. simplifying the output header
	sources += ClueManager.knownClueMapArray[nc.count][nc.name][0];
    });

    let csvNameList = args.nameList.map(name => NameCount.makeNew(name)).map(nc => nc.name);

    console.log(args.srcList + format2(args.srcList, FIRST_COLUMN_WIDTH) +
		' ' + sources + format2(sources, SECOND_COLUMN_WIDTH) +
		' ' + args.nameList +
		'\n');

    for (count = 1; count < ClueManager.maxClues; ++count) {
	// copy all map entries to array and sort by source length,
	// source #s, alpha name

	let map = args.nameMapArray[count];
	let dumpList = [];
	_.forOwn(map, (value, key) => {
	    value.forEach(elem => {
		// TODO: can't I do this up higher somewwhere?
		elem.name = NameCount.makeNew(key).name; // .split(':')[0]; // isolate name from name:count
		dumpList.push(elem);
	    });
	});

	if (!_.isEmpty(dumpList)) {
	    dumpList.sort((a, b) => {
		return a.primarySrcList.toString().localeCompare(b.primarySrcList.toString());
	    });
	    dumpCompatibleClueList(
		dumpList,
		args.asCsv ? csvNameList.toString() : undefined
	    );
	}
    }
    console.log('');
}

//

function dumpCompatibleClueList(list, csvNames = undefined) {
    list.forEach(elem => {
	if (csvNames) {
	    console.log(csvNames + ',' + elem.name);
	}
	else {
	    console.log(elem.primarySrcList + format2(elem.primarySrcList, FIRST_COLUMN_WIDTH) +
		    ' ' + elem.src + format2(elem.src, SECOND_COLUMN_WIDTH) + 
			' ' + elem.name);
	}
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

var showExports = {
    compatibleKnownClues : compatibleKnownClues
};

module.exports = showExports;

