'use strict';

var _              = require('lodash');
var expect         = require('chai').expect;

var ClueManager    = require('./clue_manager');
var Validator      = require('./validator');
var NameCount      = require('./name_count');

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
    expect(args).to.have.property('nameList')
	.that.is.an('array');
    expect(args).to.have.property('max')
	.that.is.a('number');

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

    // first, make sure the supplied nameList by itself is a valid clue
    // combination, and find out how many primary-clue variations there
    // are in which the clue names in useNameList exist.
    // strip :COUNT from names in nameList.
    let realNameList = args.nameList.map(name => NameCount.makeNew(name)).map(nc => nc.name);
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
	    _.uniqWith(ClueManager.clueListArray[count], (c1, c2) => c1.name === c2.name)
		.forEach(clue => {
		    log('checking: ' + clue.name);
		    let nc = NameCount.makeNew(clue.name, count);
		    let resultList = getCompatibleResults(nc, primarySrcList);
		    if (!_.isEmpty(resultList)) {
			log('success, adding ' + clue.name);
			addToCompatibleMap({
			    nameCount:      nc,
			    resultList:     resultList,
			    excludeSrcList: primarySrcList,
			    nameMapArray:   matchNameMapArray,
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
// TODO: bad function name;
// getCompatibleResults
function getCompatibleResults(nameCount, excludeSrcList) {
    expect(nameCount).to.be.an('object');
    expect(excludeSrcList).to.be.an('array');

    log(`Validating ${nameCount.name} (${nameCount.count})`);
    let result = Validator.validateSources({
	sum:            nameCount.count,
	nameList:       [ nameCount.name ],
	count:          1,
	excludeSrcList: excludeSrcList,
	validateAll:    true,
	wantResults:    true
    });
    log(`isCompatible: ${nameCount.name} (${nameCount.count}), ${result.success}`);
    return result.success ? result.ncListArray : [];
}

// args:
//  nameCount
//  resultList
//  excludeSrcList
//  mapArray
//
function addToCompatibleMap(args) {
    expect(args.nameCount).to.be.an('object');
    expect(args.resultList).to.be.an('array').that.is.not.empty;
    expect(args.excludeSrcList).to.be.an('array');//.that.is.not.empty;
    expect(args.nameMapArray).to.be.an('array');//.that.is.not.empty;

    if (LOGGING) {
	log(`++addToCompatibleMap, clue: ${args.nameCount.name}, resultList(${args.resultList.length})`);
    }
    let map = args.nameMapArray[args.nameCount.count];
    if (_.isUndefined(map)) {
	map = args.nameMapArray[args.nameCount.count] = {};
    }
    args.resultList.forEach(result => {
	let primarySrcList = _.chain(result.nameSrcList) // from array of name:source
	    .map(nc => _.toNumber(nc.count))             // to array of sources
	    .without(args.excludeSrcList).value();       // to array of non-excluded sources
	expect(primarySrcList.length).to.be.at.least(1);
	// TODO: shouldn't this be name:nameCount.count
	let key = args.nameCount.name + ':' + primarySrcList;
	if (_.isUndefined(map[key])) {
	    map[key] = [];
	}
	if (LOGGING) {
	    result.resultMap.dump();
	}
	expect(result.resultMap.map(), args.nameCount).to.have.property(args.nameCount.toString());
	let inner = result.resultMap.map()[args.nameCount.toString()];
	let csvNames;
	if (_.isArray(inner)) { // special case, a single primary clue is represented by an array
	    csvNames = args.nameCount.name;
	}
	else {
	    csvNames = _.chain(inner).keys() 	             // from array of name:count csv strings
		.map(ncCsv => ncCsv.split(',')).flatten()    // to array of name:count strings
		.map(ncStr => NameCount.makeNew(ncStr).name) // to array of names
		.sort().value().toString();                  // to sorted csv names
	}
	expect(csvNames).to.be.a('string');
	map[key].push({
	    src:            csvNames,
	    primarySrcList: primarySrcList
	});
	if (LOGGING) {
	    log(`adding: ${primarySrcList} - ${csvNames}`);
	}
    });
}

// args:
//  nameList:    
//  srcList:     
//  nameMapArray:
//
function dumpCompatibleClues(args) {
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

    console.log(args.srcList + format2(args.srcList, FIRST_COLUMN_WIDTH) +
		' ' + sources + format2(sources, SECOND_COLUMN_WIDTH) +
		' ' + args.nameList +
		'\n');

    // strip ":COUNT" suffix from names in nameList
    let rawNameList = args.nameList.map(name => NameCount.makeNew(name)).map(nc => nc.name);
    for (let count = 1; count < ClueManager.maxClues; ++count) {
	let map = args.nameMapArray[count];
//	let dumpList = [];
	let dumpList = _.chain(map).keys().map(key => {              // from array of name:src,src,src
	    let name = NameCount.makeNew(key).name;
	    return map[key].map(elem => {             // to array of array of elements
		elem.name = name;
		return elem;
	    });
	}).flatten().value();                         // to array of elements
	/*
	_.forOwn(map, (value, key) => {
	    value.forEach(elem => {
		elem.name = NameCount.makeNew(key).name;
		dumpList.push(elem);
	    });
	});
	*/

	if (!_.isEmpty(dumpList)) {
	    dumpList.sort((a, b) => {
		return a.primarySrcList.toString().localeCompare(b.primarySrcList.toString());
	    });
	    dumpCompatibleClueList(dumpList, args.asCsv ? rawNameList : undefined);
	}
    }
    console.log('');
}

//

function dumpCompatibleClueList(list, nameList = undefined) {
    list.forEach(elem => {
	if (_.isUndefined(nameList)) {
	    console.log(elem.primarySrcList + format2(elem.primarySrcList, FIRST_COLUMN_WIDTH) +
			' ' + elem.src + format2(elem.src, SECOND_COLUMN_WIDTH) + 
			' ' + elem.name);
	}
	else {
	    console.log(_.concat(nameList, elem.name).sort().toString());
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

