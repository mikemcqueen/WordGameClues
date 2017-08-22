'use strict';

var _              = require('lodash');
var ClueManager    = require('./clue-manager');
var Clues          = require('./clue-types');
var Debug          = require('debug')('show');
var Expect         = require('chai').expect;
var NameCount      = require('./name-count');
var Result         = require('./tools/result-mod');
var Validator      = require('./validator');


var FIRST_COLUMN_WIDTH    = 15;
var SECOND_COLUMN_WIDTH   = 25;

//
//

//
// Show.compatibleKnownClues()
//
// TODO: take an ncList as an arg, not a nameList
//
function compatibleKnownClues(args) {
    Expect(args.nameList).to.be.an('array'); // this is actually a NAME:COUNT string (ncStr) list
    Expect(args.max).to.be.a('number');

    // build ncList of supplied name:counts
    let ncList = args.nameList.map(name => NameCount.makeNew(name));
    if (!ncList.every(nc => {
	if (!nc.count) {
	    console.log('All -u names require a count (for now)');
	}
	return !!nc.count;
    })) {
	return;
    }

    // TODO: some more clear way to extract just ".count"s into an array, then reduce them
    let sum = ncList.reduce((a, b) => Object({ count: (a.count + b.count) })).count;
    let remain = args.max - sum;
    if (remain < 1) {
	console.log('The sum of the specified clue counts (' + sum + ')' +
		    ' equals or exceeds the maximum clue count (' + args.max + ')');
	return;
    }
    
    Debug('Show.compatibleKnownClues, ' + args.nameList + 
		', sum: ' + sum + ', remain: ' + remain);

    // first, make sure the supplied nameList:sum by itself is a valid clue
    // combination, and find out how many primary-clue variations there
    // are in which the clue names in args.nameList exist.
    let vsResult = Validator.validateSources({
	sum:         sum,
	nameList:    ncList.map(nc => nc.name),
	count:       args.nameList.length,
	validateAll: true
    });
    if (!vsResult.success) {
	console.log(`The nameList [${args.nameList}] is not a valid clue combination`);
	return;
    }

    // for each primary-clue variation from validateResults
    vsResult.list.forEach(result => {
	Debug(`final result: ${result.nameSrcList}`);
	let primarySrcList = result.nameSrcList.map(nc => _.toNumber(nc.count));

	// for each clue in each clueListArray[count] where count is
	// less than or equal to the remaining count, add compatible
	// clues to the compatible map.

	// should probably be SrcMap[src] = [ { name: name, srcCounts: [1,2,3] }, ... ]
	let matchNameMapArray = [];
	for (let count = 1; count <= remain; ++count) {
	    // for each non-duplicate clue names
	    _.uniqWith(ClueManager.clueListArray[count], (c1, c2) => c1.name === c2.name)
		.forEach(clue => {
		    let nc = NameCount.makeNew(clue.name, count);
		    Debug(`checking: ${nc}`);
		    let resultList = getCompatibleResults(nc, primarySrcList);
		    if (!_.isEmpty(resultList)) {
			Debug('success, adding ' + clue.name);
			addToCompatibleMap({
			    nameCount:      nc,
			    resultList:     resultList,
			    excludeSrcList: primarySrcList,
			    nameMapArray:   matchNameMapArray,
			});
		    }
		    else {
			Debug('not compatible: ' + clue.name);
		    }
		});
	}
	dumpCompatibleClues({
	    nameList:     args.nameList,
	    srcList:      primarySrcList,
	    nameMapArray: matchNameMapArray,
	    format:       args.format,
	    root:         args.root
	});
    });
}

// args:
//  nameCount:
//  excludeSrcList:
//
function getCompatibleResults(nameCount, excludeSrcList) {
    Expect(nameCount).to.be.an('object');
    Expect(excludeSrcList).to.be.an('array');

    Debug(`Validating ${nameCount}`);
    let result = Validator.validateSources({
	sum:            nameCount.count,
	nameList:       [ nameCount.name ],
	count:          1,
	excludeSrcList: excludeSrcList,
	validateAll:    true
    });
    Debug(`isCompatible: ${nameCount}, ${result.success}`);
    return result.success ? result.list : [];
}

// args:
//  nameCount
//  resultList
//  excludeSrcList
//  nameMapArray
//
function addToCompatibleMap(args) {
    Expect(args.nameCount).to.be.an('object');
    Expect(args.resultList).to.be.an('array').that.is.not.empty;
    Expect(args.excludeSrcList).to.be.an('array');//.that.is.not.empty;
    Expect(args.nameMapArray).to.be.an('array');//.that.is.not.empty;

    Debug(`++addToCompatibleMap, clue: ${args.nameCount.name}, resultList(${args.resultList.length})`);
    let map = args.nameMapArray[args.nameCount.count];
    if (_.isUndefined(map)) {
	map = args.nameMapArray[args.nameCount.count] = {};
    }
    args.resultList.forEach(result => {
	let primarySrcList = _.chain(result.nameSrcList) // from array of name:source
	    .map(nc => _.toNumber(nc.count))             // to array of sources
	    .without(args.excludeSrcList).value();       // to array of non-excluded sources
	Expect(primarySrcList.length).to.be.at.least(1);
	// TODO: shouldn't this be name:nameCount.count (maybe not)
	let key = args.nameCount.name + ':' + primarySrcList.sort();
	if (!_.has(map, key)) {
	    map[key] = [];
	}
	if (false) {
	    result.resultMap.dump();
	}
	// the root resultMap property should be args.nameCount
	Expect(result.resultMap.map(), args.nameCount).to.have.property(args.nameCount.toString());
	// inner = the inner object (value) of the root object
	let inner = result.resultMap.map()[args.nameCount.toString()];
	let csvNames;
	if (_.isArray(inner)) { // special case, a single primary clue is represented by an array
	    csvNames = args.nameCount.name;
	}
	else {
	    csvNames = _.chain(inner).keys() 	             // from array of name:count (sometimes csv) strings
		.map(ncCsv => ncCsv.split(',')).flatten()    // to array of name:count strings
		.map(ncStr => NameCount.makeNew(ncStr).name) // to array of names
		.sort().value().toString();                  // to sorted csv names
	}
	Expect(csvNames).to.be.a('string');
	map[key].push({
	    src:            csvNames,
	    primarySrcList: primarySrcList
	});
	Debug(`adding: ${primarySrcList} - ${csvNames}`);
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

    Debug(args.srcList + format2(args.srcList, FIRST_COLUMN_WIDTH) +
		' ' + sources + format2(sources, SECOND_COLUMN_WIDTH) +
		' ' + args.nameList + '\n');

    // strip ":COUNT" suffix from names in nameList
    let rawNameList = args.nameList.map(name => NameCount.makeNew(name)).map(nc => nc.name);
    for (let count = 1; count < ClueManager.maxClues; ++count) {
	let map = args.nameMapArray[count];
	let dumpList = _.chain(map).keys().map(key => { // from array of name:src,src,src
	    let name = NameCount.makeNew(key).name;
	    return map[key].map(elem => {               // to array of array of elements
		elem.name = name;
		return elem;
	    });
	}).flatten().value();                           // to array of elements

	if (!_.isEmpty(dumpList)) {
	    dumpList.sort((a, b) => {
		return a.primarySrcList.toString().localeCompare(b.primarySrcList.toString());
	    });
	    dumpCompatibleClueList(dumpList, args, rawNameList);
	}
    }
//    console.log('');
}

//

function dumpCompatibleClueList (list, args, nameList = undefined) {
    list.forEach(elem => {
	if (args.format.csv) {
	    Expect(nameList).is.a('array');
	    console.log(_.concat(nameList, elem.name).sort().toString());
	} else if (args.format.files) {
	    Expect(nameList).is.a('array');
	    const nl = _.concat(nameList, elem.name);
	    let path = Result.pathFormat({
		//root:  args.root,
		dir:  _.toString(nl.length), // args.dir ||
		base: Result.makeFilename(nl.sort())
	    });
	    console.log(`'${path}'`);
	} else {
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

