//
// VALIDATOR.JS
//

'use strict';

// export a singleton

module.exports = exports = new Validator();

//

var _             = require('lodash');

var ClueManager   = require('./clue_manager');
var ClueList      = require('./clue_list');
var NameCount     = require('./name_count');
var Peco          = require('./peco');

//

function Validator() {

    this.rvsSuccessSeconds = 0;
    this.rvsFailDuration  = 0;

    this.allowDupeNameSrc = false;
    this.allowDupeSrc     = false;
    this.allowDupeName    = true;

    this.logging          = true;
    this.logLevel         = 0;

    this.PRIMARY_KEY      = '__primary';
    this.SOURCES_KEY      = '__sources';

    this.STAGING_KEY      = '__staging';
    this.COMPOUND_KEY     = '__compound';
    this.FINAL_KEY        = '__final';
}

//

Validator.prototype.log = function(text) {
    if (this.logging) {
	var pad = '';
	var index;
	for (var index=0; index<this.logLevel; ++index) {
	    pad += ' ';
	}
	console.log(pad + text);
    }
}

//

Validator.prototype.setAllowDupeFlags = function(args) {
    if (args.allowDupeNameSrc != undefined) {
	this.allowDupeNameSrc = args.allowDupeNameSrc;
    }
    if (args.allowDupeName != undefined) {
	this.allowDupeName = args.allowDupeName;
    }
    if (args.allowDupeSrc != undefined) {
	this.allowDupeSrc = args.allowDupeSrc;
    }
}

// args:
//  nameList:       list of clue names, e.g. ['bob','jim']
//  sum:            primary clue count
//  max:            max # of sources to combine (either this -or- count must be set)
//  count:          exact # of sources to combine (either this -or- max must be set)
//  require:        list of required clue counts, e.g. [2, 4]
//  exclude:        list of excluded clue counts, e.g. [3, 5]
//  excludeSrcList: list of excluded primary sources
//  validateAll:    flag; check all combinations, auto-sets wantResults
//  wantResults:    flag; return resultMap
//  quiet:          flag; quiet Peco
//
// All the primary clues which make up the clues in clueNameList should
// be unique and their total count should add up to clueCount. Verify
// that some combination of the cluelists of all possible addends of
// clueCount makes this true.

Validator.prototype.validateSources = function(args, nameCountList) {
    var clueCountListArray;
    var clueCountList;
    var index;
    var found;
    var vsCount;
    var peco;
    var resultMap;
    var vsResultMap;
    var result;

    if (args.validateAll) {
	args.wantResults = true;
    }

    if (this.logging) {
	this.log('++validateSources' +
		 ', nameList(' + args.nameList.length + 
		 '): ' + args.nameList +
		 ', nameCountList: ' + args.nameCountList +
		 ', count(' + args.count + ')' +
		 ', require: ' + args.require +
		 ', exclude: ' + args.exclude +
		 ', excludeSrcList: ' + args.excludeSrcList +
		 ', validateAll: ' + args.validateAll +
		 ', wantResults: ' + args.wantResults);
	this.log('  names: ' + args.nameList);
    }

    // TODO: pass resultmap IF wantResults is set. don't need
    // to pass both

    if (!args.resultMap) {
	resultMap = {};
    }
    else {
	resultMap = args.resultMap;
    }
    found = false;
    vsCount = args.vsCount ? args.vsCount : args.sum;

    Peco.makeNew({
	sum:     args.sum,
	count:   args.count,
	max:     args.max,
	require: args.require,
	exclude: args.exclude,
	quiet:   args.quiet
    }).getCombinations().some(clueCountList => {
	var rvsResult;

	rvsResult = this.recursiveValidateSources({
	    clueNameList:  args.nameList,
	    clueCountList: clueCountList,
	    nameCountList: nameCountList,
	    excludeSrcList:args.excludeSrcList,
	    validateAll:   args.validateAll,
	    wantResults:   args.wantResults,
	    vsCount:       vsCount,
	    resultMap:     resultMap
	});
	if (rvsResult.success) {
	    if (this.logging) {
		this.log('recursiveValidateSources: VALIDATE SUCCESS!');
	    }
	    vsResultMap = rvsResult.resultMap;

	    // this is a non-recursive success that meets all constraints
	    found = true;

	    // We may have been called recursively. Don't set 'found' flag
	    // in this case, because we don't know if we meet require/exlude
	    // constraints
	    if (args.recursive) {
		// TODO: in this case, there are potentially additional
		// combinations for the names in nameList. We're skipping
		// them which is bad.

		return true; // some.exit, return to checkUniqueSources()
	    }

	    // we may not care about all possible combinations
	    if (!args.validateAll) {
		return true; // found a match; some.exit
	    }

	    // validing all, continue searching
	    if (this.logging) {
		this.log('recursiveValidateSources: validateAll set, continuing...');
	    }
	}
	return false; // some.continue
    }, this);

    if (this.logging) {
	if (nameCountList) {
	    this.log('--validateSources' +
		     ', nameCountList(' + nameCountList.length + ')' +
		     ', ' + nameCountList);
	}
    }

    return {
	resultMap:   resultMap,
	vsResultMap: vsResultMap,
	success:     found
    }
}

// args:
//  clueNameList:
//  clueCountList:
//  nameCountList:
//  excludeSrcList: list of excluded primary sources
//  validateAll:
//  wantResults:
//  resultMap:      args.resultMap
//  vsCount:        pass-through
//
Validator.prototype.recursiveValidateSources = function(args) {
    var nameIndex;
    var clueName;
    var count;
    var someResult;
    var rvsResult;

    if (!_.has(args, 'vsCount')) {
	throw new Error ('missing args');
    }

    this.logLevel++;

    if (!args.nameCountList) {
	args.nameCountList = [];
    }
    if (this.logging) {
	this.log('name: looking for [' + args.clueNameList + '] in [' + args.clueCountList + ']');
    }

    if (args.clueNameList.length != args.clueCountList.length) {
	throw new Error('Mismatched list lengths');
    }

    nameIndex = 0;
    clueName = args.clueNameList[nameIndex];

    // optimization: could have a map of count:boolean entries here
    // on a per-name basis (new map for each outer loop; once a
    // count is checked for a name, no need to check it again

    someResult = args.clueCountList.some(count => {

	if (this.logging) {
	    this.log('count: looking for ' + clueName + ' in ' + count);
	}

	if (!ClueManager.knownClueMapArray[count][clueName]) {
	    if (this.logging) {
		this.log(' not found, ' + clueName + ':' + count);
	    }
	    return false; // some.continue
	}

	if (this.logging) {
	    this.log(' found');
	}

	if (0 && this.logging) {
	    this.log('++rvsWorker input for:' + clueName +
		     ', ncList(' + args.nameCountList.length + ') ' +
		     ' ' + args.nameCountList);
	}

	rvsResult = this.rvsWorker({
	    name:           clueName,
	    count:          count,
	    nameList:       args.clueNameList,
	    countList:      args.clueCountList,
	    ncList:         args.nameCountList,
	    excludeSrcList: args.excludeSrcList,
	    validateAll:    args.validateAll,
	    wantResults:    args.wantResults,
	    vsCount:        args.vsCount,      // aka "original count"
	    resultMap:      args.resultMap
	});
	if (!rvsResult.success) {
	    return false; // some.continue;
	}

	if (this.logging) {
	    this.log('--rvsWorker output for:' + clueName +
		     ', ncList(' + args.nameCountList.length + ') ' +
		     ' ' + args.nameCountList);
	}

	if (!args.validateAll && (args.nameCountList.length < 2)) {
	    // TODO: add "allowSingleEntry" ? or was this just a sanity check?
	    // can i check vs. clueNameList.length?
	    // throw new Error('list should have at least two entries1');
	}
	return true; // success: some.exit
    });
    this.logLevel--;

    return someResult ? {
	resultMap: rvsResult.resultMap,
	success:   true
    } : {
	success: false
    }
}

// args:
//   name     : clueName,
//   count    : count,
//   nameList : clueNameList,
//   countList: clueCountList,
//   ncList   : nameCountList,
//   excludeSrcList: list of excluded primary sources
//   validateAll:
//   wantResults:
//   resultMap:
//   vsCount:       pass-through
//
Validator.prototype.rvsWorker = function(args) {
    var newNameCountList;
    var uniqResult;
    var rvsResult;
    var nc;

    if (!args.vsCount) {
	throw new Error ('missing args');
    }

    newNameCountList = this.copyAddNcList(args.ncList, args.name, args.count);
    if (!newNameCountList) {
	// duplicate name:count entry. technically this is allowable for
	// count > 1 if the there are multiple entries of this clue name
	// in the clueList[count]. (at least as many entries as there are
	// copies of name in ncList)
	if (this.logging) {
	    this.log('duplicate name:count, ' + args.name + ':' + args.count);
	}
	return { success: false }; // fail
    }

    if (this.logging) {
	this.log('added NC name: ' + args.name +
		 ', count = ' + args.count +
		 ', length = ' + newNameCountList.length);
    }

    // If only one name & count remain, we're done.
    // (name & count lists are equal length, just test one)
    if (args.nameList.length == 1) {
	uniqResult = this.checkUniqueSources(newNameCountList, args);
	if (uniqResult.success) {

	    if (this.logging) {
		this.log('checkUniqueSources --- success!\n' +
			 '---------------------------------------------------');
	    }

	    nc = NameCount.makeNew(args.name, args.count);
	    args.ncList.push(nc);

	    if (this.logging) {
		this.log('add1, ' + args.name + ':' + args.count +
			 ', newNc(' + newNameCountList.length + ')' +
			 ', ' + newNameCountList);
	    }

	    return {
		success:   true,
		resultMap: uniqResult.resultMap
	    };
	}

	if (this.logging) {
	    this.log('checkUniqueSources --- failure\n' +
		     '---------------------------------------------------');
	}
	return { success: false }; // fail
    }

    // nameList.length > 1, remove current name & count,
    // and validate remaining
    rvsResult = this.recursiveValidateSources({
	clueNameList:  this.chop(args.nameList, args.name),
	clueCountList: this.chop(args.countList, args.count),
	nameCountList: newNameCountList,
	excludeSrcList:args.excludeSrcList,
	validateAll:   args.validateAll,
	wantResults:   args.wantResults,
	vsCount:       args.vsCount,      // aka "original count"
	resultMap:     args.resultMap
    });
    if (!rvsResult.success) {
	if (this.logging) {
	    this.log('recursiveValidateSources failed');
	}
	return { success: false }; // fail
    }

    // does this achieve anything? modifies args.ncList.
    // TODO: probably need to remove why that matters.
    args.ncList.length = 0;
    newNameCountList.forEach(nc => args.ncList.push(nc));

    if (this.logging) {
	this.log('add2, ' + args.name + ':' + args.count +
		 ', newNc(' + newNameCountList.length + ')' +
		 ', ' + newNameCountList);
    }

    return {
	success:   true,
	resultMap: rvsResult.resultMap
    };
}

// args:
//   name:           clueName,
//   count:          # of primary clues represented by cluename?
//   nameList:
//   countList:
//   ncList:
//   excludeSrcList: list of excluded primary sources
//   validateAll:
//   wantResults:
//   resultMap:
//   vsCount:       pass-through
///
Validator.prototype.checkUniqueSources = function(nameCountList, args) {
    var origNcList = nameCountList;
    var srcNcList;
    var nameSrcList;
    var buildArgs;
    var findResult;
    var buildResult;
    var cycleList;
    var resultMap;
    var vsResult;
    var ncListCsv;
    var pendingMap;
    var rootNcStr;

    var saveNcList;
    var anyFlag;
    var anyNcList;

    if (!args.vsCount) {
	throw new Error ('missing args');
    }

    // assert(nameCountList) && Array.isArray(nameCountList)

    if (this.logging) {
	this.log('++checkUniqueSouces' +
		 ', name: ' + args.name +
		 ', count: ' + args.count +
		 ', nameList: ' + args.nameList +
		 ', ncList: ' + nameCountList);
    }

    // TODO: or just pass in as arg, like a sane person
    rootNcStr = _.toString(NameCount.makeNew(args.name, args.count));

    resultMap = {};
    pendingMap = {};
    buildArgs = {
	excludeSrcList: args.excludeSrcList,
	wantResults:    args.wantResults,
	ncList:         nameCountList,
	resultMap:      args.resultMap
    };

//	nameCountList = origNcList;

	// seems like i'm still doing this too much.  either i just
	// entered the function, or we've just incremented the index
	// map. if the latter, we haven't built a new source list yet
	// so it seems this is exactly the same check as we did on entry.

	// first, check for duplicate primary clues, add all to primaryMap
	findResult = this.findDuplicatePrimaryClue({
	    ncList:              nameCountList,
	});
	if (!this.evalFindDuplicateResult(findResult, '1st')) {
	    if (this.logging) {
		this.log('FAIL - duplicate primary' +
			 ', nameCountList; ' + nameCountList);
	    }
	    return this.uniqueResult(false); // failure
	}
	else if (findResult.allPrimary) {
	    pendingMap = this.allUniquePrimary({
		origNcList:     nameCountList,
		ncList:         nameCountList,
		// TODO: return this list from findDupe, get rid of srcMap? result.primaryNcList?
		nameSrcList:    this.getNameSrcList(findResult.srcMap), //primaryClueData.srcMap),
		map:            args.resultMap,
		excludeSrcList: args.excludeSrcList,
		wantResults:    args.wantResults,
		vsCount:        args.vsCount,
	    });

	    if (pendingMap) {
		this.mergePending(resultMap, pendingMap);
	    }
	    return this.uniqueResult(true, resultMap, nameCountList);
	}

    for(;;) {
	if (buildResult) {
	    buildArgs.indexMap = buildResult.indexMap;
	}
	nameCountList = origNcList;

	for (;;) {
	    // break down all compound clues into source components
	    buildArgs.ncList = nameCountList;
	    buildResult = this.buildSrcNameList(buildArgs);

	    // skip recursive call to validateSources if we have all primary clues
	    if (buildResult.allPrimary) {

		ncListCsv = _.toString(nameCountList);
		if (this.logging) {
		    this.log('cUS: adding all_primary result: ' +
			     ncListCsv + ' = ' + buildResult.primaryNcList);
		}
		pendingMap = buildResult.srcMap;
		saveNcList = nameCountList;
		nameCountList = buildResult.primaryNcList;
	    }
	    else {
		// call validateSources recursively with compound clues
		srcNcList = [];
		// TODO: use vsResult.success property, not null on failure
		vsResult = this.validateSources({
		    sum:            buildResult.count,
		    nameList:       buildResult.compoundSrcNameList,
		    count:          buildResult.compoundSrcNameList.length,
		    excludeSrcList: args.excludeSrcList,
		    validateAll:    args.validateAll,
		    wantResults:    args.wantResults,
		    //vsCount:        args.vsCount,
		    resultMap:      args.resultMap,
		    recursive:      true
		}, srcNcList);
		if (!vsResult.success) {
		    break; // fail, try other combos
		}
		// sanity check
		if (/*!args.validateAll && */(srcNcList.length < 2)) {
		    throw new Error('list should have at least two entries2');
		}

		// Some kind of success needs to be indicated here, if
		// validateSources has returned true.
		// VERY QUESTIONABLE but helps me test
		anyFlag = true;

		// we only sent compound clues to validateSources, so add the
		// primary clues that build filtered out to make a complete list

		saveNcList = nameCountList;
		nameCountList = srcNcList;
		// TODO: this is good if validateSoruces returns primary only
		nameCountList = _.concat(srcNcList, buildResult.primaryNcList);
		
		console.log('from validateSources(' + buildResult.count + ')' +
			    ' compoundSrcNameList: ' + buildResult.compoundSrcNameList +
			   ', srcNcList: ' + srcNcList);
		this.dumpResultMap(vsResult.vsResultMap);

		pendingMap = vsResult.vsResultMap;
	    }

	    findResult = this.findDuplicatePrimaryClue({ ncList: nameCountList });
	    if (!this.evalFindDuplicateResult(findResult, '2nd')) {
		break; // fail, try other combos
	    }

	    if (!findResult.allPrimary) {
		// I think the only way this can happen is if we are unwinding
		// a successful recursive call to validateSources. The returned
		// resultMap is our new resultMap.

		// FALSE. Currently validateSources does not always return primary
		// sources (it should). It just finds "counts" for a namelist,
		// regardless of count size.
		// break out of innerloop here

		this.log('!allPrimary, pending: ' + pendingMap);

		// not so sure about this!
		resultMap = pendingMap;
		continue;
	    }

	    // TODO:
	    //  if ((this.addResultAndCyclePrimary()) break; }
	    // all the source clues we just validated are primary clues
	    if (args.wantResults || args.validateAll) {
		nameSrcList = this.getNameSrcList(findResult.srcMap);
		if (_.isEmpty(nameSrcList)) {
		    throw new Error('empty nameSrclist for: ' + nameCountList);
		}
		ncListCsv = _.toString(nameCountList);
		if (args.wantResults) {
		    if (this.addFinalResult({
			map:           args.resultMap,
			ncList:        nameSrcList,
			excludeSrcList:args.excludeSrcList,
			vsCount:       args.vsCount,
			stagingNcList: nameCountList
		    })) {
			if (this.logging) {
			    this.log('cUS: adding 2nd primary result: ' +
				     ncListCsv + ' = ' + nameSrcList);
			}
			this.addPendingResult({
			    pendingMap:     pendingMap,
			    //rootKey:        rootNcStr,
			    origNcList:     saveNcList,
			    primaryNcList:  nameCountList,
			    nameSrcList:    nameSrcList
			});
		    }
		    else {
			if (logging) {
			    this.log('AddFinalResult failed');
			}
		    }
		}
		if (args.validateAll) {
		    if (cycleList = this.cyclePrimaryClueSources({
			resultMap:     args.resultMap,
			ncList:        nameSrcList,
			vsCount:       args.vsCount,
			excludeSrcList:args.excludeSrcList
		    })) {
			if (this.logging) {
			    this.log('cUS: adding 2nd cycle result: ' +
				     ncListCsv + ' = ' + cycleList);
			}
			cycleList.forEach(nameSrcList => {
			    this.addPendingResult({
				pendingMap:     pendingMap,
				//rootKey:        rootNcStr,
				origNcList:     saveNcList,
				primaryNcList:  nameCountList,
				nameSrcList:    nameSrcList
			    });
			});
		    }
		}

		console.log('AFTER pendingMap:');
		this.dumpResultMap(pendingMap);

		this.mergePending(resultMap, pendingMap);

		if (args.validateAll) {
		    anyFlag = true;
		    break; // success - try other combos
		}
	    }
	    return this.uniqueResult(anyFlag, resultMap, anyNcList); // success - exit function
	}

	if (!this.incrementIndexMap(buildResult.indexMap)) {
	    if (this.logging) {
		this.log('done, success: ' + anyFlag);
	    }
	    return this.uniqueResult(anyFlag, resultMap, anyNcList);
	    /*
	    if (anyFlag) {
		return resultMap;
	    }
	    return false;
	    */
	}

	if (this.logging) {
	    this.log('++outer looping');
	}
    }
}

Validator.prototype.uniqueResult = function(success, map, list) {
    return {
	success:   success,
	resultMap: map,
	ncList:    list
    };
}


// origNcList:
// primaryNcList:
// nameSrcList:
// resultMap:      args.resultMap,
// excludeSrcList: args.excludeSrcList,
// vsCount:        args.vsCount,
// wantResults:    args.wantResults,
//

Validator.prototype.allUniquePrimary = function(args) {
    var pendingMap;
    var ncListCsv;
    var cycleList;

    // no duplicates, and all clues are primary, success!
    if (this.logging) {
	this.log('SUCCESS! all primary on entry, ' + args.nameCountList);
    }

    if (args.vsCount != args.ncList.length) {
	if (this.logging) {
	    this.log('vsCount: ' + args.vsCount +
		     ', ncList: ' + args.ncList);
	}
	throw new Error('logic dictates this is impossible');
    }

    pendingMap = {};
    ncListCsv = _.toString(args.ncList);
    if (this.addFinalResult({
	map:           args.map,
	ncList:        args.ncList,
	excludeSrcList:args.excludeSrcList,
	vsCount:       args.vsCount,
	stagingName:   'whatever', //args.name,
	stagingNcList: args.ncList
    })) {
	if (args.wantResults) {
	    if (this.logging) {
		this.log('aPU: adding primary result: ' +
			 ncListCsv + ' = ' + args.ncList);
	    }
	    this.addPendingResult({
		pendingMap:     pendingMap,
		origNcList:     args.ncList,
		primaryNcList:  args.ncList,
		nameSrcList:    args.nameSrcList
	    });
	}
    }
    if (args.validateAll) {
	if (cycleList = this.cyclePrimaryClueSources({
	    ncList:        args.ncList,
	    resultMap:     args.resultMap,
	    excludeSrcList:args.excludeSrcList,
	    vsCount:       args.vsCount
	})) {
	    if (args.wantResults) {
		cycleList.forEach(nameSrcList => {
		    this.addPendingResult({
			pendingMap:     pendingMap,
			origNcList:     args.ncList,
			primaryNcList:  args.ncList,
			nameSrcList:    args.nameSrcList
		    });
		});
	    }
	}
    }
    return pendingMap;
}

//pendingMap:     pendingMap,
//rootKey
//origNcList:
//primaryNcList:
//nameSrcList:    nameSrcList
//

Validator.prototype.addPendingResult = function(args) {
    var primaryNcList;
    var nameSrcList;
    var result;
    
    primaryNcList = _.clone(args.primaryNcList);
    nameSrcList = _.clone(args.nameSrcList);

    if (this.logging) {
	this.log('++addPendingResult' +
		 ', rootKey: ' + args.rootKey +
		 ', origNcList: ' + args.origNcList +
		 ', primaryNcList: ' + primaryNcList +
		 ', nameSrcList: ' + nameSrcList);
	this.dumpResultMap(args.pendingMap);
    }

    this.addAllPendingPrimary(args.pendingMap, args.origNcList, primaryNcList, nameSrcList);

    if (args.rootKey) {
	this.addPendingResultToRoot({
	    pendingMap:     args.pendingMap,
	    origNcList:     args.origNcList,
	    primaryNcList:  primaryNcList,
	    nameSrcList:    nameSrcList
	});
    }
    else {
	this.addPendingResultToNcList({
	    pendingMap:     args.pendingMap,
	    origNcList:     args.origNcList,
	    primaryNcList:  primaryNcList,
	    nameSrcList:    nameSrcList
	});
    }

    if (this.logging) {
	console.log('--addPendingResult');
	this.dumpResultMap(args.pendingMap);
    }
}

//
//pendingMap:     pendingMap,
//rootKey
//origNcList:
//primaryNcList:
//nameSrcList:    nameSrcList
//
Validator.prototype.addPendingResultToNcList = function(args) {
    var pathMap;

    pathMap = {};
    args.origNcList.forEach(nc => {
	var pathList;
	var ncStr = _.toString(nc);
	// ensure all origNc entries exit
	if (!args.pendingMap[ncStr]) {
 	    throw new Error('pendingMap missing nc key, ' + ncStr);
	}
	pathList = this.recursiveGetPendingPrimaryPathList(ncStr, args.pendingMap[ncStr]);
	if (!pathList) {
	    throw new Error('no primary paths found in, ' + ncStr);
	}
	if (this.logging) {
	    this.log('pathList for: ' + ncStr);
	    pathList.forEach(path => {
		this.log('  path: ' + path.path + ', primary: ' + path.primaryNcCsv);
	    });
	}
	this.addSourcesToPendingPathList(pathList, args.nameSrcList);
    });
}

//
//

Validator.prototype.addSourcesToPendingPathList = function(pathList, nameSrcList) {
    pathList.forEach(path => {
	var index;
	path.nameSrcList = [];
	NameCount.makeListFromCsv(path.primaryNcCsv).forEach(nc => {
	    index = _.findIndex(nameSrcList, { name : nc.name });
	    if (index === -1) {
		throw new Error('primary clue not in nameSrcList, ' + nc.name +
				', list: ' + nameSrcList);
	    }
	    path.nameSrcList.push(nameSrcList[index]);
	    _.pullAt(nameSrcList, [ index ]);
	});
    });
}

//
//

Validator.prototype.recursiveGetPendingPrimaryPathList = function(path, map) {
    var pathList;
    pathList = [];
    _.forOwn(map, (value, key) => {
	if (_.isArray(value)) { // array means primary (i hope)
	    pathList.push({
		path:         path,
		primaryNcCsv: key
	    });
	}
	else {
	    _.concat(pathList, this.recursiveGetPendingPrimaryPathList(path + '.' + key));
	}
    });
    return pathList;
}


//
//pendingMap:
//rootKey
//origNcList:
//primaryNcList:
//nameSrcList:
//
Validator.prototype.addPendingResultToRoot = function(args) {
    var map;
    var listKey;
    var list;

    // ensure root key exists
    if (!args.pendingMap[args.rootKey]) {
	throw new Error('pendingMap missing rootKey, ' + args.rootKey);
    }
    map = args.pendingMap[args.rootKey];
    listKey = _.toString(args.primaryNcList);

    if (!map[listKey]) {
	throw new Error('pendingMap rootKey: ' + args.rootKey +
			', missing list: ' + listKey);
    }

    list = map[listKey];
    if (!_.isArray(list)) {
	throw new Error('pendingMap rootKey: ' + args.rootKey +
			' list: ' + listKey +
			' is not an array, type: ' + (typeof list));

    }
    list.push(_.toString(args.nameSrcList));
}

//
//

Validator.prototype.addAllPendingPrimary = function(map, origNcList, primaryNcList, nameSrcList) {
    // find primary NCs in the orignal NC list
    origNcList.forEach((nc, ncIndex) => {
	var primaryIndex;
	var ncStr;
	if (nc.count > 1) {
	    return; // forEach.continue
	}
	// is original primary NC in supplied primaryNcList?
	if (!_.includes(primaryNcList, nc)) {
	    return; // forEach.continue
	}
	ncStr = _.toString(nc);
	// try to add the primary NC to pendingMap
	this.addPendingPrimary(map, ncStr, _.toString(nameSrcList[ncIndex]));
	primaryIndex = _.indexOf(primaryNcList, nc);
	if (primaryIndex === -1) {
	    throw new Error('alternate universe, nc: ' + ncStr +
			    ', primaryNcList: ' + primaryNcList);
	}
	// remove the primary NC and it's corresponding name:src from
	// the supplied lists
	_.pullAt(primaryNcList, [ primaryIndex ]);
	_.pullAt(nameSrcList, [ primaryIndex ]);
    });

    if (this.logging) {
	this.log('--addAllPendingPrimary' +
		 ', primaryNcList: ' + primaryNcList +
		 ', nameSrcList: ' + nameSrcList);
    }
}

//
//

Validator.prototype.addPendingPrimary = function(map, ncPrimaryStr, nameSrcStr) {
    var list;

    if (!map[ncPrimaryStr]) {
	if (!this.resolvePendingPrimary(map, ncPrimaryStr)) {
	    throw new Error('failure to resolve pending primary, ' + ncPrimaryStr +
			    ', for: ' + nameSrcStr);
	}
    }
    list = map[ncPrimaryStr] = [];
    if (!_.isArray(list)) {
	throw new Error('pending primary list, ' + list +
			' is not an array, type: ' + (typeof list));
    }
    list.push(nameSrcStr);
}

//
//

Validator.prototype.resolvePendingPrimary = function(map, ncPrimaryStr) {
    var primaryNcStrList;
    var index;

    primaryNcStrList = map[this.PRIMARY_KEY];
    if (!primaryNcStrList || _.isEmpty(primaryNcStrList)) {
	if (this.logging) {
	    this.log('missing or empty unresolved primary list, ' + primaryNcStrList +
		     ', for nc: ' + ncPrimaryStr);
	    this.log('map.keys: ' + _.keys(map));
	}
	return false;
    }

    index = _.indexOf(primaryNcStrList, ncPrimaryStr);
    if (index === -1) {
	if (this.logging) {
	    this.log('nc not in unresolved list, ' + ncPrimaryStr +
		     ', list: ' + primaryNcStrList);
	}
	return false;
    }

    if (this.logging) {
	this.log('found unresolved pending primary nc: ' + ncPrimaryStr +
		 ', at index: ' + index);
    }

    _.pullAt(primaryNcStrList, [ index ]);
    return true;
}

//
//

Validator.prototype.mergePending = function(resultMap, pendingMap) {
    var rootKey;
    var subKey;
    var key;
    var map;
    var list;

    console.log('mergePending');

    console.log('before pendingMap:');
    this.dumpResultMap(pendingMap);
    console.log('before resultMap:');
    this.dumpResultMap(resultMap);

    this.recursiveMergePending(resultMap, pendingMap);

    /*
    rootKey = Object.keys(pendingMap)[0];
    subKey = Object.keys(pendingMap[rootKey])[0];
    console.log('mergePending, subKey: ' + subKey);

    if (!resultMap[key]) {
	map = resultMap[key] = {};
    }
    else {
	map = resultMap[key];
    }

    if (rootKey != key) {
	if (!map[rootKey]) {
	    map[rootKey] = {};
	}
	map = map[rootKey];
    }

    if (!map[subKey]) {
	map[subKey] = [];
    }

    return;

    pendingMap[rootKey][subKey].forEach(item => {
	map[subKey].push(item);
    });
    */

    console.log('after resultMap:');
    this.dumpResultMap(resultMap);
}

//
//

Validator.prototype.recursiveMergePending = function(resultSeq, pendingSeq) {
    // empty resultSeq, add everything from pendingSeq
    if (_.isEmpty(resultSeq)) {
	_.forEach(pendingSeq, (value, key) => {
	    resultSeq[key] = value;
	});
	return;
    }

    if (_.isArray(pendingSeq) != _.isArray(resultSeq)) {
	throw new Error('array/object type mismatch');
    }

    if (_.isArray(resultSeq)) {
    }
    else {
	_.forOwn(pendingSeq, (value, key) => {
	    if (_.has(resultSeq, key)) {
		this.recursiveMergePending(resultSeq[key], pendingSeq[key]);
	    }
	    else {
		resultSeq[key] = value;
	    }
	});
    }
}

// Simplified version of checkUniqueSources, for all-primary clues.
//
// args:
//  ncList:
//  exclueSrcList:
//  resultMap:      args.resultMap
//
Validator.prototype.cyclePrimaryClueSources = function(args) {
    var srcMap;
    var buildArgs;
    var buildResult;
    var findResult;
    var localNcList;
    var resultList;
    var ncListCsv;
    var nameSrcList;

    if (!args.vsCount ||
	!args.ncList || _.isEmpty(args.ncList)) {
	throw new Error ('missing args' +
			 ', ncList: ' + args.ncList +
			 ', vsCount: ' + args.vsCount);
    }

    this.log('++cyclePrimaryClueSources');

    // must copy the NameCount objects within the list
    localNcList = [];
    args.ncList.forEach(nc => {
	localNcList.push(NameCount.makeCopy(nc));
    });
    buildArgs = {
	excludeSrcList: args.excludeSrcList,
	ncList:         args.ncList,   // always pass same unmodified ncList
	resultMap:      args.resultMap,
	allPrimary:     true
    };
    do {
	// build src name list of any duplicate-sourced primary clues
	if (buildResult) {
	    buildArgs.indexMap = buildResult.indexMap;
	}
	buildResult = this.buildSrcNameList(buildArgs);

	// change local copy of ncList sources to buildResult's sources
	localNcList.forEach((nc, index) => {
	    nc.count = buildResult.primarySrcNameList[index];
	});

	srcMap = {};
	findResult = this.findDuplicatePrimarySource({
	    ncList: localNcList,
	    srcMap: srcMap
	});
	if (findResult.duplicateSrc) {
	    continue;
	}
	if (localNcList.length != Object.keys(srcMap).length) {
	    throw new Error('localNcCount.length != keys.length!' +
			    ', findResult.duplicateSrc:' + findResult.duplicateSrc);
	}

	if (this.logging) {
	    this.log('[srcMap] primary keys: ' + this.getNameSrcList(srcMap));
	}

	// all the source clues we just validated are primary clues
	nameSrcList = this.getNameSrcList(srcMap);
	if (this.addFinalResult({
	    map:           args.resultMap,
	    ncList:        nameSrcList,
	    excludeSrcList:args.excludeSrcList
	})) {
	    ncListCsv = _.toString(localNcList);
	    if (this.logging) {
		this.log('cycle: adding result: ' +
			 ncListCsv + ' = ' + nameSrcList);
	    }
	    if (!resultList) {
		resultList = [];
	    }
	    resultList.push(this.getNameSrcList(srcMap));
	}
    } while (this.incrementIndexMap(buildResult.indexMap));

    this.log('--cyclePrimaryClueSources');

    return resultList; // return real list eventually
}

// ncList:              nameCountList
// clueData:            primaryClueData
//
Validator.prototype.findDuplicatePrimaryClue = function(args) {
    var duplicateName;
    var duplicateSrc;
    var duplicateSrcName;
    var allPrimary;
    var key;
    var findResult;
    var resolveResult;

    if (this.logging) {
	this.log('++findDuplicatePrimaryClue' +
		    ', ncList: ' + args.ncList);
    }

    // look for duplicate primary clue sources, return conflict map
    // also checks for duplicate names
    findResult = this.findPrimarySourceConflicts({ncList: args.ncList });
    duplicateName = findResult.duplicateName;
    allPrimary = findResult.allPrimary;

    if (!_.isEmpty(findResult.conflictSrcMap)) {
	// resolve duplicate primary source conflicts
	resolveResult = this.resolvePrimarySourceConflicts({
	    srcMap:         findResult.srcMap,
	    conflictSrcMap: findResult.conflictSrcMap
	});
	duplicateSrcName = resolveResult.duplicateSrcName;
	duplicateSrc = resolveResult.duplicateSrc;
    }
    if (this.logging) {
	this.log('--findDuplicatePrimaryClue, duplicateName: ' + duplicateName +
		 ', duplicateSrc: ' + duplicateSrc +
		 ', allPrimary: ' + allPrimary +
		 ', srcMap.size: ' + _.size(findResult.srcMap));
    }

    return {
	duplicateName:    duplicateName,
	duplicateSrcName: duplicateSrcName,
	duplicateSrc:     duplicateSrc,
	allPrimary:       allPrimary,
	srcMap:           findResult.srcMap
    };
}


// args:
//  ncList:
//
// result:
//

Validator.prototype.findPrimarySourceConflicts = function(args) {
    var duplicateName;
    var allPrimary;
    var nameMap;
    var srcMap;
    var conflictSrcMap;

    if (!args.ncList) {
	throw new Error('missing args' + ', ncList: ' + args.ncList);
    }

    allPrimary = true;
    nameMap = {};
    srcMap = {};
    conflictSrcMap = {};

    args.ncList.forEach(nc => {
	var srcList;

	if (nc.count != 1) {
	    if (this.logging) {
		this.log('fPSC: non-primary, ' + nc);
	    }
	    allPrimary = false;
	    return; // forEach.continue
	}

	// if name is in nameMap then it's a duplicate
	if (nameMap[nc.name]) {
	    duplicateName = nc.name;
	}
	srcList = ClueManager.knownClueMapArray[1][nc.name];
	// look for an as-yet-unused src for the given clue name
	if (!srcList.some(src => {
	    if (!srcMap[src]) {
		srcMap[src] = nc.name;
		return true; // found; some.exit
	    }
	    return false; // not found; some.continue
	})) {
	    // unused src not found: add to conflict map, resolve later
	    if (!conflictSrcMap[srcList]) {
		conflictSrcMap[srcList] = [];
	    }
	    conflictSrcMap[srcList].push(nc.name);
	}
    }, this);

    if (this.logging) {
	this.log('findPrimarySourceConflicts: ' +
		 (duplicateName ? duplicateName : 'none') +
		 ', allPrimary: ' + allPrimary);
    }

    return {
	duplicateName: duplicateName,
	allPrimary:    allPrimary,
	srcMap:        srcMap,
	conflictSrcMap:conflictSrcMap
    };
}

// args:
//  srcMap:
//  conflictSrcMap:
//

Validator.prototype.resolvePrimarySourceConflicts = function(args) {
    var duplicateSrcName;
    var duplicateSrc;
    var conflictSrc;
    var conflictNameList;
    var srcList;
    var key;

    if (!args.srcMap || !args.conflictSrcMap) {
	throw new Error('missing args' +
			', srcMap:' + args.srcMap +
			' conflictSrcMap: ' + args.conflictSrcMap);
    }

    // resolve primary source conflicts
    for (conflictSrc in args.conflictSrcMap) {

	if (this.logging) {
	    this.log('Attempting to resolve conflict...');
	}

	srcList = conflictSrc.split(',');
	conflictNameList = args.conflictSrcMap[conflictSrc];

	// if conflictList.length > srcList.length then there
	// are more uses of this clue than there are sources for it.
	if (conflictNameList.length > srcList.length) {
	    duplicateSrcName = conflictNameList.toString();
	    duplicateSrc = conflictSrc;
	    break;
	}
	// otherwise we may be able to support the clue count; see
	// if any conflicting clue names can be moved to other sources
	if (!srcList.some(src => {
	    var conflictSrcList;
	    var candidateName;
	    var candidateSrcList;

	    // look for alternate unused sources for candidateName
	    candidateName = args.srcMap[src];
	    candidateSrcList = ClueManager.knownClueMapArray[1][candidateName];
	    if (candidateSrcList.some(candidateSrc => {
		if (!args.srcMap[candidateSrc]) {
		    if (this.logging) {
			this.log('resolved success!');
		    }
		    // success! update srcMap
		    args.srcMap[candidateSrc] = candidateName;
		    // any name will do?!
		    args.srcMap[src] = conflictNameList.pop();
		    if (conflictNameList.length == 0) {
			return true; // candidateSrcList.some.exit
		    }
		}
		return false; // candidateSrcList.some.next
	    })) {
		return true; // srcList.some.stop
	    }
	    return false; // srcList.some.next
	}, this)) {
	    // failed to find an alternate unused source for all conflict names
	    duplicateSrcName = _.toString(conflictNameList);
	    duplicateSrc = conflictSrc;

	    if (this.logging) {
		this.log('cannot resolve conflict, ' + conflictNameList);
		this.log('used sources, ');
		for (key in args.srcMap) {
		    this.log('  ' + key + ':' + args.srcMap[key]);
		}
	    }
	    break;
	}
    }
    return {
	duplicateSrcName: duplicateSrcName,
	duplicateSrc:     duplicateSrc
    };
}

// args:
//  ncList:          args.ncList, // ALL PRIMARY clues in name:source format (not name:count)
//  srcMap:
//  nameMap:
//
Validator.prototype.findDuplicatePrimarySource = function(args) {
    var duplicateSrcName;
    var duplicateSrc;

    args.ncList.some(nc => {
	var src;
	src = nc.count;
	if (args.srcMap[src]) {
	    // duplicate source
	    duplicateSrcName = nc.name;
	    duplicateSrc = src;
	    return true; // some.exit
	}
	// mark source used
	args.srcMap[src] = nc.name;
	return false; // some.continue
    });
    return {
	duplicateSrcName: duplicateSrcName,
	duplicateSrc:     duplicateSrc
    };
}

//
//

Validator.prototype.evalFindDuplicateResult = function(result, logPrefix) {
    var dupeType = '';
    var dupeValue = '';

    if (this.logging) {
	if (result.duplicateName || result.duplicateSrc) {
	    this.log('duplicate name: ' + result.duplicateName +
			', src: ' + result.duplicateSrc);
	}
    }

    if (!this.allowDupeName && result.duplicateName) {
	dupeType = 'name';
	dupeValue = result.duplicateName;
    }
    if (!this.allowDupeSrc && result.duplicateSrc) {
	if (dupeType.length) {
	    dupeType += ', ';
	    dupeValue += ', ';
	}
	dupeType += 'source';
	dupeValue += result.duplicateSrcName + '(' + result.duplicateSrc + ')';
    }

    // NOTE: need some extra logic here to support:'
    // * NO dupe src with 2-source clue
    // * NO dupe name with 2-source clue

    if (dupeType.length) {
	if (this.logging) {
	    this.log(logPrefix + ' duplicate primary ' + dupeType + ', ' + dupeValue);
	}
	else {
	    //console.log(logPrefix + ' duplicate primary ' + dupeType + ', ' + dupeValue);
	}
	return false;
    }
    return true;
}

// args:
//  ncList
//  excludeSrcList:
//  wantResults:
//  resultMap:
//  allPrimary:  boolean
//  indexMap:
//
// Divide clues in ncList into known compound and known primary sources.
//
// compoundSrcNameList: compound source clue names
// compoundClueCount:   # of primary clues of which compoundSrcNameList consists
// primaryNcList:       primary source clue NCs
// compoundNcList:      subset of args.ncList which contains only compound NCs
//
// NOTE: is problem below solved now?
// So there is some potenetial trickiness here for a 100% correct solution.
// If a clue has multiple source combinations, we technically need to build
// a separate clueNameList for each possible combination. if two or more clues
// have multiples source combinations, we need to build all combinations of those
// combinations. for the second case, punt until it happens.
//
Validator.prototype.buildSrcNameList = function(args) {
    var allPrimary;   // all *sources* primary
    var clueCount;    // # of primary clues represented by component source clues
    var indexMap;
    var compoundSrcNameList;
    var compoundNcList;
    var primaryNcList;
    var primarySrcNameList;
    var srcMap;
    var primaryPathList;

    if (this.logging) {
	this.log('++buildSrcNameList, ncList(' + args.ncList.length + ')' +
		 ', ncList: ' + args.ncList +
		 ', allPrimary: ' + args.allPrimary);
    }

    indexMap = this.getIndexMap(args.indexMap);

    allPrimary = true;
    clueCount = 0;
    compoundNcList = [];
    compoundSrcNameList = [];
    primaryNcList = [];
    primarySrcNameList = [];
    primaryPathList = [];
    srcMap = {};

    args.ncList.forEach((nc, ncIndex) => {
	var srcList;          // e.g. [ 'src1,src2,src3', 'src2,src3,src4' ]
	var slIndex;          // srcListIndex
	var srcNameList;      // e.g. [ 'src1', 'src2', 'src3' ]
	var src;
	var localPrimaryNcList;
	var map;
	var path;

	src = args.allPrimary ? 1 : nc.count;
	srcList = ClueManager.knownClueMapArray[src][nc.name];
	if (!srcList) {
	    throw new Error('kind of impossible but missing clue!');
	}

	// only do indexing if all clues are primary, or if this
	// is a compound clue
	if (args.allPrimary || (nc.count > 1)) {
	    slIndex = this.getSrcListIndex(indexMap, nc, srcList);
	}
	else {
	    slIndex = 0;
	}
	if (this.logging ) {
	    this.log('build: index: ' + ncIndex +
		     ', source: ' + srcList[slIndex]);
	}

	srcNameList = srcList[slIndex].split(',');
	if (args.allPrimary) {
	    srcNameList.forEach(name => {
		primarySrcNameList.push(name);
	    });
	    return; // forEach.next;
	}

	if (srcMap[nc]) {
	    throw new Error('already in map, ' + nc);
	}

	// if nc is a primary clue
	if (nc.count == 1) {
	    // add map entry for list of primary name:sources
	    if (!srcMap[this.PRIMARY_KEY]) {
		srcMap[this.PRIMARY_KEY] = [];
	    }
	    srcMap[this.PRIMARY_KEY].push(_.toString(nc)); // consider nc.name here instead
	    primaryNcList.push(nc);
	    return; // forEach.next;
	}

	// nc is a compound clue
	map = srcMap[nc] = {};
	// if sources for this nc are all primary clues
	if (srcNameList.length === nc.count) {
	    // build component primary NC list
	    localPrimaryNcList = [];
	    srcNameList.forEach(name => {
		localPrimaryNcList.push(NameCount.makeNew(name, 1));
	    });
	    // add map entry for list of (eventual) primary name:sources
	    map[localPrimaryNcList] = [];
	    primaryNcList = _.concat(primaryNcList, localPrimaryNcList);
	    return; // forEach.next;
	}

	// sources for this nc include a compound clue
	clueCount += nc.count;
	allPrimary = false;

	// add map entry for list of source names
	// why don't we just add empty maps here? because we don't
	// know the nc.count for these names
	map[this.SOURCES_KEY] = srcNameList;
	compoundSrcNameList = _.concat(compoundSrcNameList, srcNameList);
	compoundNcList.push(nc);
    }, this);

    if (args.allPrimary && (primarySrcNameList.length != args.ncList.length)) {
	throw new Error('something went wrong, primary: ' + primarySrcNameList.length +
			', ncList: ' + args.ncList.length);
    }

    if (this.logging) {
	this.log('--buildSrcNameList' +
		    ', compoundSrcNameList: ' + compoundSrcNameList +
		    ', compoundNcList: ' + compoundNcList +
		    ', primarySrcNameList: ' + primarySrcNameList +
		    ', primaryNcList: ' + primaryNcList +
		    ', count: ' + clueCount);

	if (!_.isEmpty(srcMap)) {
	    this.dumpResultMap(srcMap);
	    //this.log(_(srcMap).toJSON());
	}
    }

    return {
	compoundSrcNameList:   compoundSrcNameList,
	compoundNcList:        compoundNcList,
	primaryNcList:         primaryNcList,
	primarySrcNameList:    primarySrcNameList,
	srcMap:                srcMap,
	count:                 clueCount,
	allPrimary:            allPrimary,
	indexMap:              indexMap
    };
}

//
//

Validator.prototype.getIndexMap = function(indexMap) {
    if (indexMap) {
	if (this.logging) {
	    this.log('using index map');
	}
	return indexMap;
    }
    if (this.logging) {
	this.log('new index map');
    }
    return {};
}

//
//

Validator.prototype.getSrcListIndex = function(indexMap, nc, srcList) {
    var slIndex;
    if (indexMap[nc]) {
	slIndex = indexMap[nc].index;
	// sanity check
	if (indexMap[nc].length != srcList.length) {
	    throw new Error('mismatched index lengths');
	}
	if (this.logging) {
	    this.log(nc.name + ': using preset index ' + slIndex +
		     ', length(' + indexMap[nc].length + ')' +
		     ', actual length(' + srcList.length + ')');
	}
    }
    else {
	slIndex = 0;
	indexMap[nc] = { index: 0, length: srcList.length };
	if (this.logging) {
	    this.log(nc.name + ': using first index ' + slIndex +
		     ', actual length(' + srcList.length + ')');
	}
    }
    return slIndex;
}

//
//

Validator.prototype.incrementIndexMap = function(indexMap) {
    var keyList;
    var keyIndex;
    var indexObj;

    if (!indexMap || _.isEmpty(indexMap)) {
	throw new Error('null or empty index map, ' + indexMap);
    }

    if (this.logging) {
	this.log('++indexMap: ' + this.indexMapToJSON(indexMap));
    }

    // TODO: this is a bit flaky. assumes the order of keys isn't changing.
    keyList = Object.keys(indexMap);

    // start at last index
    keyIndex = keyList.length - 1;
    indexObj = indexMap[keyList[keyIndex]];
    ++indexObj.index;

    // while index is maxed reset to zero, increment next-to-last index, etc.
    // using >= because it's possible both index and length are zero, for
    // primary clues, which are skipped.
    while (indexObj.index >= indexObj.length) {
	this.log('keyIndex ' + keyIndex + ': ' + indexObj.index +
		 ' >= ' + indexObj.length + ', resetting')
	indexObj.index = 0;
	--keyIndex;
	if (keyIndex < 0) {
	    return false;
	}
	indexObj = indexMap[keyList[keyIndex]];
	++indexObj.index;
	this.log('keyIndex ' + keyIndex + ': ' + indexObj.index +
		 ', length: ' + indexObj.length);

    }
    this.log('--indexMap: ' + this.indexMapToJSON(indexMap));
    return true;
}

//
//

Validator.prototype.indexMapToJSON = function(map) {
    var key;
    var s;
    s = '';
    for (key in map) {
	if (map.hasOwnProperty(key)) {
	    if (s.length > 0) {
		s += ',';
	    }
	    s += map[key].index;
	}
    }
    return '[' + s + ']';
}

//
//

Validator.prototype.copyAddNcList = function(ncList, name, count) {
    var newNcList;

    // for non-primary check for duplicate name:count entry
    // technically this is allowable for count > 1 if the there are
    // multiple entries of this clue name in the clueList[count].
    // (at least as many entries as there are copies of name in ncList)
    // TODO: make knownSourceMapArray store a count instead of boolean

    if (!ncList.every(nc => {
	if (nc.count > 1) {
	    if ((name == nc.name) && (count == nc.count)) {
		return false;
	    }
	}
	return true;
    })) {
	return null;
    }

    newNcList = Array.from(ncList);
    newNcList.push(NameCount.makeNew(name, count));

    return newNcList;
}

//
//

Validator.prototype.getDiffNcList = function(origNcList, nameCountList) {
    var ncList;
    var index;

    ncList = [];
    for (index = origNcList.length; index < nameCountList.length; ++index) {
	ncList.push(nameCountList[index]);
    }
    return ncList;
}

//
//

Validator.prototype.getNameSrcList = function(srcMap) {
    var srcMapKey;
    var ncList;
    ncList = [];
    for (srcMapKey in srcMap) {
	ncList.push(NameCount.makeNew(srcMap[srcMapKey], srcMapKey));
    }
    return ncList;
}

//
//

Validator.prototype.chop = function(list, removeValue) {
    var copy = [];
    list.forEach(value => {
	if (value == removeValue) {
	    removeValue = undefined;
	}
	else {
	    copy.push(value);
	}
    });
    return copy;
}

// args:
//  ncList:        this.getNameSrcList(primaryClueData),
//  stagingNcList: nameCountList,
//  map:           args.resultMap,
//  excludeSrcList:args.excludeSrcList,
//  stagingName:   name
//
Validator.prototype.addFinalResult = function(args) {
    if (this.addResultEntry({
	map:           args.map,
	key:           this.PRIMARY_KEY,
	name:          this.FINAL_KEY,
	ncList:        args.ncList,
	excludeSrcList:args.excludeSrcList
    })) {
	//console.log('Added: ' + args.ncList.toString() + ', length=' + args.ncList.length);
	// seems like we might want to do this even if addResultEntry fails.
	// just because all primary name:src are duplicate doesn't mean
	// it didn't derive from higher level clues that are unique
	if (args.stagingNcList) {
	    this.moveStagingToCompound({
		map:    args.map,
		ncList: args.stagingNcList
	    });
	}
	return true;
    }
    return false;
}

// args:
//  map:      args.resultMap,
//  key:      PRIMARY_KEY
//  name:     clue name
//  ncList:   ncList
//  excludeSrcList: list of excluded primary sources
//
Validator.prototype.addResultEntry = function(args) {
    var map;
    var srcList;
    var stringList;

    if (args.excludeSrcList) {
	// TODO: this looks wrong
	if (this.listContainsAny(args.excludeSrcList,
				 NameCount.makeCountList(args.ncList)))
	{
	    return false;
	}
    }

    if (!args.map[args.key]) {
	map = args.map[args.key] = {};
    }
    else {
	map = args.map[args.key];
    }
    stringList = args.ncList.toString();
    if (!map[args.name]) {
	map[args.name] = [ stringList ];
    }
    else if (map[args.name].indexOf(stringList) === -1) {
	map[args.name].push(stringList);
    }

    return true;
}

// Array.prototype
Validator.prototype.listContainsAny = function(list1, list2) {
    return list1.some(elem => {
	return list2.indexOf(elem) != -1;
    });
}

// args:
//  map:        args.resultMap
//  ncList:     ncList
//

Validator.prototype.moveStagingToCompound = function(args) {
    var stagingMap;
    var compoundMap;
    var name;
    var srcList;

    if (!args.map[this.COMPOUND_KEY]) {
	compoundMap = args.map[this.COMPOUND_KEY] = {};
    }
    else {
	compoundMap = args.map[this.COMPOUND_KEY];
    }

    stagingMap = args.map[this.STAGING_KEY];
    for (name in stagingMap) {
	stagingMap[name].forEach(src => {
	    if (!compoundMap[name]) {
		compoundMap[name] = [ src ];
	    }
	    else {
		if (compoundMap[name].indexOf(src) == -1) {
		    compoundMap[name].push(src);
		}
	    }
	});
    }
    args.map[this.STAGING_KEY] = {};
}


//

Validator.prototype.getFinalResultList = function(result) {
    var primaryMap;
    var resultList;

    resultList = [];
    primaryMap = result[this.PRIMARY_KEY];
    if (primaryMap) {
	if (primaryMap[this.FINAL_KEY]) {
	    resultList = primaryMap[this.FINAL_KEY];
	}
    }
    return resultList;
}

Validator.prototype.dumpIndexMap = function(indexMap) {
    var s = '';
    var key;
    var entry;

    for (key in indexMap) {
	entry = indexMap[key];
	if (s.length > 0) {
	    s += '; ';
	}
	s += 'index ' + entry.index + ', length ' + entry.length;
    }
    this.log(s);
}

// args:
//  header:
//  result:
//  primary:  t/f
//  compound: t/f
//
Validator.prototype.dumpResult = function(args) {
    var name;
    var map;
    var header = 'Validate results';
    var keyNameList;

    if (args.header) {
	header += ' for ' + args.header;
    }
    console.log(header);

    keyNameList = [];
    if (args.compound) {
	keyNameList.push({
	    key:  this.COMPOUND_KEY,
	    name: 'compound'
	});
    }
    if (args.primary) {
	keyNameList.push({
	    key:  this.PRIMARY_KEY,
	    name: 'primary'
	});
    }

    keyNameList.forEach(keyName => {
	var header = keyName.name + ':';
	if (!args.result[keyName.key]) {
	    console.log(header + ' empty');
	    return;
	}
	map = args.result[keyName.key];
	if (Object.keys(map).length == 0) {
	    console.log(header + ' empty');
	    return;
	}
	console.log(header);
	for (name in map) {
	    map[name].forEach(ncList => {
		console.log(name + ':' + ncList);
	    });
	}
    });
}

// args:
//  header:
//  result:
//  primary:  t/f
//  compound: t/f
//
Validator.prototype.dumpResultMap = function(seq, level) {
//    console.log(_(seq).toJSON());
//    return;
    if (!level) level = 0;
    if (typeof seq === 'object') {
	console.log(spaces(2 * level) + (_.isArray(seq) ? '[' : '{'));
	++level;

	if (_.isArray(seq)) {
	    seq.forEach(elem => {
		if (typeof elem === 'object') {
		    this.dumpResultMap(elem, level + 1);
		}
		else {
		    console.log(spaces(2*level) + elem);
		}
	    }, this);
	}
	else {
	    _.forOwn(seq, function(value, key) {
		if (typeof value === 'object') {
		    console.log(spaces(2*level) + key + ':');
		    this.dumpResultMap(value, level + 1);
		}
		else {
		    console.log(spaces(2*level) + key + ': ' + value);
		}
	    }.bind(this));
	}

	--level;
	console.log(spaces(2 * level) + (_.isArray(seq) ? ']' : '}'));
    }
}

function spaces(length){
    var count;
    var result = '';
    for (count = 0; count < length; result += ' ', count++);
    return result;
}
