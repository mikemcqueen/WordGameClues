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
var ResultMap     = require('./resultmap');
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

    // TODO: these are duplicated in ResultMap
    this.PRIMARY_KEY      = '__primary';
    this.SOURCES_KEY      = '__sources';

    this.STAGING_KEY      = '__staging';
    this.COMPOUND_KEY     = '__compound';
    this.FINAL_KEY        = '__final';
}

//

Validator.prototype.log = function(text) {
    if (this.logging) {
	console.log(this.indent() + text);
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
//  sum:            # of primary clues represented by names in nameList
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
    var ncListArray = [];
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
	resultMap = {}; // ResultMap.makeNew();
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
	    recursive:     args.recursive,
	    resultMap:     resultMap
	});
	if (rvsResult.success) {
	    if (this.logging) {
		this.log('recursiveValidateSources: VALIDATE SUCCESS!');
	    }
	    vsResultMap = rvsResult.resultMap;
	    ncListArray = _.concat(ncListArray, rvsResult.ncListArray);

	    found = true;

	    // We may have been called recursively. Don't set 'found' flag
	    // in this case, because we don't know if we meet require/exlude
	    // constraints
	    if (args.recursive) {
		// TODO: in this case, there are potentially additional
		// combinations for the names in nameList. We're skipping
		// them which is bad.

		///return true; // some.exit, return to checkUniqueSources()
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
    if (!args.recursive && vsResultMap) {
	vsResultMap.ensureUniquePrimaryLists();
    }
    return {
	resultMap:   resultMap,
	vsResultMap: vsResultMap,
	ncListArray: ncListArray,
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
// TODO: ForNameList

Validator.prototype.recursiveValidateSources = function(args) {
    var nameIndex;
    var clueName;
    var count;
    var someResult;
    var rvsResult;

    if (!args.vsCount || !args.clueNameList || !args.clueCountList) {
	throw new Error ('missing args' +
			 ', clueNameList: ' + args.clueNameList +
			 ', clueCountList: ' + args.clueCountList +
			 ', vsCount: ' + args.vsCount);
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
	    recursive:      args.recursive,
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
	success:     true,
	resultMap:   rvsResult.resultMap,
	ncListArray: rvsResult.ncListArray
    } : {
	success: false
    };
}

// args:
//   name     : clueName,
//   count    : count,
//   nameList : clueNameList,
//   countList: clueCountList,
//   ncList
//   excludeSrcList: list of excluded primary sources
//   validateAll:
//   wantResults:
//   resultMap:
//   vsCount:       pass-through
//
// TODO: ForName
Validator.prototype.rvsWorker = function(args) {
    var newNameCountList;
    var uniqResult;
    var rvsResult;
    var nc;

    if (this.logging) {
	this.log('++rvsWorker' +
		 ', name: ' + args.name +
		 ', count: ' + args.count +
		 ', validateAll: ' + args.validateAll +
		 ', ncList: ' + args.ncList +
		 ', nameList: ' + args.nameList);
    }

    if (!args.vsCount || !args.name || !args.count || 
	!args.ncList || !args.nameList) {
	throw new Error ('missing args');
    }

    newNameCountList = this.copyAddNcList(args.ncList, args.name, args.count);
    if (!newNameCountList) {
	// duplicate name:count entry. technically this is allowable for
	// count > 1 if the there are multiple entries of this clue name
	// in the clueList[count]. (at least as many entries as there are
	// copies of name in ncList)
	if (this.logging) {
	    this.log('--rvsWorker, duplicate name:count, ' + args.name + ':' + args.count);
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
		this.log('checkUniqueSources --- success!');
	    }

	    nc = NameCount.makeNew(args.name, args.count);
	    args.ncList.push(nc);

	    if (this.logging) {
		this.log('add1, ' + args.name + ':' + args.count +
			 ', newNc(' + newNameCountList.length + ')' +
			 ', ' + newNameCountList);
	    }

	    return {
		success:     true,
		resultMap:   uniqResult.resultMap,
		ncListArray: uniqResult.ncListArray
	    };
	}

	if (this.logging) {
	    this.log('checkUniqueSources --- failure');
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
	recursive:     args.recursive,
	resultMap:     args.resultMap
    });
    if (!rvsResult.success) {
	if (this.logging) {
	    this.log('--rvsWorker, recursiveValidateSources failed');
	}
	return { success: false }; // fail
    }

    // does this achieve anything? modifies args.ncList.
    // TODO: probably need to remove why that matters.
    args.ncList.length = 0;
    newNameCountList.forEach(nc => args.ncList.push(nc));

    if (this.logging) {
	this.log('--rvsWorker, add ' + args.name + ':' + args.count +
		 ', newNc(' + newNameCountList.length + ')' +
		 ', ' + newNameCountList);
    }

    return {
	success:     true,
	resultMap:   rvsResult.resultMap,
	ncListArray: rvsResult.ncListArray
    };
}

// args:
//   name:           clueName,
//   count:          # of primary clues represented by cluename
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
    var vsResult;
    var ncListCsv;
    var pendingMap;
    var resultMap;
    var anyCandidate;
    var candidateNcListArray;
    var ncListArray;
    var uniqResult;
    var anyFlag = false;


    if (!args.vsCount) {
	throw new Error ('missing args');
    }

    // assert(nameCountList) && Array.isArray(nameCountList)

    if (this.logging) {
	this.log('++checkUniqueSouces' +
		 ', name: ' + args.name +
		 ', count: ' + args.count +
		 ', nameList: ' + args.nameList +
		 ', validateAll: ' + args.validateAll +
		 ', recursive: ' + args.recursive +
		 ', ncList: ' + args.nameCountList);
    }

    // TODO: or just pass in as arg, like a sane person
    pendingMap = ResultMap.makeNew();
    resultMap = ResultMap.makeNew();

    // first, check for duplicate primary clues, add all to primaryMap
    findResult = this.findDuplicatePrimaryClue({ ncList: nameCountList });
    if (!this.evalFindDuplicateResult(findResult, '1st')) {
	if (this.logging) {
	    this.log('FAIL - duplicate primary' +
		     ', nameCountList; ' + nameCountList);
	}
	return this.uniqueResult(false); // failure
    }
    else if (findResult.allPrimary) {
	nameSrcList = this.getNameSrcList(findResult.srcMap);
	if (args.wantResults) {
	    pendingMap.addPrimaryLists(nameCountList, nameSrcList);
	}
	uniqResult = this.allUniquePrimary({
	    origNcList:     nameCountList, // [NameCount.makeNew(args.name,args.count)],//nameCountList,
	    ncList:         nameCountList,
	    nameSrcList:    nameSrcList,
	    map:            args.resultMap,
	    excludeSrcList: args.excludeSrcList,
	    vsCount:        args.vsCount,
	    wantResults:    args.wantResults,
	    validateAll:    args.validateAll,
	    pendingMap:     pendingMap
	});
	if (uniqResult.success) {
	    if (args.wantResults) {
		resultMap.merge(uniqResult.pendingMap);
	    }
	    return this.uniqueResult(true, resultMap, [nameCountList]);
	}
	return this.uniqueResult(false);
    }

    ncListArray = [];
    buildArgs = {
	excludeSrcList: args.excludeSrcList,
	wantResults:    args.wantResults,
	ncList:         nameCountList,
	resultMap:      args.resultMap
    };

    for(;;) {
	if (buildResult) {
	    buildArgs.indexMap = buildResult.indexMap;
	}
	nameCountList = origNcList;

	for (;;) {
	    // break down all compound clues into source components
	    buildArgs.ncList = nameCountList;
	    buildResult = this.buildSrcNameList(buildArgs);
	    pendingMap = buildResult.resultMap;

	    // skip recursive call to validateSources if we have all primary clues
	    if (buildResult.allPrimary) {
		if (this.logging) {
		    this.log('cUS: adding all_primary result: ' +
			     nameCountList + ' = ' + buildResult.primaryNcList);
		}
		nameCountList = buildResult.primaryNcList;
		candidateNcListArray = [ buildResult.primaryNcList ];
	    }
	    else {
		// call validateSources recursively with compound clues
		srcNcList = [];
		vsResult = this.validateSources({
		    sum:            buildResult.count,
		    nameList:       buildResult.compoundSrcNameList,
		    count:          buildResult.compoundSrcNameList.length,
		    excludeSrcList: args.excludeSrcList,
		    validateAll:    args.validateAll,
		    wantResults:    args.wantResults,
		    vsCount:        args.vsCount,
		    resultMap:      args.resultMap,
		    recursive:      true
		}, srcNcList);
		if (!vsResult.success) {
		    break; // fail, try other combos
		}
		// sanity check
		if (srcNcList.length < 2) {
		    throw new Error('list should have at least two entries2');
		}
		if (_.isEmpty(vsResult.ncListArray)) {
		    throw new Error('empty nc list array');
		}

		if (this.logging) {
		    this.log('from validateSources(' + buildResult.count + ')' +
			     this.indentNewline() + '  compoundSrcNameList: ' + buildResult.compoundSrcNameList +
			     this.indentNewline() + '  srcNcList: ' + srcNcList);
		    this.log('  ncListArray.size: ' + _.size(vsResult.ncListArray));
		    vsResult.ncListArray.forEach(ncList => {
			this.log('  ncList: ' + ncList);
		    });
		    vsResult.vsResultMap.dump();
		}
		if (args.wantResults) {
		    // looks backwards, but correct.
		    pendingMap.merge(vsResult.vsResultMap, buildResult.compoundNcList);
		}
		// we only sent compound clues to validateSources, so add the
		// primary clues that build filtered out to make a complete list
		candidateNcListArray = [];
		vsResult.ncListArray.forEach(ncList => {
		    candidateNcListArray.push(_.concat(ncList, buildResult.primaryNcList));
		});
	    }
	    
	    anyCandidate = false;
	    candidateNcListArray.some(ncList => {
		var findResult;
		var uniqResult;

		findResult = this.findDuplicatePrimaryClue({ ncList: ncList });
		if (!this.evalFindDuplicateResult(findResult, '2nd')) {
		    return false; // some.continue, then break
		}
		if (!findResult.allPrimary) {
		    throw new Error ('shit happens, ' + ncList);
		}
		nameSrcList = this.getNameSrcList(findResult.srcMap),
		uniqResult = this.allUniquePrimary({
		    origNcList:     buildArgs.ncList, //origNcList,
		    ncList:         ncList,
		    ncNameListPairs:buildResult.compoundNcNameListPairs,
		    nameSrcList:    nameSrcList,
		    map:            args.resultMap,
		    excludeSrcList: args.excludeSrcList,
		    vsCount:        args.vsCount,
		    wantResults:    args.wantResults,
		    validateAll:    args.validateAll,
		    pendingMap:     pendingMap
		});
		if (uniqResult.success) {
		    anyCandidate = true;
		    ncListArray.push(ncList);
		    if (args.recursive) {
			return false; // some.continue if we're in a recursive call
		    }
		    return !args.validateAll; // some.continue if we're validating all, else exit
		}
	    });
	    if (!anyCandidate) {
		break; // none of those results were good, try other combos
	    }
	    anyFlag = true;
	    // found a good result
	    if (args.wantResults) {
		resultMap.merge(pendingMap);
	    }
	    if (args.validateAll || args.recursive) {
		break; // success - but keep searching for other combos
	    }
	    return this.uniqueResult(anyFlag, resultMap, ncListArray); // success - exit function
	}
	// sanity check
	if (!buildResult) {
	    throw new Error('!buildResult');
	}
	if (buildResult && !this.incrementIndexMap(buildResult.indexMap)) {
	    if (this.logging) {
		this.log('done, success: ' + anyFlag);
	    }
	    return this.uniqueResult(anyFlag, resultMap, ncListArray);
	}

	if (this.logging) {
	    this.log('++outer looping');
	}
    }
}

//
//

Validator.prototype.uniqueResult = function(success, map, ncListArray) {
    return {
	success:     success,
	resultMap:   map,
	ncListArray: ncListArray
    };
}


//  origNcList:
//  ncList:
//  nameSrcList:
//  ncNameListPairs:
//  ncName
//  map:
//  excludeSrcList:
//  vsCount:
//  wantResults:
//  pendingMap:
//  validateAll:
//
Validator.prototype.allUniquePrimary = function(args) {
    var cycleList;
    var nameSrcCsvArray;

    // no duplicates, and all clues are primary, success!
    if (this.logging) {
	this.log('++allUniquePrimary: SUCCESS! vsCount: ' + args.vsCount +
		 ', ncList: ' + args.ncList +
		 this.indentNewline() + '  origNcList: ' + args.origNcList +
		 this.indentNewline() + '  nameSrcList: ' + args.nameSrcList);
    }

    if (this.addFinalResult({
	map:           args.map,
	nameSrcList:   args.nameSrcList,
	excludeSrcList:args.excludeSrcList,
	vsCount:       args.vsCount,
    })) {
	if (args.wantResults) {
	    if (this.logging) {
		this.log('aUP: adding primary result: ' +
			 this.indentNewline() + '  ' +
			 args.ncList + ' = ' + args.nameSrcList);
	    }
	    args.pendingMap.addResult({
		origNcList:     args.origNcList,
		primaryNcList:  args.ncList,
		nameSrcList:    args.nameSrcList,
		ncNameListPairs:args.ncNameListPairs
	    });
	}
    }
    else {
	if (this.logging) {
	    this.log('isValidResult failed');
	}
	return { success: false };
    }
    if (args.validateAll) {
	if (cycleList = this.cyclePrimaryClueSources({
	    resultMap:     args.map,
	    ncList:        args.ncList,
	    vsCount:       args.vsCount,
	    excludeSrcList:args.excludeSrcList
	})) {
	    if (args.wantResults) {
		cycleList.forEach(nameSrcList => {
		    args.pendingMap.addResult({
			origNcList:     args.origNcList,
			primaryNcList:  args.ncList,
			nameSrcList:    nameSrcList,
			ncNameListPairs:args.ncNameListPairs
		    });
		});
	    }
	}
    }
    return {
	success:    true,
	pendingMap: args.pendingMap
    };
}

// Simplified version of checkUniqueSources, for all-primary clues.
//
// args:
//  ncList:
//  exclueSrcList:
//  resultMap:      args.resultMap
//  vsCount:       args.vsCount,
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
    localNcList = _.cloneDeep(args.ncList)
//    args.ncList.forEach(nc => {
//	localNcList.push(NameCount.makeCopy(nc));
//    });
    resultList = [];
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

	// TODO: return srcMap in findResult
	srcMap = {};
	findResult = this.findDuplicatePrimarySource({
	    ncList: localNcList,
	    srcMap: srcMap
	});
	if (findResult.duplicateSrc) {
	    continue;
	}
	if (_.size(localNcList) != _.size(srcMap)) {
	    throw new Error('localNcCount.length != keys.length!' +
			    ', findResult.duplicateSrc:' + findResult.duplicateSrc);
	}

	nameSrcList = this.getNameSrcList(srcMap);

	if (this.logging) {
	    this.log('[srcMap] primary keys: ' + nameSrcList);
	}

	// all the source clues we just validated are primary clues
	if (this.addFinalResult({
	    map:           args.resultMap,
	    nameSrcList:   nameSrcList,
	    excludeSrcList:args.excludeSrcList,
	    vsCount:       args.vsCount
	})) {
	    if (this.logging) {
		this.log('cycle: adding result: ' +
			 this.indentNewline() + '  ' +
			 args.ncList + ' = ' + nameSrcList);
	    }
	    resultList.push(nameSrcList);
	}
    } while (this.incrementIndexMap(buildResult.indexMap));

    this.log('--cyclePrimaryClueSources, size: ' + _.size(resultList));
    resultList.forEach(result => {
	this.log('  list: ' + result);
    });

    return resultList;
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

    // log before possible exception, to provide more info
    if (this.logging) {
	this.log('--findDuplicatePrimaryClue, duplicateName: ' + duplicateName +
		 ', duplicateSrc: ' + duplicateSrc +
		 ', allPrimary: ' + allPrimary +
		 ', srcMap.size: ' + _.size(findResult.srcMap));
    }

    if (allPrimary && !duplicateSrc &&
	(_.size(findResult.srcMap) != _.size(args.ncList)))
    {
	console.log('ncList: ' + args.ncList +
		    '\nsrcMap.keys: ' + _.keys(findResult.srcMap));
	throw new Error('srcMap.size != ncList.size');
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
    var compoundNcNameListPairs;
    var primaryNcList;
    var primarySrcNameList;
    var resultMap;
    var primaryPathList; // TODO: i had half an idea here

    if (this.logging) {
	this.log('++buildSrcNameList, ncList(' + args.ncList.length + ')' +
		 this.indentNewline() + '  ncList: ' + args.ncList +
		 this.indentNewline() + '  allPrimary: ' + args.allPrimary);
    }

    indexMap = this.getIndexMap(args.indexMap);

    allPrimary = true;
    clueCount = 0;
    compoundNcList = [];
    compoundSrcNameList = [];
    compoundNcNameListPairs = [];
    primaryNcList = [];
    primarySrcNameList = [];
    primaryPathList = [];
    resultMap = ResultMap.makeNew();

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

	// short circuit if we're called with allPrimary:true.
	if (nc.count === 1) {
	    primaryNcList.push(nc);
	    if (args.allPrimary) {
		srcNameList.forEach(name => {
		    primarySrcNameList.push(name);
		});
		return; // forEach.next
	    }
	}

	if (resultMap.map()[nc]) {
	    throw new Error('already in map, ' + nc);
	}

	// if nc is a primary clue
	if (nc.count == 1) {
	    // add map entry for list of primary name:sources
	    if (!resultMap.map()[this.PRIMARY_KEY]) {
		resultMap.map()[this.PRIMARY_KEY] = [];
	    }
	    resultMap.map()[this.PRIMARY_KEY].push(_.toString(nc)); // consider nc.name here instead
	    return; // forEach.next;
	}

	compoundNcNameListPairs.push([nc, _.clone(srcNameList)]);

	// nc is a compound clue
	map = resultMap.map()[nc] = {};
	// if sources for this nc are all primary clues
	if (_.size(srcNameList) === nc.count) {
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
		 ', count: ' + clueCount +
		 this.indentNewline() + '  compoundNcNameListPairs: ' + compoundNcNameListPairs +
		 this.indentNewline() + '  primarySrcNameList: ' + primarySrcNameList +
		 this.indentNewline() + '  primaryNcList: ' + primaryNcList);

	if (!_.isEmpty(resultMap.map())) {
	    this.log('resultMap:');
	    resultMap.dump();
	    //this.log(_(srcMap).toJSON());
	}
	else {
	    this.log('resultMap: empty');
	}
    }

    return {
	compoundNcNameListPairs:compoundNcNameListPairs,
	compoundSrcNameList:    compoundSrcNameList,
	compoundNcList:         compoundNcList,
	primaryNcList:          primaryNcList,
	primarySrcNameList:     primarySrcNameList,
	resultMap:              resultMap,
	count:                  clueCount,
	allPrimary:             allPrimary,
	indexMap:               indexMap
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
//  nameSrcList:
//  map:
//  excludeSrcList:
//  vsCount:
//
Validator.prototype.addFinalResult = function(args) {
    if (!args.vsCount || !args.map ||
	!args.nameSrcList || _.isEmpty(args.nameSrcList)) {
	throw new Error ('missing args' +
			 ', nameSrcList: ' + args.nameSrcList +
			 ', vsCount: ' + args.vsCount);
    }
    if (this.addResultEntry({
	map:           args.map,
	nameSrcList:   args.nameSrcList,
	excludeSrcList:args.excludeSrcList,
	vsCount:       args.vsCount
    })) {
	return true;
    }
    return false;
}

// args:
//  map:
//  nameSrcList:
//  excludeSrcList: list of excluded primary sources
//  vsCount:
//
Validator.prototype.addResultEntry = function(args) {
    var map;
    var srcList;
    var nameSrcCsv;

    var key = this.PRIMARY_KEY;
    var name = this.FINAL_KEY;

    if (args.excludeSrcList) {
	// TODO: this looks wrong
	if (this.listContainsAny(args.excludeSrcList,
				 NameCount.makeCountList(args.nameSrcList)))
	{
	    return false;
	}
    }

    // using vsCount is bogus
    if (args.vsCount === _.size(args.nameSrcList)) {
	if (!args.map[key]) {
	    map = args.map[key] = {};
	}
	else {
	    map = args.map[key];
	}
	nameSrcCsv = args.nameSrcList.toString();
	if (!map[name]) {
	    map[name] = [ nameSrcCsv ];
	}
	else if (map[name].indexOf(nameSrcCsv) === -1) {
	    map[name].push(nameSrcCsv);
	}
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
	console.log(this.indent() + spaces(2 * level) + (_.isArray(seq) ? '[' : '{'));
	++level;

	if (_.isArray(seq)) {
	    seq.forEach(elem => {
		if (typeof elem === 'object') {
		    this.dumpResultMap(elem, level + 1);
		}
		else {
		    console.log(this.indent() + spaces(2*level) + elem);
		}
	    }, this);
	}
	else {
	    _.forOwn(seq, function(value, key) {
		if (typeof value === 'object') {
		    console.log(this.indent() + spaces(2 * level) + key + ':');
		    this.dumpResultMap(value, level + 1);
		}
		else {
		    console.log(this.indent() + spaces(2 * level) + key + ': ' + value);
		}
	    }.bind(this));
	}

	--level;
	console.log(this.indent() + spaces(2 * level) + (_.isArray(seq) ? ']' : '}'));
    }
}

Validator.prototype.indent = function() {
    return spaces(this.logLevel);
}

Validator.prototype.indentNewline = function() {
    return '\n' + this.indent()
}

function spaces(length){
    var count;
    var result = '';
    for (count = 0; count < length; result += ' ', count++);
    return result;
}
