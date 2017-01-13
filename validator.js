//
// VALIDATOR.JS
//

'use strict';

// export a singleton

module.exports = exports = new Validator();

var ClueManager    = require('./clue_manager');
var ClueList       = require('./clue_list');
var NameCount      = require('./name_count');
var Peco           = require('./peco');

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
    var result;

    if (args.validateAll) {
	args.wantResults = true;
    }

    if (this.logging) {
	this.log('++validateSources' +
		 ', nameList(' + args.nameList.length + 
		 ', count(' + args.count + ')' +
		 ', require: ' + args.require +
		 ', exclude: ' + args.exclude +
		 ', excludeSrc: ' + args.excludeSrc +
		 ', validateAll: ' + args.validateAll +
		 ', wantResults: ' + args.wantResults);
	this.log('  names: ' + args.nameList);
    }

    if (!args.resultMap) {
	resultMap = {};
    }
    else {
	resultMap = args.resultMap;
    }

    found = false;
    vsCount = args.vsCount ? args.vsCount : args.sum;
    (new Peco({
	sum:     args.sum,
	count:   args.count,
	max:     args.max,
	require: args.require,
	exclude: args.exclude,
	quiet:   args.quiet
    })).getCombinations().some(clueCountList => {
	if (this.recursiveValidateSources({
	    clueNameList:  args.nameList,
	    clueCountList: clueCountList,
	    nameCountList: nameCountList,
	    excludeSrcList:args.excludeSrcList,
	    validateAll:   args.validateAll,
	    wantResults:   args.wantResults,
	    vsCount:       vsCount,
	    resultMap:     resultMap
	})) {
	    found = true;
	    if (!args.validateAll) {
		return true; // found a match; some.exit
	    }
	}
	return false; // some.continue
    }, this);
    
    if (this.logging) {
	this.log('--validateSources');
	if (nameCountList) {
	    this.log('nameCountList(' + nameCountList.length + ')');
	    nameCountList.forEach(nc => {
		this.log(' nc.name = ' + nc.name + ', nc.count = ' + nc.count);
	    });
	}
    }

    if (args.wantResults) {
	result = found ? resultMap : null;
    }
    else {
	result = found;
    }
    return result;
}

//  clueNameList:  
//  clueCountList: 
//  nameCountList: 
//  excludeSrcList: list of excluded primary sources
//  validateAll:       
//  wantResults:
//  vsCount:       pass-through
//

Validator.prototype.recursiveValidateSources = function(args) {
    var nameIndex;
    var clueName;
    var count;
    var args;
    var result;

    this.logLevel++;

    if (!args.nameCountList) {
	args.nameCountList = []; // new Array();
    }
    if (this.logging) {
	this.log('name: looking for [' + args.clueNameList + '] in [' + args.clueCountList + ']');
    }

    if (args.clueNameList.length != args.clueCountList.length) {
	throw new Error('Mismatched list lengths');
    }

    nameIndex = 0;
    clueName = args.clueNameList[nameIndex];
    
    // optimzation: could have a map of count:boolean entries here
    // on a per-name basis (new map for each outer loop; once a
    // count is checked for a name, no need to check it again
    
    // using ordinary for loop so we can return directly out of function.
    result = args.clueCountList.some(count => {

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
	    this.log('+++input(' + args.nameCountList.length + ')');
	    nameCountList.forEach(nc => {
		this.log(' nc.name = ' + nc.name + ', nc.count = ' + nc.count)
	    });
	    this.log('---input');
	}
	
	if (!this.rvsWorker({
	    name:        clueName,
	    count:       count,
	    nameList:    args.clueNameList,
	    countList:   args.clueCountList,
	    ncList:         args.nameCountList,
	    excludeSrcList: args.excludeSrcList,
	    validateAll: args.validateAll,
	    wantResults: args.wantResults,
	    vsCount:     args.vsCount,      // aka "original count"
	    resultMap:   args.resultMap

	})) {
	    return false; // some.continue;
	}
	
	if (this.logging) {
	    this.log('++rvsWorker output(' + args.nameCountList.length + ') for: ' + clueName);
	    args.nameCountList.forEach(nc => {
		this.log('nc.name = ' + nc.name + ', nc.count = ' + nc.count)
	    });
	    this.log('--rvsWorker output(' + args.nameCountList.length + ')');
	}
	
	if (!args.validateAll && (args.nameCountList.length < 2)) {
	    // TODO: add "allowSingleEntry" ? or was this just a sanity check?
	    // can i check vs. clueNameList.length?
//	    throw new Error('list should have at least two entries1');
	}

	/*if (args.validateAll) {
	    this.addResultEntry({
		map:    args.resultMap,
		key:    this.STAGING_KEY,
		name:   clueName + '.rvs',
		ncList: Array.from(args.nameCountList)
	    });
	}*/

	return true; // success: some.exit
    });

    this.logLevel--;
    return result;
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
//   vsCount:       pass-through
//
Validator.prototype.rvsWorker = function(args) {
    var newNameCountList;
    var result;

    newNameCountList = this.copyAddNcList(args.ncList, args.name, args.count);
    if (!newNameCountList) {
	// duplicate name:count entry. technically this is allowable for
	// count > 1 if the there are multiple entries of this clue name
	// in the clueList[count]. (at least as many entries as there are
	// copies of name in ncList)
	if (this.logging) {
	    this.log('duplicate name:count, ' + args.name + ':' + args.count);
	}

	return false; // fail
    }

    if (this.logging) {
	this.log('added NC name: ' + args.name +
		 ', count = ' + args.count +
		 ', length = ' + newNameCountList.length);
    }

    // If only one name & count remain, we're done.
    // (name & count lists are equal length, just test one)
    if (args.nameList.length == 1) {
	if (this.checkUniqueSources(newNameCountList, args)) {

	    if (this.logging) {
		this.log('checkUniqueSources --- success!\n' +
			 '---------------------------------------------------');
	    }

	    args.ncList.push(NameCount.makeNew(args.name, args.count));

	    if (this.logging) {
		this.log('add1(' + args.ncList.length + '), ' + args.name + ':' + args.count);
	    }

	    return true; // success!
	}

	if (this.logging) {
	    this.log('checkUniqueSources --- failure\n' +
		     '---------------------------------------------------');
	}

	return false; // fail
    }
    // Otherwise, remove current name & count, and validate remaining
    if (!this.recursiveValidateSources({
	clueNameList:  this.chop(args.nameList, args.name),
	clueCountList: this.chop(args.countList, args.count),
	nameCountList: newNameCountList,
	excludeSrcList:args.excludeSrcList,
	validateAll:   args.validateAll,
	wantResults:   args.wantResults,
	vsCount:       args.vsCount,      // aka "original count"
	resultMap:     args.resultMap
    })) {
	return false; // fail
    }

    // does this achieve anything?
    args.ncList.length = 0;
    newNameCountList.forEach(nc => args.ncList.push(nc));

    /*if (args.validateAll) {
	this.addResultEntry({
	    map:    args.resultMap,
	    key:    this.STAGING_KEY,
	    name:   args.name + ".first",
	    ncList: Array.from(newNameCountList)
	});
    }*/

    if (this.logging) {
	    this.log('add2(' + args.ncList.length + ')' +
		    ', newNc(' + newNameCountList.length + ')' +
		    ', ' + args.name + ':' + args.count);
    }

    return true; // success!
}

//
//

const KEY_COMPLETE            = '__key_complete';

// args:
//   name     : clueName,
//   count    : count,
//   nameList : clueNameList,
//   countList: clueCountList,
//   ncList   : nameCountList,
//   excludeSrcList: list of excluded primary sources
//   validateAll:       
//   wantResults:
//   vsCount:       pass-through
///
Validator.prototype.checkUniqueSources = function(nameCountList, args) {
    var origNameCountList = nameCountList;
    var primaryClueData;
    var srcNcList;
    var buildArgs;
    var findResult;
    var buildResult;
    var anyFlag = false;

    // assert(nameCountList) && Array.isArray(nameCountList)

    if (this.logging) {
	this.log('CheckUniqueSouces, ncList.length = ' + nameCountList.length);
    }

    primaryClueData = {};
    buildArgs = {
	excludeSrcList: args.excludeSrcList,
	validateAll:    args.validateAll,
	wantResults:    args.wantResults,
	resultMap:      args.resultMap
	//,primaryClueData:primaryClueData
    };
    for(;;) {
	nameCountList = origNameCountList;
	primaryClueData.nameMap = {};
	primaryClueData.srcMap = {};
	
	// should have findDupe check here, and after build()
	
	for (;;) {

	    // first, check for duplicate primary clues, add all to primaryMap
	    findResult = this.findDuplicatePrimaryClue({
		ncList:              nameCountList, 
		clueData:            primaryClueData,
		onlyAddIfAllPrimary: false
	    });
	    if (!this.evalFindDuplicateResult(findResult, '1st')) {
		if (this.logging) {
		    this.log('FAIL - duplicate primary');
		    nameCountList.forEach(nc => {
			this.log('  nc.name = ' + nc.name + 
				 ' nc.count = ' + nc.count);
		    });
		    this.log('keys:');
		    Object.keys(primaryClueData.nameMap).forEach(key => {
			this.log('  ' + key);
		    });
		}
		break; // failure
	    }
	    else if (findResult.allPrimary) {
		// no duplicates, and all clues are primary, success!
		if (args.wantResults) {
		    if ((args.vsCount == Object.keys(primaryClueData.srcMap).length)) {
			if (this.addResultEntry({
			    map:           args.resultMap,
			    key:           this.PRIMARY_KEY,
			    name:          this.FINAL_KEY,
			    ncList:        this.getClueDataNcList(primaryClueData),
			    excludeSrcList:args.excludeSrcList
			})) {
			    this.moveStagingToCompound({
				map:    args.resultMap,
				ncList: nameCountList
			    });
			}
		    }
		    else {
			this.addResultEntry({
			    map:    args.resultMap,
			    key:    this.STAGING_KEY,
			    name:   args.name,
			    ncList: this.getClueDataNcList(primaryClueData),
			    excludeSrcList:args.excludeSrcList
			});
		    }
		}
		
		if (this.logging) {
		    this.log('SUCCESS! all primary');
		    nameCountList.forEach(nc => {
			this.log('nc.name = ' + nc.name + ', nc.count = ' + nc.count)
		    });
		}

		return true; // success - exit function
	    }
	    
	    // break down all compound clues into source components
	    buildArgs.ncList = nameCountList;
	    buildResult = this.buildCompoundClueSourceNameList(buildArgs);

	    // skip recursive call to validateSources if we have all primary clues
	    if (buildResult.allPrimary) {
		// build an NC list of primary clues
		nameCountList = [];
		buildResult.clueNameList.forEach(name => {
		    nameCountList.push(NameCount.makeNew(name, 1));
		});
	    }
	    else {

		if (this.logging) {
		    this.log('[input] primary keys:');
		    Object.keys(primaryClueData.nameMap).forEach(key => this.log('  ' + key));
		}
		
		// call validateSources recursively with compound clues
		srcNcList = [];
		if (!this.validateSources({
		    sum:            buildResult.count,
		    nameList:       buildResult.clueNameList,
		    count:          buildResult.clueNameList.length,
		    excludeSrcList: args.excludeSrcList,
		    validateAll:    args.validateAll,
		    wantResults:    args.wantResults,
		    vsCount:        args.vsCount,
		    resultMap:      args.resultMap
		}, srcNcList)) {
		    // possible edge case here. we don't want to ignore invalid known
		    // clue combinations, only invalid generated combos. so we may need
		    // differentiate the types of validation failure. duplicate?
		    // non-existant sources?
		    break;
		}

		// sanity check
		if (/*!args.validateAll && */(srcNcList.length < 2)) {
		    throw new Error('list should have at least two entries2');
		}
		nameCountList = srcNcList;
	    }

	    // here, we only add primary clues to the map if *all* clues in
	    // nameCountList are primary
	    findResult = this.findDuplicatePrimaryClue({
		ncList:              nameCountList, 
		clueData:            primaryClueData,
		onlyAddIfAllPrimary: true
	    });
	    if (!this.evalFindDuplicateResult(findResult, '2nd')) {
		break;
	    }

	    if (this.logging) {
		this.log('[nameMap] primary keys:');
		Object.keys(primaryClueData.nameMap).forEach(key => this.log('  ' + key));
		this.log('[srcMap] primary keys:');
		Object.keys(primaryClueData.srcMap).forEach(key => {
		    this.log('  ' + primaryClueData.srcMap[key] + ':' + key);
		});
		this.log(' all_primary: ' + findResult.allPrimary);
	    }

	    if (findResult.allPrimary) {
		// all the source clues we just validated are primary clues
		if (args.wantResults) {
		    // TODO: why isn't nameMap the right length?! is it now?
		    if ((args.vsCount == Object.keys(primaryClueData.srcMap).length)) {
			if (this.addResultEntry({
			    map:           args.resultMap,
			    key:           this.PRIMARY_KEY,
			    name:          this.FINAL_KEY,
			    ncList:        this.getClueDataNcList(primaryClueData),
			    excludeSrcList:args.excludeSrcList
			})) {
			    this.moveStagingToCompound({
				map:    args.resultMap,
				ncList: nameCountList
			    });
			}
		    }
		}
		if (args.validateAll) {
		    anyFlag = true; // success - tryAll - try other variations
		    break; 
		}
		return true; // success - exit function
	    }
	    
	    if (this.logging) {
		this.log('++innner looping');
	    }
	}
	
	if (!this.incrementIndexLengths(buildArgs.indexLengthMap)) {
	    if (this.logging) {
		this.log('done')
	    }
	    return args.validateAll ? anyFlag : false;
	}

	if (this.logging) {
	    this.log('++outer looping');
	}
    }
}

// ncList:              nameCountList
// clueData:            primaryClueData
// onlyAddIfAllPrimary: boolean
//
Validator.prototype.findDuplicatePrimaryClue = function(args) {
    var duplicateName;
    var duplicateSrc;
    var duplicateSrcName;
    var allPrimary;
    var conflictSrcMap;
    var localSrcMap;
    var key;
    var result;

    // copy srcMap locally
    localSrcMap = {};
    for (key in args.clueData.srcMap) {
	localSrcMap[key] = args.clueData.srcMap[key];
    }

    conflictSrcMap = {};
    // look for duplicate primary clue sources, add conflicts to map
    result = this.findDuplicatePrimarySource({
	ncList:          args.ncList,
	nameMap:         args.clueData.nameMap,
	srcMap:          localSrcMap,
	conflictSrcMap:  conflictSrcMap
    });
    duplicateName = result.duplicateName;
    allPrimary = result.allPrimary;

    // resolve duplicate primary source conflicts
    result = this.resolvePrimarySourceConflicts({
	srcMap:         localSrcMap,
	conflictSrcMap: conflictSrcMap
    });
    duplicateSrcName = result.duplicateSrcName;
    duplicateSrc = result.duplicateSrc;

    // add PRIMARY names and sources to their respective maps if all
    // sourcces are primary, or if onlyAddIfAllPrimary flag is NOT set
    //
    // NOTE that we have to do thise regardless of the existence of
    // duplicate names/sources, because this function doesn't enforce
    // rules about allowing those
    if (allPrimary || !args.onlyAddIfAllPrimary) {
	args.ncList.forEach(nc => {
	    if (nc.count == 1) {
		args.clueData.nameMap[nc.name] = true;
	    }
	});
	for (key in localSrcMap) {
	    args.clueData.srcMap[key] = localSrcMap[key];
	}
    }

    if (this.loggign) {
	this.log('findDuplicatePrimaryClue, duplicateName: ' + duplicateName +
		 ', duplicateSrc: ' + duplicateSrc);
    }

    return {
	duplicateName:    duplicateName,
	duplicateSrcName: duplicateSrcName,
	duplicateSrc:     duplicateSrc,
	allPrimary:       allPrimary
    };
}


// args:
//  ncList:          args.ncList,
//  nameMap:         args.clueData.nameMap,
//  srcMap:          localSrcMap,
//  conflictSrcMap : conflictSrcMap
//

Validator.prototype.findDuplicatePrimarySource = function(args) {
    var duplicateName;
    var allPrimary = true;

    if (!args.ncList || !args.nameMap || !args.srcMap ||
	!args.conflictSrcMap) {
	throw new Error('missing args, srcMap: ' + args.srcMap +
			' conflictSrcMap: ' + args.conflictSrcMap +
			' ncList: ' + args.ncList +
			' nameMap: ' + args.nameMap);
    }

    args.ncList.forEach(nc => {
	var srcList;

	if (nc.count == 1) {
	    if (args.nameMap[nc.name]) {
		duplicateName = nc.name;
	    }
	    srcList = ClueManager.knownClueMapArray[1][nc.name]; 
	    // look for an as-yet-unused src for the given clue name
	    if (!srcList.some(src => {
		if (!args.srcMap[src]) {
		    args.srcMap[src] = nc.name;
		    return true; // found; some.exit
		}
		return false; // not found; some.continue
	    })) {
		// unused src not found: add to conflict map, resolve later
		if (!args.conflictSrcMap[srcList]) {
		    args.conflictSrcMap[srcList] = { list: [ nc.name ] };
		}
		else {
		    args.conflictSrcMap[srcList].list.push(nc.name);
		}
	    }
	}
	else {
	    allPrimary = false;
	}
    }, this);

    if (this.logging) {
	this.log('findDuplicatePrimarySource: ' + 
		 (duplicateName ? duplicateName : 'none') +
		 ', allPrimary: ' + allPrimary);
    }

    return {
	duplicateName: duplicateName,
	allPrimary:    allPrimary
    };
}

// args:
//  srcMap:         localSrcMap,
//  conflictSrcMap: conflictSrcMap
//

Validator.prototype.resolvePrimarySourceConflicts = function(args) {
    var duplicateSrcName;
    var duplicateSrc;
    var conflictSrc;
    var conflictNameList;
    var srcList;
    var key;

    if (!args.srcMap || !args.conflictSrcMap) {
	throw new Error('missing args, srcMap:' + args.srcMap +
			' conflictSrcMap: ' + args.conflictSrcMap);
    }

    // resolve primary source conflicts
    for (conflictSrc in args.conflictSrcMap) {

	if (this.logging) {
	    this.log('Attempting to resolve conflict...');
	}

	srcList = conflictSrc.split(',');
	conflictNameList = args.conflictSrcMap[conflictSrc].list;

	// if conflictList.length > srcList.length then there
	// are more uses of this clue than there are sources for it.
	if (conflictNameList.length > srcList.length) {
	    duplicateSrcName = conflictNameList.toString();
	    duplicateSrc = conflictSrc;
	    break;
	}
	// otherwise we may be able to support the clue count; see
	// if any conflicting clue names can be moved to other sources
	else if (!srcList.some(src => {
	    var conflictSrcList;
	    var candidateName;
	    var candidateSrcList;

	    // look for alternate unused sources for candidateName
	    candidateName = args.srcMap[src];
	    candidateSrcList = ClueManager.knownClueMapArray[1][candidateName]; 
	    if (candidateSrcList.some(candidateSrc => {
		if (!args.srcMap[candidateSrc]) {
		    this.log('resolved success!');
		    // success! update srcMap
		    args.srcMap[candidateSrc] = candidateName;
		    args.srcMap[src] = conflictNameList.pop();
		    if (conflictNameList.length == 0) {
			return true; // candidateSrcList.some.exit
		    }
		}
		return false; // candidateSrcList.some.next
	    })) {
		return true; // srcList.some.stop
	    }
	    return false; // srcList..some.next
	}, this)) {
	    // failed to find an alternate unused source for all conflict names
	    duplicateSrcName = conflictNameList.toString();
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
	// TODO! get rid of, or comment purpose of, else clause
	else { /* conflictNameList.length == 0 */
	    this.log('looping!');
	    //break;
	    // keep trying
	}
    }
    return {
	duplicateSrcName: duplicateSrcName,
	duplicateSrc:     duplicateSrc
    };
}

//
//

Validator.prototype.checkDuplicateSource = function(args, name, src) {
    if (args.clueData.srcMap[src]) {
	return true;
    }
    if (!args.onlyAddIfAllPrimary) {
	args.clueData.srcMap[src] = name;
    }
    return false;
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

    if (dupeType) {
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

// For compound clues in ncList, add component source clue names to
// the supplied clueNameList.
//
// NOTE: is problem below solved now?
// So there is some potenetial trickiness here for a 100% correct solution.
// If a clue has multiple source combinations, we technically need to build
// a separate clueNameList for each possible combination. if two or more clues
// have multiples source combinations, we need to build all combinations of those
// combinations. for the second case, punt until it happens.

Validator.prototype.buildCompoundClueSourceNameList = function(args) {
    var allPrimary = true;
    var clueCount = 0;    // # of primary clues represented by component source clues
    var duplicateName;
    var clueNameList;
    var indexLengthMap;

    if (this.logging) {
	this.log('buildCompoundCluesSourceNameList, ncList(' + args.ncList.length + ')');
    }
 
    if (args.indexLengthMap) {
	if (this.logging) {
	    this.log('using index map');
	}
	indexLengthMap = args.indexLengthMap;
    }
    else {
	if (this.logging) {
	    this.log('no index map');
	}
	indexLengthMap = args.indexLengthMap = {};
    }

    clueNameList = [];
    if (!args.ncList.every((nc, ncIndex) => {
	var srcList;          // e.g. [ 'src1,src2,src3', 'src2,src3,src4' ]
	var slIndex;          // srcListIndex
	var srcNameList;      // e.g. [ 'src1', 'src2', 'src3' ]
	var thisPrimary;
	
	// skip primary clues, we only care about compound clues here
	if (nc.count == 1) {
	    return true; // ncList.next
	}

	if (this.logging) {
	    this.log('adding compound clue: nc.name = ' +
		     nc.name + ', nc.count = ' + nc.count);
	}
	
	clueCount += nc.count;
	srcList = ClueManager.knownClueMapArray[nc.count][nc.name];
	if (!srcList) {
	    throw new Error('kind of impossible but missing clue!');
	}
	
	if (indexLengthMap[nc]) {
	    slIndex = indexLengthMap[nc].index;

	    // sanity check
	    if (indexLengthMap[nc].length != srcList.length) {
		throw new Error('mismatched index lengths');
	    }

	    if (this.logging) {
		this.log(nc.name + ': using preset index ' + slIndex +
			 ', length(' + indexLengthMap[nc].length + ')' +
			 ', actual length(' + srcList.length + ')');
	    }
	}
	else {
	    slIndex = 0;
	    if (this.logging) {
		this.log(nc.name + ': using first index ' + slIndex +
			 ', actual length(' + srcList.length + ')');
	    }
	    indexLengthMap[nc] = { index: 0, length: srcList.length };
	}

	srcNameList = srcList[slIndex].split(',');
	thisPrimary = (srcNameList.length == nc.count);
	if (!thisPrimary) {
	    allPrimary = false;
	}
	srcNameList.forEach(name => clueNameList.push(name));

	if (args.wantResults) {
	    this.addResultEntry({
		map:    args.resultMap,
		key:    this.STAGING_KEY,
		name:   nc.toString(),
		ncList: Array.from(srcNameList)
	    });
	}

	return true; // ncList.next
    }, this)) {
	clueCount = 0;
    }

    return {
	clueNameList:  clueNameList, 
	count:         clueCount,
	duplicateName: duplicateName,
	allPrimary:    allPrimary
    };
}

//
//

Validator.prototype.incrementIndexLengths = function(indexLengthMap) {
    var keyList;
    var keyIndex;
    var indexLength;

    if (!indexLengthMap) {
	return false;
    }

    // TODO: this is a bit flaky. assuming the order of keys isn't changing.
    keyList = Object.keys(indexLengthMap);

    // start at last index
    keyIndex = keyList.length - 1;
    indexLength = indexLengthMap[keyList[keyIndex]];
    ++indexLength.index;
    
    // while index is maxed reset to zero, increment next-to-last index, etc.
    // using >= because it's possible both index and length are zero, for
    // primary clues, which are skipped.
    while (indexLength.index >= indexLength.length) {
	indexLength.index = 0;
	--keyIndex;
	if (keyIndex < 0) {
	    return false;
	}
	indexLength = indexLengthMap[keyList[keyIndex]];
	++indexLength.index;
    }
    return true;
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

    newNcList = Array.from(ncList)
    newNcList.push(NameCount.makeNew(name, count));

    return newNcList;
}

//
//

Validator.prototype.getDiffNcList = function(origNameCountList, nameCountList) {
    var ncList;
    var index;
    
    ncList = [];
    for (index = origNameCountList.length; index < nameCountList.length; ++index) {
	ncList.push(nameCountList[index]);
    }
    return ncList;
}

//
//

Validator.prototype.getClueDataNcList = function(clueData) {
    var srcMapKey;
    var ncList;
    
    ncList = [];
    for (srcMapKey in clueData.srcMap) {
	ncList.push(NameCount.makeNew(clueData.srcMap[srcMapKey], srcMapKey));
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

// map:      args.resultMap,
//  key:      PRIMARY_KEY
//  name:     clue name
//  ncList:   ncList
//  excludeSrcList: list of excluded primary sources
//
Validator.prototype.addResultEntry = function(args) {
    var map;
    var srcList;

    if (args.excludeSrcList) {
	if (this.listContainsAny(args.excludeSrcList,
				 NameCount.makeCountList(args.ncList)))
	{
	    return false;
	}
    }

    if (!args.map[args.key]) {
	map = args.map[args.key] = {}
    }
    else {
	map = args.map[args.key];
    }
    if (!map[args.name]) {
	map[args.name] = [ args.ncList.toString() ];
    }
    else {
	if (map[args.name].indexOf(args.ncList.toString()) == -1) {
	    map[args.name].push(args.ncList.toString());
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
//

Validator.prototype.displayClueData = function(clueData) {
    console.log('src: ' + this.getClueDataNcList(clueData));
}

//
//

Validator.prototype.getClueDataNcList = function(clueData) {
    var srcMapKey;
    var ncList;
    
    ncList = [];
    for (srcMapKey in clueData.srcMap) {
	ncList.push(NameCount.makeNew(clueData.srcMap[srcMapKey], srcMapKey));
    }
    return ncList;
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
		console.log(name + ': ' + ncList);
	    });
	}
    });
}

