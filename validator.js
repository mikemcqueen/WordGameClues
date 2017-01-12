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
    this.rvsFailDuration = 0;

    this.allowDupeNameSrc = false;
    this.allowDupeSrc     = false;
    this.allowDupeName    = true;

    this.logging = true;
    this.logLevel = 0;
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

//

Validator.prototype.log = function(text) {
    var pad = '';
    var index;
    for (var index=0; index<this.logLevel; ++index) {
	pad += ' ';
    }
    console.log(pad + text);
}

// args:
//  nameList: list of clue names, e.g. ['bob','jim']
//  sum:      primary clue count
//  max:      max # of sources to combine (either this -or- count must be set)
//  count:    exact # of sources to combine (either this -or- max must be set)
//  require:  list of required clue counts, e.g. [2, 4]
//  showAll:  flag; show all combinations
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
    var peco

    if (this.logging) {
	this.log('++validateSources' +
		 ', nameList(' + args.nameList.length + 
		 ', count(' + args.count + ')' +
		 ', require: ' + args.require +
		 ', showAll: ' + args.showAll);
	this.log('  names: ' + args.nameList);
    }

    found = false;

    vsCount = args.vsCount ? args.vsCount : args.sum;
    (new Peco({
	sum:     args.sum,
	count:   args.count,
	max:     args.max,
	require: args.require
    })).getCombinations().some(clueCountList => {
	if (this.recursiveValidateSources({
	    clueNameList:  args.nameList,
	    clueCountList: clueCountList,
	    nameCountList: nameCountList,
	    showAll:       args.showAll,
	    vsCount:       vsCount
	})) {
	    found = true;
	    if (!args.showAll) {
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

    return found;
}

//
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
	    name:      clueName,
	    count:     count,
	    nameList:  args.clueNameList,
	    countList: args.clueCountList,
	    ncList:    args.nameCountList,
	    showAll:   args.showAll,
	    vsCount:   args.vsCount      // aka "original count"
	})) {
	    return false; // some.continue;
	}
	
	if (this.logging) {
	    this.log('++rvsWorker output(' + args.nameCountList.length + ')');
	    args.nameCountList.forEach(nc => {
		this.log('nc.name = ' + nc.name + ', nc.count = ' + nc.count)
	    });
	    this.log('--rvsWorker output(' + args.nameCountList.length + ')');
	}
	
	if (!args.showAll && (args.nameCountList.length < 2)) {
	    throw new Error('list should have at least two entries1');
	}

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
//   fail     : failObject
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

	    args.ncList.push(new NameCount(args.name, args.count));

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
	showAll:       args.showAll,
	vsCount:       args.vsCount      // aka "original count"
    })) {
	return false; // fail
    }

    // does this achieve anything?
    args.ncList.length = 0;
    newNameCountList.forEach(nc => args.ncList.push(nc));

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

//
//

Validator.prototype.checkUniqueSources = function(nameCountList, args) {
    // assert(nameCountList) && Array.isArray(nameCountList)

    var origNameCountList = nameCountList;
    var primaryClueData = {};
    var srcNcList;
    var buildArgs;
    var result;
    var anyFlag = false;

    if (this.logging) {
	this.log('CheckUniqueSouces, ncList.length = ' + nameCountList.length);
    }

    buildArgs = {
	showAll: args.showAll
    };
    for(;;) {
	nameCountList = origNameCountList;
	primaryClueData.nameMap = {};
	primaryClueData.srcMap = {};
	buildArgs.primaryClueData = primaryClueData;
	
	// should have findDupe check here, and after build()
	
	for (;;) {

	    // first, check for duplicate primary clues, add all to primaryMap
	    result = this.findDuplicatePrimaryClue({
		ncList:              nameCountList, 
		clueData:            primaryClueData,
		onlyAddIfAllPrimary: false
	    });
	    if (this.evalFindDuplicateResult(result, '1st')) {

		if (this.logging) {
		    nameCountList.forEach(nc => {
			this.log('  nc.name = ' + nc.name + 
				 ' nc.count = ' + nc.count);
		    });
		    this.log('keys:');
		    Object.keys(primaryClueData.nameMap).forEach(key => {
			this.log('  ' + key);
		    });
		}

		break; // failure;
	    }
	    else if (result.allPrimary) {
		// if no duplicates, and all clues are primary, success!
		
		if (this.logging) {
		    this.log('SUCCESS! all primary');
		    nameCountList.forEach(nc => {
			this.log('nc.name = ' + nc.name + ', nc.count = ' + nc.count)
		    });
		}
		return true; // success
	    }
	    
	    // next, break down all compound clues into source components
	    result = this.buildCompoundClueSourceNameList(nameCountList, buildArgs);

	    // skip recursive call to validateSources if we have all primary clues
	    if (result.allPrimary) {
		// build an NC list of primary clues
		nameCountList = [];
		result.clueNameList.forEach(name => {
		    nameCountList.push(new NameCount(name, 1));
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
		    sum:      result.count,
		    nameList: result.clueNameList,
		    count:    result.clueNameList.length,
		    showAll:  args.showAll,
		    vsCount:  args.vsCount
		}, srcNcList)) {
		    // possible edge case here. we don't want to ignore invalid known
		    // clue combinations, only invalid generated combos. so we may need
		    // differentiate the types of validation failure. duplicate?
		    // non-existant sources?
		    break;
		}

		// sanity check
		if (!args.showAll && (srcNcList.length < 2)) {
		    throw new Error('list should have at least two entries2');
		}
		nameCountList = srcNcList;
	    }

	    // here, we only add primary clues to the map if *all* clues in
	    // nameCountList are primary
	    result = this.findDuplicatePrimaryClue({
		ncList:              nameCountList, 
		clueData:            primaryClueData,
		onlyAddIfAllPrimary: true
	    });
	    if (this.evalFindDuplicateResult(result, '2nd')) {
		break;
	    }

	    if (this.logging) {
		this.log('[nameMap] primary keys:');
		Object.keys(primaryClueData.nameMap).forEach(key => this.log('  ' + key));
		this.log('[srcMap] primary keys:');
		Object.keys(primaryClueData.srcMap).forEach(key => {
		    this.log('  ' + primaryClueData.srcMap[key] + ':' + key);
		});
		this.log(' all_primary: ' + result.allPrimary);
	    }

	    if (result.allPrimary) {
		// all the source clues we just validated  primary clues
		if (args.showAll) {
		    // TODO: why isn't nameMap the right length?!
		    if ((args.vsCount == Object.keys(primaryClueData.srcMap).length)) {
			this.displayClueData(primaryClueData);
		    }
		}
		anyFlag = true;
		if (args.showAll) {
		    break;
		}
		return true;
	    }

	    // !KEY_COMPLETE && !result.allPrimary
	    
	    if (this.logging) {
		this.log('++innner looping');
	    }
	}
	
	if (!this.incrementIndexLengths(buildArgs.indexLengthMap)) {
	    if (this.logging) {
		this.log('done')
	    }
	    return args.showAll ? anyFlag : false;
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
	return true;
    }
    return false;
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

Validator.prototype.buildCompoundClueSourceNameList = function(ncList, args) {
    var allPrimary = true;
    var clueCount = 0;    // # of primary clues represented by component source clues
    var duplicateName;
    var clueNameList;
    var indexLengthMap;

    if (this.logging) {
	this.log('buildCompoundCluesSourceNameList, ncList(' + ncList.length + ')');
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
    if (!ncList.every((nc, ncIndex) => {
	var srcStringList; // e.g. [ 'src1,src2,src3', 'src2,src3,src4' ]
	var sslIndex;
	var srcNameList;   // e.g. [ 'src1', 'src2', 'src3' ]
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
	srcStringList = ClueManager.knownClueMapArray[nc.count][nc.name];
	if (!srcStringList) {
	    throw new Error('kind of impossible but missing clue!');
	}
	
	if (indexLengthMap[nc]) {
	    sslIndex = indexLengthMap[nc].index;

	    // sanity check
	    if (indexLengthMap[nc].length != srcStringList.length) {
		throw new Error('mismatched index lengths');
	    }

	    if (this.logging) {
		this.log(nc.name + ': using preset index ' + sslIndex +
			 ', length(' + indexLengthMap[nc].length + ')' +
			 ', actual length(' + srcStringList.length + ')');
	    }
	}
	else {
	    sslIndex = 0;
	    if (this.logging) {
		this.log(nc.name + ': using first index ' + sslIndex +
			 ', actual length(' + srcStringList.length + ')');
	    }
	    indexLengthMap[nc] = { index: 0, length: srcStringList.length };
	}

	srcNameList = srcStringList[sslIndex].split(',');
	thisPrimary = (srcNameList.length == nc.count);
	if (!thisPrimary) {
	    allPrimary = false;
	}
	srcNameList.forEach(name => clueNameList.push(name));

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
    newNcList.push(new NameCount(name, count));

    return newNcList;
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

//
//

Validator.prototype.displayClueData = function(clueData) {
    var nameMapKey;
    var srcMapKey;
    var nameList;
    var ncList;
    
    //for (nameMapKey in clueData.nameMap) {
    ncList = [];
    for (srcMapKey in clueData.srcMap) {
	//nameList.push(clueData.srcMap[srcMapKey]);
	// don't know how to sort these yet
	ncList.push(new NameCount(clueData.srcMap[srcMapKey], srcMapKey));
    }
    console.log('src: ' + ncList);
}

//
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
