//
// VALIDATOR.JS
//

'use strict';

// export a singleton

module.exports = exports = new Validator();

//

const _             = require('lodash');
const ClueList      = require('./clue_list');
const ClueManager   = require('./clue_manager');
//const Expect        = require('chai').expect;
const Expect        = require('should/as-function')
const NameCount     = require('./name_count');
const Peco          = require('./peco');
const ResultMap     = require('./resultmap');

//

//const xp = false;
const xp = true;

//

function Validator() {

    this.rvsSuccessSeconds = 0;
    this.rvsFailDuration  = 0;

    this.allowDupeNameSrc = false;
    this.allowDupeSrc     = false;
    this.allowDupeName    = true;

    if (0) {
	this.freezeLogging    = true;
	this.logging          = true;
    } else {
	this.freezeLogging    = false;
	this.logging          = false;
    }
    this.logLevel         = 0;

    // TODO: these are duplicated in ResultMap
    this.PRIMARY_KEY      = '__primary';
    this.SOURCES_KEY      = '__sources';
}

//

Validator.prototype.setLogging = function(flag) {
    if (!this.freezeLogging) {
	this.logging = flag;
    }
}

//

Validator.prototype.log = function(text) {
    console.log(this.indent() + text);
}

//

Validator.prototype.setAllowDupeFlags = function(args) {
    if (!_.isUndefined(args.allowDupeNameSrc)) {
	this.allowDupeNameSrc = args.allowDupeNameSrc;
    }
    if (!_.isUndefined(args.allowDupeName)) {
	this.allowDupeName = args.allowDupeName;
    }
    if (!_.isUndefined(args.allowDupeSrc)) {
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
//  validateAll:    flag; check all combinations
//  quiet:          flag; quiet Peco
//
// All the primary clues which make up the clues in clueNameList should
// be unique and their total count should add up to clueCount. Verify
// that some combination of the cluelists of all possible addends of
// clueCount makes this true.

Validator.prototype.validateSources = function(args) {
    if (this.logging) {
	this.log('++validateSources' +
		 `${this.indentNewline()}  nameList(${args.nameList.length}): ${args.nameList}` +
		 `, sum(${args.sum})` +
		 `, count(${args.count})` +
		 `, excludeSrcList: ${args.excludeSrcList}` +
		 `${this.indentNewline()}  require: ${args.require}` +
		 `, exclude: ${args.exclude}` +
		 `, validateAll: ${args.validateAll}`);
    }

    let found = false;
    let resultList = [];
    Peco.makeNew({
	sum:     args.sum,
	count:   args.count,
	max:     args.max,
	require: args.require,
	exclude: args.exclude,
	quiet:   args.quiet
    }).getCombinations().some(clueCountList => {
	let rvsResult = this.recursiveValidateSources({
	    clueNameList:   args.nameList,
	    clueCountList:  clueCountList,
	    excludeSrcList: args.excludeSrcList,
	    validateAll:    args.validateAll
	});
	if (rvsResult.success) {
	    if (this.logging) {
		this.log('validateSources: VALIDATE SUCCESS!');
	    }
	    resultList.push(...rvsResult.list);
	    found = true;
	    // we may not care about all possible combinations
	    if (!args.validateAll) {
		return true; // found a match; some.exit
	    }
	    // validing all, continue searching
	    if (this.logging) {
		this.log('validateSources: validateAll set, continuing...');
	    }
	}
	return false; // some.continue
    }, this);

    if (this.logging) {
	this.log('--validateSources');
    }
    return {
	success:     found,
	list:        found ? resultList : undefined
    };
}

// args:
//  clueNameList:
//  clueCountList:
//  nameCountList:
//  excludeSrcList: list of excluded primary sources
//  validateAll:
//
// TODO: ForNameList

Validator.prototype.recursiveValidateSources = function(args) {
    if (xp) {
	Expect(args.clueNameList).is.Array().not.empty;
	Expect(args.clueCountList).is.Array().not.empty;
//	Expect(args.clueNameList).to.be.an('array').that.is.not.empty;
//	Expect(args.clueCountList).to.be.an('array').that.is.not.empty;
    }

    this.logLevel++;
    if (this.logging) {
	this.log('++recursiveValidateSources' +
		 ', looking for [' + args.clueNameList + ']' +
		 ' in [' + args.clueCountList + ']');
    }
    if (xp) Expect(args.clueNameList.length).is.equal(args.clueCountList.length);

    let ncList = _.isUndefined(args.nameCountList) ? [] : args.nameCountList;
    let nameIndex = 0;
    let clueName = args.clueNameList[nameIndex];
    let rvsResult;

    // optimization: could have a map of count:boolean entries here
    // on a per-name basis (new map for each outer loop; once a
    // count is checked for a name, no need to check it again

    let someResult = args.clueCountList.some(count => {
	if (this.logging) {
	    this.log('looking for ' + clueName + ' in ' + count);
	}
	if (!_.has(ClueManager.knownClueMapArray[count], clueName)) {
	    if (this.logging) {
		this.log(' not found, ' + clueName + ':' + count);
	    }
	    return false; // some.continue
	}
	if (this.logging) {
	    this.log(' found');
	}
	rvsResult = this.rvsWorker({
	    name:           clueName,
	    count:          count,
	    nameList:       args.clueNameList,
	    countList:      args.clueCountList,
	    ncList:         ncList,
	    excludeSrcList: args.excludeSrcList,
	    validateAll:    args.validateAll
	});
	if (!rvsResult.success) {
	    return false; // some.continue;
	}
	if (this.logging) {
	    this.log('--rvsWorker output for:' + clueName +
		     ', ncList(' + ncList.length + ') ' +
		     ' ' + ncList);
	}
	// sanity check
	if (!args.validateAll && (ncList.length < 2)) {
	    // TODO: add "allowSingleEntry" ?
	    // can i check vs. clueNameList.length?
	    // throw new Error('list should have at least two entries1');
	}
	return true; // success: some.exit
    });
    --this.logLevel;

    return {
	success: someResult,
	list:    (someResult ? rvsResult.list : undefined)
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
//
// TODO: ForName

Validator.prototype.rvsWorker = function(args) {
    if (this.logging) {
	this.log('++rvsWorker' +
		 ', name: ' + args.name +
		 ', count: ' + args.count +
		 ', validateAll: ' + args.validateAll +
		 this.indentNewline() + '  ncList: ' + args.ncList +
		 ', nameList: ' + args.nameList);
    }
    if (xp) {
	Expect(args.name).is.String().not.empty;
	Expect(args.count).is.Number()
	    .aboveOrEqual(1)
	    .belowOrEqual(ClueManager.maxClues);
	Expect(args.ncList).is.Array();
	Expect(args.nameList).is.Array();
/*
	Expect(args.name).to.be.a('string').that.is.not.empty;
	Expect(args.count).to.be.a('number')
	    .that.is.at.least(1)
	    .and.at.most(ClueManager.maxClues);
	Expect(args.ncList).to.be.an('array');
	Expect(args.nameList).to.be.an('array');
*/
    }
	
    let newNameCountList = this.copyAddNcList(args.ncList, args.name, args.count);
    if (newNameCountList === null) {
	// TODO:
	// duplicate name:count entry. technically this is allowable for
	// count > 1 if the there are multiple entries of this clue name
	// in the clueList[count]. (at least as many entries as there are
	// copies of name in ncList)
	// SEE ALSO: copyAddNcList()
	if (this.logging) {
	    this.log('++rvsWorker, duplicate name:count, ' + args.name + ':' + args.count);
	}
	return { success: false }; // fail
    }
    if (this.logging) {
	this.log('added NC name: ' + args.name +
		 ', count: ' + args.count +
		 ', list.length: ' + newNameCountList.length);
    }
    // If only one name & count remain, we're done.
    // (name & count lists are equal length, just test one)
    if (args.nameList.length === 1) {
	let uniqResult = this.checkUniqueSources(newNameCountList, args);
	if (uniqResult.success) {
	    if (this.logging) {
		this.log('checkUniqueSources --- success!');
	    }
	    args.ncList.push(NameCount.makeNew(args.name, args.count));
	    if (this.logging) {
		this.log('add1, ' + args.name + ':' + args.count +
			 ', newNc(' + newNameCountList.length + ')' +
			 ', ' + newNameCountList);
	    }
	    return {
		success: true,
		list:    uniqResult.list
	    };
	}
	if (this.logging) {
	    this.log('checkUniqueSources --- failure');
	}
	return { success: false }; // fail
    }

    // nameList.length > 1, remove current name & count,
    // and validate remaining
    if (this.logging) {
	this.log(`calling rvs recursively, ncList: ${newNameCountList}`);
    }
    let rvsResult = this.recursiveValidateSources({
	clueNameList:  this.chop(args.nameList, args.name),
	clueCountList: this.chop(args.countList, args.count),
	nameCountList: newNameCountList,
	excludeSrcList:args.excludeSrcList,
	validateAll:   args.validateAll
    });
    if (!rvsResult.success) {
	if (this.logging) {
	    this.log('--rvsWorker, recursiveValidateSources failed');
	}
	return { success: false }; // fail
    }
    // does this achieve anything? modifies args.ncList.
    // TODO: probably need to remove why that matters.
    // TODO2: use _clone() until then
    args.ncList.length = 0;
    newNameCountList.forEach(nc => args.ncList.push(nc));
    if (this.logging) {
	this.log('--rvsWorker, add ' + args.name + ':' + args.count +
		 ', newNcList(' + newNameCountList.length + ')' +
		 ', ' + newNameCountList);
    }
    return {
	success: true,
	list:    rvsResult.list
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
///
Validator.prototype.checkUniqueSources = function(nameCountList, args) {
    let origNcList = nameCountList;

    // assert(nameCountList) && Array.isArray(nameCountList)

    if (this.logging) {
	this.log('++checkUniqueSouces' +
		 ', name: ' + args.name +
		 ', count: ' + args.count +
		 ', nameList: ' + args.nameList +
		 ', validateAll: ' + args.validateAll +
		 ', ncList: ' + args.nameCountList);
    }

    // first, check for duplicate primary clues, add all to primaryMap
    let findResult = this.findDuplicatePrimaryClue({ ncList: nameCountList });
    if (!this.evalFindDuplicateResult(findResult, '1st')) {
	if (this.logging) {
	    this.log('FAIL , duplicate primary' +
		     ', nameCountList; ' + nameCountList);
	}
	return this.uniqueResult(false); // failure
    }
    else if (findResult.allPrimary) {
	let nameSrcList = this.getNameSrcList(findResult.srcMap);
	let resultMap = ResultMap.makeNew();
	resultMap.addPrimaryLists(nameCountList, nameSrcList);
	let compatList = this.getCompatibleResults({
	    origNcList:     nameCountList,
	    ncList:         nameCountList,
	    nameSrcList:    nameSrcList,
	    pendingMap:     resultMap,
	    excludeSrcList: args.excludeSrcList,
	    validateAll:    args.validateAll
	});
	if (!_.isEmpty(compatList)) {
	    return this.uniqueResult(true, compatList);
	}
	return this.uniqueResult(false);
    }

    let resultMap;
    let buildResult;
    let candidateResultList;
    let anyFlag = false;
    let resultList = [];
    let buildArgs = {
	ncList: nameCountList
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
//	    if (xp) Expect(buildResult.count).to.be.at.most(ClueManager.maxClues);
	    if (xp) Expect(buildResult.count).is.belowOrEqual(ClueManager.maxClues);

	    // skip recursive call to validateSources if we have all primary clues
	    if (buildResult.allPrimary) {
		if (this.logging) {
		    this.log('cUS: adding all_primary result: ' +
			     nameCountList + ' = ' + buildResult.primaryNcList);
		}
		nameCountList = buildResult.primaryNcList;
		candidateResultList = [{
		    ncList:    buildResult.primaryNcList,
		    resultMap: buildResult.resultMap // no need to merge?
		}];
	    } else {
		// call validateSources recursively with compound clues
		let vsResult = this.validateSources({
		    sum:            buildResult.count,
		    nameList:       buildResult.compoundSrcNameList,
		    count:          buildResult.compoundSrcNameList.length,
		    excludeSrcList: args.excludeSrcList,
		    validateAll:    true  // always validate all on recursive call
		});
		if (!vsResult.success) {
		    break; // fail, try other combos
		}
		// sanity check
		if (xp) Expect(vsResult.list).is.not.empty;
		if (this.logging) {
		    this.log('from validateSources(' + buildResult.count + ')');
		    this.log('  compoundSrcNameList: ' + buildResult.compoundSrcNameList);
		    this.log('  list.size: ' + _.size(vsResult.list));
		    vsResult.list.forEach(result => {
			this.log('   ncList:      ' + result.ncList);
			this.log('   nameSrcList: ' + result.nameSrcList);
			this.log('   -----------');
		    });
		}
		// we only sent compound clues to validateSources, so add the primary
		// clues that were filtered out by build(), to make a complete list.
		// also merge buildResults data into resultMap.
		candidateResultList = vsResult.list.map(result => Object({
		    ncList:    _.concat(result.ncList, buildResult.primaryNcList),
		    resultMap: _.cloneDeep(buildResult.resultMap).merge(result.resultMap, buildResult.compoundNcList)
		}));
	    }

	    let anyCandidate = false;
	    candidateResultList.some(result => {
		let findResult = this.findDuplicatePrimaryClue({ ncList: result.ncList });
		if (xp) Expect(findResult.allPrimary).is.true;
		if (!this.evalFindDuplicateResult(findResult, '2nd')) {
		    return false; // some.continue
		}
		let compatList = this.getCompatibleResults({
		    origNcList:     buildArgs.ncList,
		    ncList:         result.ncList,
		    nameSrcList:    this.getNameSrcList(findResult.srcMap),
		    pendingMap:     result.resultMap, 
		    excludeSrcList: args.excludeSrcList,
		    validateAll:    args.validateAll,
		    ncNameListPairs:buildResult.compoundNcNameListPairs // TODO: remove here and in build()
		});
		if (!_.isEmpty(compatList)) {
		    anyCandidate = true;
		    compatList.forEach(result => {
			if (!this.hasNameSrcList(resultList, result.nameSrcList)) {
			    resultList.push(result);
			}
		    });
		    // TODO: remove duplicates in uniqueResults()
		    //resultList = _.concat(resultList, compatList);
		    return !args.validateAll; // some.exit if !validateAll, else some.continue
		}
		return false; // some.continue;
	    });
	    if (!anyCandidate) {
		break; // none of those results were good, try other combos
	    }
	    anyFlag = true;
	    if (args.validateAll) {
		break; // success , but keep searching for other combos
	    }
	    if (this.logging) {
		this.log('--checkUniqueSources, single validate, success : ' + anyFlag);
	    }
	    return this.uniqueResult(anyFlag, resultList); // success , exit function
	}
	// sanity check
	if (xp) Expect(buildResult, 'buildResult').exist;
	if (!this.incrementIndexMap(buildResult.indexMap)) {
	    if (this.logging) {
		this.log(`--checkUniqueSources, full validate, ${anyFlag}`);
	    }
	    return this.uniqueResult(anyFlag, resultList);
	}
	if (this.logging) {
	    this.log('++outer looping');
	}
    }
}

//
Validator.prototype.uniqueResult = function(success, list) {
    return {
	success: success,
	list:    list // _.uniqBy(resultList, () => {})
    };
}

//
//
//  origNcList:
//  ncList:
//  nameSrcList:
//  excludeSrcList:
//  pendingMap:
//  validateAll:
//  ncNameListPairs:
//
Validator.prototype.getCompatibleResults = function(args) {
    // no duplicates, and all clues are primary, success!
    if (this.logging) {
	this.log('++allUniquePrimary' +
		 this.indentNewline() + '  origNcList:  ' + args.origNcList +
		 this.indentNewline() + '  ncList:      ' + args.ncList +
		 this.indentNewline() + '  nameSrcList: ' + args.nameSrcList);
    }
    if (xp) {
	Expect(args.origNcList).is.Array().not.empty;
	Expect(args.ncList).is.Array().not.empty;
	Expect(args.nameSrcList).is.Array().not.empty;
/*
	Expect(args.origNcList).to.be.an('array').that.is.not.empty;
	Expect(args.ncList).to.be.an('array').that.is.not.empty;
	Expect(args.nameSrcList).to.be.an('array').that.is.not.empty;
*/
    }

    let logit = false;
    let resultList = [];
    if (!this.hasExcludedSource(args.nameSrcList, args.excludeSrcList)) {
	if (logit || this.logging) {
	    this.log('aUP: adding primary result');
	    this.log(`  ${args.ncList}`);
	    this.log(`  -as- ${args.nameSrcList}`);
	}
	this.addCompatibleResult(resultList, args.nameSrcList, args);
    }
    if (_.isEmpty(resultList) || args.validateAll) {
	this.cyclePrimaryClueSources({
	    ncList:         args.ncList,
	    excludeSrcList: args.excludeSrcList
	}).some(nameSrcList => {
	    // check if nameSrcList is already in result list
	    if (this.hasNameSrcList(resultList, nameSrcList)) {
		if (logit || this.logging) {
		    this.log(`aUP cycle: already in resultList: ${nameSrcList}`);
		}
		return false;
	    }
	    if (logit || this.logging) {
		this.log('aUP cycle: adding primary result');
		this.log(`  ${args.ncList}`);
		this.log(`  -as- ${nameSrcList}`);
	    }
	    this.addCompatibleResult(resultList, nameSrcList, args);
	    return !args.validateAll; // some.exit if !validateAll, else some.continue
	});
    }
    return resultList;
}

//
Validator.prototype.addCompatibleResult = function(resultList, nameSrcList, args) {
    if (xp) {
	Expect(resultList).is.Array();
	Expect(nameSrcList).is.Array();
	Expect(args.ncList).is.Array();
/*
	Expect(resultList).to.be.an('array');
	Expect(nameSrcList).to.be.an('array');
	Expect(args.ncList).to.be.an('array');
*/
    }

    resultList.push({
	ncList:      args.ncList,
	nameSrcList: nameSrcList,
	resultMap:   _.cloneDeep(args.pendingMap).addResult({
	    origNcList:      args.origNcList,
	    primaryNcList:   args.ncList,
	    nameSrcList:     nameSrcList,
	    ncNameListPairs: args.ncNameListPairs
	})
    });
}


// Simplified version of checkUniqueSources, for all-primary clues.
// Check all variations of any duplicate-sourced primary clues.
//
// args:
//  ncList:
//  exclueSrcList:
//
Validator.prototype.cyclePrimaryClueSources = function(args) {
    if (xp) Expect(args.ncList).is.Array().not.empty;

    if (this.logging) {
	this.log('++cyclePrimaryClueSources');
    }

    // must copy the NameCount objects within the list
    let localNcList = _.cloneDeep(args.ncList)
    let resultList = [];
    let buildArgs = {
	ncList:     args.ncList,   // always pass same unmodified ncList
	allPrimary: true
    };
    let buildResult;
    do {
	// build src name list of any duplicate-sourced primary clues
	if (buildResult) {
	    buildArgs.indexMap = buildResult.indexMap;
	}
	buildResult = this.buildSrcNameList(buildArgs);

	// change local copy of ncList sources to buildResult's sources
	localNcList.forEach((nc, index) => {
	    localNcList[index].count = buildResult.primarySrcNameList[index];
	});

	// TODO: return srcMap in findResult
	let srcMap = {};
	let findResult = this.findDuplicatePrimarySource({
	    ncList: localNcList,
	    srcMap: srcMap
	});
	if (findResult.duplicateSrc) {
	    continue;
	}
	if (xp) Expect(_.size(localNcList), 'localNcList').is.equal(_.size(srcMap));

	let nameSrcList = this.getNameSrcList(srcMap);
	if (this.logging) {
	    this.log('[srcMap] primary keys: ' + nameSrcList);
	}
	// all the source clues we just validated are primary clues
	if (!this.hasExcludedSource(nameSrcList, args.excludeSrcList)) {
	    if (this.logging) {
		this.log('cycle: adding result: ' +
			 `${this.indentNewline()} ${args.ncList} = ` +
			 `${this.indentNewline()}   ${nameSrcList}`);
	    }
	    resultList.push(nameSrcList);
	}
    } while (this.incrementIndexMap(buildResult.indexMap));

    if (this.logging) {
	this.log(`--cyclePrimaryClueSources, size: ${_.size(resultList)}`);
	resultList.forEach(result => {
	    this.log(`  list: ${result}`);
	});
    }

    return resultList;
}

// ncList:              nameCountList
//
Validator.prototype.findDuplicatePrimaryClue = function(args) {
    let duplicateName;
    let duplicateSrc;
    let duplicateSrcName;

    if (this.logging) {
	this.log('++findDuplicatePrimaryClue' +
		    ', ncList: ' + args.ncList);
    }

    // look for duplicate primary clue sources, return conflict map
    // also checks for duplicate names
    let findResult = this.findPrimarySourceConflicts({ ncList: args.ncList });
    duplicateName = findResult.duplicateName;

    if (!_.isEmpty(findResult.conflictSrcMap)) {
	// resolve duplicate primary source conflicts
	let resolveResult = this.resolvePrimarySourceConflicts({
	    srcMap:         findResult.srcMap,
	    conflictSrcMap: findResult.conflictSrcMap
	});
	duplicateSrcName = resolveResult.duplicateSrcName;
	duplicateSrc = resolveResult.duplicateSrc;
    }

    // log before possible exception, to provide more info
    if (this.logging) {
	this.log('--findDuplicatePrimaryClue' +
		 `, duplicateName: ${duplicateName}` +
		 `, duplicateSrcName: ${duplicateSrcName}` +
		 `, duplicateSrc: ${duplicateSrc}` +
		 `, allPrimary: ${findResult.allPrimary}` +
		 `, srcMap.size: ${_.size(findResult.srcMap)}`);
    }

    if (findResult.allPrimary && _.isUndefined(duplicateSrc) &&
	(_.size(findResult.srcMap) != _.size(args.ncList)))
    {
	console.log('ncList: ' + args.ncList);
	console.log('srcMap.keys: ' + _.keys(findResult.srcMap));
	throw new Error('srcMap.size != ncList.size');
    }

    return {
	duplicateName:    duplicateName,
	duplicateSrcName: duplicateSrcName,
	duplicateSrc:     duplicateSrc,
	allPrimary:       findResult.allPrimary,
	srcMap:           findResult.srcMap
    };
}


// args:
//  ncList:
//
// result:
//

Validator.prototype.findPrimarySourceConflicts = function(args) {
    let duplicateName;

    if (this.logging) {
	this.log(`++findPrimarySourceConflicts, ncList: ${args.ncList}`);
    }

    if (xp) Expect(args.ncList, 'args.ncList').exist;

    let allPrimary = true;
    let nameMap = {};
    let srcMap = {};
    let conflictSrcMap = {};

    args.ncList.forEach(nc => {
	if (nc.count > 1) {
	    if (this.logging) {
		this.log(`fPSC: non-primary, ${nc}`);
	    }
	    allPrimary = false;
	    return; // forEach.continue
	}
	if (this.logging) {
	    this.log(`fPSC: primary, ${nc}`);
	}

	// if name is in nameMap then it's a duplicate
	if (_.has(nameMap, nc.name)) {
	    duplicateName = nc.name;
	} else {
	    nameMap[nc.name] = true;
	}

	let srcList = ClueManager.knownClueMapArray[1][nc.name];
	//console.log(`srcList for ${nc.name}:1, ${srcList}`);
	// look for an as-yet-unused src for the given clue name
	if (!srcList.some(src => {
	    if (!_.has(srcMap, src)) {
		srcMap[src] = nc.name;
		return true; // found; some.exit
	    }
	    return false; // not found; some.continue
	})) {
	    // unused src not found: add to conflict map, resolve later
	    if (!_.has(conflictSrcMap, srcList)) {
		conflictSrcMap[srcList] = [];
	    }
	    conflictSrcMap[srcList].push(nc.name);
	}
    }, this);

    if (this.logging) {
	this.log('--findPrimarySourceConflicts: ' +
		 (duplicateName ? duplicateName : 'none') +
		 ', allPrimary: ' + allPrimary);
    }

    return {
	duplicateName:  duplicateName,
	allPrimary:     allPrimary,
	srcMap:         srcMap,
	conflictSrcMap: conflictSrcMap
    };
}

// args:
//  srcMap:
//  conflictSrcMap:
//

Validator.prototype.resolvePrimarySourceConflicts = function(args) {
    let duplicateSrcName;
    let duplicateSrc;

    if (!args.srcMap || !args.conflictSrcMap) {
	throw new Error('missing args' +
			', srcMap:' + args.srcMap +
			' conflictSrcMap: ' + args.conflictSrcMap);
    }

    if (this.logging) {
	this.log(`++resolvePrimarySourceConflicts`);
	this.log(`  srcMap keys: ${_.keys(args.srcMap)}`);
	this.log(`  conflictSrcMap keys: ${_.keys(args.conflictSrcMap)}`);
    }

    // resolve primary source conflicts
    _.keys(args.conflictSrcMap).every(conflictSrc => {
	let srcList = conflictSrc.split(',');
	let conflictNameList = args.conflictSrcMap[conflictSrc];
	if (this.logging) {
	    this.log(`Attempting to resolve source conflict at ${conflictSrc}, names: ${conflictNameList}`);
	}

	// if conflictNameList.length > srcList.length then there
	// are more uses of this clue than there are sources for it.
	if (conflictNameList.length > srcList.length) {
	    duplicateSrcName = conflictNameList.toString();
	    duplicateSrc = conflictSrc;
	    return false; // every.exit
	}
	// otherwise we may be able to support the clue count; see
	// if any conflicting clue names can be moved to other sources
	if (!srcList.some(src => {
	    // look for alternate unused sources for candidateName
	    let candidateName = args.srcMap[src];
	    let candidateSrcList = ClueManager.knownClueMapArray[1][candidateName];
	    if (this.logging) {
		this.log(`Candidate sources for ${candidateName}:${src} are [${candidateSrcList}]`);
	    }
	    if (candidateSrcList.some(candidateSrc => {
		if (!_.has(args.srcMap, candidateSrc)) {
		    if (this.logging) {
			this.log(`Successfully resolved ${conflictSrc} as ${candidateSrc}!`);
		    }
		    // success! update srcMap
		    args.srcMap[candidateSrc] = candidateName;
		    // any name will do?!
		    args.srcMap[src] = conflictNameList.pop();
		    if (_.isEmpty(conflictNameList)) {
			return true; // candidateSrcList.some.exit
		    }
		}
		return false; // candidateSrcList.some.continue
	    })) {
		return true; // srcList.some.exit
	    }
	    return false; // srcList.some.continue
	})) {
	    // failed to find an alternate unused source for all conflict names
	    duplicateSrcName = _.toString(conflictNameList);
	    duplicateSrc = conflictSrc;

	    if (this.logging) {
		this.log(`cannot resolve conflict, names: ${duplicateSrcName}, src: ${duplicateSrc}`);
		this.log('used sources, ');
		_.keys(args.srcMap).forEach(key => {
		    this.log(`  ${key}: ${args.srcMap[key]}`);
		});
	    }
	    return false; // conflictSrcMap.keys().every.exit
	}
	return true;
    });
    return {
	duplicateSrcName: duplicateSrcName,
	duplicateSrc:     duplicateSrc
    };
}

// args:
//  ncList:      // ALL PRIMARY clues in name:source format (not name:count)
//  srcMap:
//  nameMap:
//
Validator.prototype.findDuplicatePrimarySource = function(args) {
    if (xp) Expect(args.ncList).is.Array();

    let duplicateSrcName;
    let duplicateSrc;

    args.ncList.some(nc => {
	let src = nc.count;
	if (_.has(args.srcMap, src)) {
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
    let dupeType = '';
    let dupeValue = '';

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
	} else {
	    //console.log(logPrefix + ' duplicate primary ' + dupeType + ', ' + dupeValue);
	}
	return false;
    }
    return true;
}

// args:
//  ncList
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
    if (this.logging) {
	this.log('++buildSrcNameList, ncList(' + args.ncList.length + ')' +
		 this.indentNewline() + '  ncList: ' + args.ncList +
		 this.indentNewline() + '  allPrimary: ' + args.allPrimary);
    }

    let indexMap = this.getIndexMap(args.indexMap);
    let allPrimary = true;
    let clueCount = 0;
    let compoundNcList = [];
    let compoundSrcNameList = [];
    let compoundNcNameListPairs = [];
    let primaryNcList = [];
    let primarySrcNameList = [];
    let primaryPathList = []; //  TODO: i had half an idea here
    let resultMap = ResultMap.makeNew();

    args.ncList.forEach((nc, ncIndex) => {
	let src = args.allPrimary ? 1 : nc.count;
	// i.e. srcNameCsvArray
	let srcList = ClueManager.knownClueMapArray[src][nc.name]; // e.g. [ 'src1,src2,src3', 'src2,src3,src4' ]
	if (!srcList) {
	    throw new Error('kind of impossible but missing clue!');
	}

	// only do indexing if all clues are primary, or if this
	// is a compound clue
	let slIndex; // srcListIndex
	if (args.allPrimary || (nc.count > 1)) {
	    slIndex = this.getSrcListIndex(indexMap, nc, srcList);
	} else {
	    slIndex = 0;
	}
	if (this.logging ) {
	    this.log(`build: index: {$ncIndex}, source: ${srcList[slIndex]}`);
	}

	let srcNameList = srcList[slIndex].split(',');      // e.g. [ 'src1', 'src2', 'src3' ]
	if (nc.count === 1) {
	    primaryNcList.push(nc);
	    // short circuit if we're called with allPrimary:true.
	    if (args.allPrimary) {
		primarySrcNameList.push(...srcNameList);
		return; // forEach.next
	    }
	}
	if (xp) Expect(resultMap.map()[nc]).is.undefined;

	// if nc is a primary clue
	if (nc.count == 1) {
	    // add map entry for list of primary name:sources
	    if (!_.has(resultMap.map(), this.PRIMARY_KEY)) {
		resultMap.map()[this.PRIMARY_KEY] = [];
	    }
	    resultMap.map()[this.PRIMARY_KEY].push(_.toString(nc)); // consider nc.name here instead
	    return; // forEach.next;
	}
	
	// nc is a compound clue
	compoundNcNameListPairs.push([nc, _.clone(srcNameList)]);
	let map = resultMap.map()[nc] = {};
	// if sources for this nc are all primary clues
	if (_.size(srcNameList) === nc.count) {
	    // build component primary NC list
	    let localPrimaryNcList = srcNameList.map(name => NameCount.makeNew(name, 1));
	    // add map entry for list of (eventual) primary name:sources
	    map[localPrimaryNcList] = [];
	    primaryNcList.push(...localPrimaryNcList);
	    return; // forEach.next;
	}

	// sources for this nc include a compound clue
	clueCount += nc.count;
	allPrimary = false;

	// add map entry for list of source names
	// why don't we just add empty maps here? because we don't
	// know the nc.count for these names
	map[this.SOURCES_KEY] = srcNameList;
	compoundSrcNameList.push(...srcNameList);
	compoundNcList.push(nc);
    }, this);

    if (args.allPrimary && (primarySrcNameList.length != args.ncList.length)) {
	throw new Error(`something went wrong, primary: ${primarySrcNameList.length}` +
			`, ncList: ${args.ncList.length}`);
    }

    if (this.logging) {
	this.log('--buildSrcNameList');
	this.log(`  compoundSrcNameList: ${compoundSrcNameList}`);
	this.log(`  compoundNcList: ${compoundNcList}`);
	this.log(`  count: ${clueCount}`);
	this.log(`  primarySrcNameList: ${primarySrcNameList}`);
	this.log(`__primaryNcList: ${primaryNcList}`);
	// this.indentNewline() + '  compoundNcNameListPairs: ' + compoundNcNameListPairs +

	if (this.logResults) {
	    if (!_.isEmpty(resultMap.map())) {
		this.log('resultMap:');
		resultMap.dump();
	    } else {
		this.log('resultMap: empty');
	    }
	}
    }
    return {
	compoundNcNameListPairs,
	compoundSrcNameList,
	compoundNcList,
	primaryNcList,
	primarySrcNameList,
	resultMap,
	allPrimary,
	indexMap,
	count: clueCount,
    };
}

//
//

Validator.prototype.getIndexMap = function(indexMap) {
    if (!_.isUndefined(indexMap)) {
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
    let slIndex;
    if (_.has(indexMap, nc)) {
	slIndex = indexMap[nc].index;
	// sanity check
	if (xp) Expect(indexMap[nc].length, 'mismatched index lengths').is.equal(srcList.length);
	if (this.logging) {
	    this.log(nc.name + ': using preset index ' + slIndex +
		     ', length(' + indexMap[nc].length + ')' +
		     ', actual length(' + srcList.length + ')');
	}
    } else {
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
    if (xp) Expect(indexMap, 'bad indexMap').is.Object().not.empty;
    if (this.logging) {
	this.log('++indexMap: ' + this.indexMapToJSON(indexMap));
    }

    // TODO: this is a bit flaky. assumes the order of keys isn't changing.
    let keyList = Object.keys(indexMap);

    // start at last index
    let keyIndex = keyList.length - 1;
    let indexObj = indexMap[keyList[keyIndex]];
    indexObj.index += 1;

    // while index is maxed reset to zero, increment next-to-last index, etc.
    // using >= because it's possible both index and length are zero
    // for primary clues, which are skipped.
    while (indexObj.index >= indexObj.length) {
	if(this.logging) {
	    this.log('keyIndex ' + keyIndex + ': ' + indexObj.index +
		     ' >= ' + indexObj.length + ', resetting');
	}
	indexObj.index = 0;
	keyIndex -= 1;
	if (keyIndex < 0) {
	    return false;
	}
	indexObj = indexMap[keyList[keyIndex]];
	indexObj.index += 1;
	if (this.logging) {
	    this.log('keyIndex ' + keyIndex + ': ' + indexObj.index +
		     ', length: ' + indexObj.length);
	}
	    
    }
    if (this.logging) {
	this.log('--indexMap: ' + this.indexMapToJSON(indexMap));
    }
    return true;
}

//
//

Validator.prototype.indexMapToJSON = function(map) {
    let s = '';
    _.keys(map).forEach(key => {
	if (s.length > 0) {
	    s += ',';
	}
	s += map[key].index;
    });
    return '[' + s + ']';
}

//
//

Validator.prototype.copyAddNcList = function(ncList, name, count) {
    let newNcList;

    // for non-primary check for duplicate name:count entry
    // technically this is allowable for count > 1 if the there are
    // multiple entries of this clue name in the clueList[count].
    // (at least as many entries as there are copies of name in ncList)
    // TODO: make knownSourceMapArray store a count instead of boolean

    if (!ncList.every(nc => {
	if (nc.count > 1) {
	    if ((name === nc.name) && (count === nc.count)) {
		return false;
	    }
	}
	return true;
    })) {
	return null;
    }

    // TODO: _.clone()
    newNcList = Array.from(ncList);
    newNcList.push(NameCount.makeNew(name, count));

    return newNcList;
}

//
//

Validator.prototype.getDiffNcList = function(origNcList, nameCountList) {
    let ncList = [];
    for (let index = origNcList.length; index < nameCountList.length; ++index) {
	ncList.push(nameCountList[index]);
    }
    return ncList;
}

//
//

Validator.prototype.getNameSrcList = function(srcMap) {
    return _.keys(srcMap).map(key => NameCount.makeNew(srcMap[key], key));
}

//
//

Validator.prototype.chop = function(list, removeValue) {
    let copy = [];
    list.forEach(value => {
	if (value == removeValue) {
	    removeValue = undefined;
	} else {
	    copy.push(value);
	}
    });
    return copy;
}

//

Validator.prototype.hasNameSrcList = function(resultList, nameSrcList) {
    return resultList.some(result => {
	return result.nameSrcList.every((nameSrc, nsIndex) => {
	    return nameSrc.equals(nameSrcList[nsIndex]);
	});
    });
    
}

// args:
//  nameSrcList:
//  excludeSrcList: list of excluded primary sources
//
// TODO: NameCount.containsAnyCount(ncList-or-nc, count-or-countlist)
//
Validator.prototype.hasExcludedSource = function(nameSrcList, excludeSrcList) {
    return _.isUndefined(excludeSrcList) ? false :
	!_.isEmpty(_.intersection(excludeSrcList, nameSrcList.map(nc => nc.count)));
}

//
//

Validator.prototype.dumpIndexMap = function(indexMap) {
    let s = '';
    _.keys(indexMap).forEach(key => {
	let entry = indexMap[key];
	if (s.length > 0) {
	    s += '; ';
	}
	s += 'index ' + entry.index + ', length ' + entry.length;
    });
    this.log(s);
}

// args:
//  header:
//  result:
//  primary:  t/f
//  compound: t/f
//
Validator.prototype.dumpResult = function(args) {
    let header = 'Validate results';
    if (args.header) {
	header += ' for ' + args.header;
    }
    console.log(header);

    let keyNameList = [];
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
	header = keyName.name + ':';
	if (!args.result[keyName.key]) {
	    console.log(header + ' empty');
	    return;
	}
	let map = args.result[keyName.key];
	if (Object.keys(map).length == 0) {
	    console.log(header + ' empty');
	    return;
	}
	console.log(header);
	_.keys(map).forEach(name => {
	    map[name].forEach(ncList => {
		console.log(name + ':' + ncList);
	    });
	});
    });
}

// args:
//  header:
//  result:
//  primary:  t/f
//  compound: t/f
//
Validator.prototype.dumpResultMap = function(seq, level) {
    if (!level) level = 0;
    if (typeof seq === 'object') {
	console.log(this.indent() + spaces(2 * level) + (_.isArray(seq) ? '[' : '{'));
	++level;

	if (_.isArray(seq)) {
	    seq.forEach(elem => {
		if (typeof elem === 'object') {
		    this.dumpResultMap(elem, level + 1);
		} else {
		    console.log(this.indent() + spaces(2*level) + elem);
		}
	    }, this);
	} else {
	    _.forOwn(seq, function(value, key) {
		if (typeof value === 'object') {
		    console.log(this.indent() + spaces(2 * level) + key + ':');
		    this.dumpResultMap(value, level + 1);
		} else {
		    console.log(this.indent() + spaces(2 * level) + key + ': ' + value);
		}
	    }.bind(this));
	}

	--level;
	console.log(this.indent() + spaces(2 * level) + (_.isArray(seq) ? ']' : '}'));
    }
}

function spaces(length) {
    return ' '.repeat(length);
}

Validator.prototype.indent = function() {
    return spaces(this.logLevel);
}

Validator.prototype.indentNewline = function() {
    return '\n' + this.indent();
}
