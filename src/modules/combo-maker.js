//
// combo-maker.js
//

'use strict';

// export a singleton

module.exports = exports = new ComboMaker();

//

const _           = require('lodash');
const ClueManager = require('./clue-manager');
const ClueList    = require('../types/clue-list');
const Debug       = require('debug')('combo-maker');
const Duration    = require('duration');
const Expect      = require('should/as-function');
const Log         = require('./log')('combo-maker');
const NameCount   = require('../types/name-count');
const Peco        = require('./peco');
const PrettyMs    = require('pretty-ms');
const ResultMap   = require('../types/result-map');
const Stringify   = require('stringify-object');
const Validator   = require('./validator');

let logging = 0;


const Op = { 'and':1, 'or':2, 'xor':3 };
Object.freeze(Op);

function OpName (opValue) {
    return _.findKey(Op, (v) => opValue === v);
}

//
//

function ComboMaker() {
    this.hash = {};
}

//

/** not used
ComboMaker.prototype.matchAny = function (srcList, nameList) {
    for (const source of srcList) {
        for (const name of nameList) {
            const regex = new RegExp(`${name}`);
            if (source.match(regex)) return true;
        }
    }
    return false;
};
*/

/** not used 
function getPrimaryNameSrcLists (entries) {
    const nameSrcLists = [];
    entries.forEach(entry => {
        entry.results.forEach(result => {
            nameSrcLists.push(result.ncList.map(nc => ClueManager.primaryNcToNameSrc(nc)));
        });
    });
    return nameSrcLists;
}
*/

/** not used
function getCompatiblePrimaryNameSrcLists (nameSrcLists1, nameSrcLists2) {
    let nameSrcLists = [];
    for (const nameSrcList1 of nameSrcLists1) {
        for (const nameSrcList2 of nameSrcLists2) {
            Debug(`nameSrcList1: ${nameSrcList1}, nameSrcList2: ${nameSrcList2}`);
            let countList1 = NameCount.makeCountList(nameSrcList1);
            let countList2 = NameCount.makeCountList(nameSrcList2);
            if (_.isEmpty(_.intersectionBy(countList1, countList2, _.toNumber))) {
                nameSrcLists.push(_.concat(nameSrcList1, nameSrcList2));
            }
        }
    }
    return nameSrcLists;
}
*/

function getSourcesList (srcMapEntries) {
    const sources = [];
    srcMapEntries.forEach(entry => {
        entry.results.forEach(result => {
	    //console.log(`  result.ncList ${result.ncList}`);
            ClueManager.primaryNcListToNameSrcLists(result.ncList).forEach(primaryNameSrcList => {
		//console.log(`  nameSrcList: ${primaryNameSrcList}`);
		let source = { primaryNameSrcList };
		source.ncList = result.ncList;
		source.primarySrcList = NameCount.makeCountList(source.primaryNameSrcList);
		source.srcNcLists = result.resultMap ? buildSrcNcLists(result.resultMap.map()) : [ result.ncList ];
		if (logging > 3) {
                    console.log(`result ncList ${source.ncList}, srcNcLists ${showNcLists(source.srcNcLists)}`);
                    if (_.isEmpty(source.srcNcLists)) console.log(`empty srcNcList: ${Stringify(result.resultMap.map())}`);
		}
		sources.push(source);
	    });
        });
    });
    return sources;
}

function showNcLists (ncLists) {
    let str = "";
    let first = true;
    for (let ncList of ncLists) {
        if (!first) str += ' - ';
        str += ncList;
        first = false;
    }
    return _.isEmpty(str) ? "[]" : str;
}

// TODO : anywhere i see primaryNameSrcList I need to change it to primarySrcLists


function mergeSources (sources1, sources2) {
    let mergedSources = {};
    mergedSources.ncList = _.concat(sources1.ncList, sources2.ncList); // TODO: _uniqBy(, _.toString)
    mergedSources.primaryNameSrcList = _.concat(sources1.primaryNameSrcList, sources2.primaryNameSrcList);// TODO: _uniqBy(, _.toString)
    mergedSources.primarySrcList = NameCount.makeCountList(mergedSources.primaryNameSrcList);// TODO: _uniqBy(, ['count'])
    // move to getSourcesLists
    let srcNcLists1 = sources1.srcNcLists;
    let srcNcLists2 = sources2.srcNcLists;
    if (logging>3) console.log(`srcNcLists1: ${showNcLists(srcNcLists1)}`);
    if (logging>3) console.log(`srcNcLists2: ${showNcLists(srcNcLists2)}`);
    mergedSources.srcNcLists = _.concat(srcNcLists1, srcNcLists2);// TODO: _uniqBy(, _.toString)? maybe not necessary here
    if (logging>2) console.log(`  merged: ${showNcLists(mergedSources.srcNcLists)}`);

    return mergedSources;
}

function mergeCompatibleSourcesLists (sources1, sources2) { // TODO sourcesList1, sourcesList2
    let mergedSources = []; // TODO mergedSourcesList
    for (const entry1 of sources1) { // TODO sources1 of sourcesList1
        for (const entry2 of sources2) { // TODO sources2 of sourcesList2
            //Debug(`merging nameSrcList1: ${entry1.primaryNameSrcList}, nameSrcList2: ${entry2.primaryNameSrcList}`);
            if (logging>2) console.log(`mergeCompat: nameSrcList1: ${entry1.primaryNameSrcList}, nameSrcList2: ${entry2.primaryNameSrcList}`);
            if (_.isEmpty(_.intersectionBy(entry1.primarySrcList, entry2.primarySrcList, _.toNumber))) {
		let merged = mergeSources(entry1, entry2);
		//console.log(`  merged nameSrcList: ${merged.primaryNameSrcList}`);
		mergedSources.push(merged);
            }
        }
    }
    return mergedSources;
}

function mergeAllCompatibleSources (ncList) {
    Expect(ncList.length).is.above(0);
    let sources = getSourcesList(ClueManager.getKnownSourceMapEntries(ncList[0]));
    for (let ncIndex = 1; ncIndex < ncList.length; ncIndex += 1) {
        const nextSources = getSourcesList(ClueManager.getKnownSourceMapEntries(ncList[ncIndex]));
	//console.log(`${ncIndex}: merging ${sources[0] ? sources[0].ncList : 'nothing?'} with ${ncList[ncIndex]}`);
        sources = mergeCompatibleSourcesLists(sources, nextSources);
        if (_.isEmpty(sources)) {
	    //console.log('  merge failed');
	    break;	    
	}
	//console.log(`  merge success`);
    }
    //console.log(`  empty before: ${_.isEmpty(sources)}`);
    return sources;
}

//{
// A:
//  'jack:3': {
//    'card:2': {
// B:
//      'bird:1,red:1': [   // multiple primary sources with array value type, split them
//        'bird:2,red:8'
//      ]
//    },
//    'face:1': {
// C:
//      'face:1': [         // single primary source with array value type, ignore
//        'face:10'
//      ]
//    }
//  }
//}
//
// {
// D:
//   'face:1': [              // single top-level primary source with array value type, allow
//     'face:10'
//   ]
// }


function recursiveAddSrcNcLists (list, obj, top) {
    // TODO This is broken for top-level primary sources as above

    let keys = _.flatMap(_.keys(obj), key => {
        let val = obj[key];
        if (_.isObject(val)) {
	    // A: non-array object value type
            if (!_.isArray(val)) return key;
            // split multiple primary sources into separate keys
            let multiplePrimarySourceKeys = key.split(',');
	    // B: comma separated key with array value type: split; TODO assert primary?
            if (multiplePrimarySourceKeys.length > 1) return multiplePrimarySourceKeys;
            // D: single top-level key with array value type: allow; TODO assert primary?
            if (top) return key;
            // C: single nested key with array value type: ignore; TODO assert primary?
        }
        return [];
    });
    if (!_.isEmpty(keys)) {
        list.push(keys);
        keys.forEach(key => {
            if (obj[key]) recursiveAddSrcNcLists(list, obj[key], false);
        });
    }
    return list;
}

function buildSrcNcLists (obj) {
    return recursiveAddSrcNcLists([], obj, true);
}

function partialMatchAnyNcList (ncList, matchNcLists) {
    if (logging > 3) console.log(`  ncList: ${ncList}`);
    for (const matchNcList of matchNcLists) {
        const matchLength = _.intersectionBy(ncList, matchNcList, _.toString).length;
        if (logging>3) console.log(`    matchNcList: ${matchNcList} len: ${matchLength}`);
	if (matchLength === ncList.length) return true;
    }
    return false;
}

//
// useNcLists:
//
//
// useNcList:
//
//
// sourcesList, list of:
//   ncList    (primary NC list)
//   resultMap
//   primaryNameSrcLists = list of all possible primary NameSrcLists which ncList resolves to
//   primarySrcLists = list of all SrcLists derived from NameSrcLists
//   srcNcLists = result.resultMap ? buildSrcNcLists(result.resultMap.map()) : [ result.ncList ];
//
// sources:
//   one entry from sources list
//
// useSourcesList:
//
//
// useSources:
//
//

function mergeAllUsedSources (sourcesList, useNcLists, op) {
    for (let useNcList of useNcLists) {
        let mergedSourcesList = [];
	//console.log(`  mergeCompat: ${useNcList}`);
        let useSourcesList = mergeAllCompatibleSources(useNcList);
	//console.log(`  empty after: ${_.isEmpty(useSourcesList)}`);
	// define problem, start with explaining possible solution
	// --or, --and V:N where N>1, we aren't actually checking V:N itself for match,
	// we're only checking the that the sources of V:N match.
	// we could add V:N to useSourcesList
	//
	// in order to match --or V:N and --and V:N, where N>1, we must:
	//  1. match one of V's direct source NcLists with one of the supplied prospective source NcLists.
	//       e.g.: --or robin; we must find red:1,bird:1 in the sourcesList.
	//  2. match V:N itself
	//useSourcesList = _.uniq(useSourcesList.push(...useNcList));
	// we can ignore this error because some useSources entries may be invalid, particularly if the sources
	// were provided without a [:COUNT] were mapped to all possible counts.
        //if (_.isEmpty(useSourcesList)) throw new Error(`sources not compatible: ${useNcList}`);
        for (let useSources of useSourcesList) {
            for (let sources of sourcesList) {
                const numCommonPrimarySources = _.intersectionBy(sources.primarySrcList, useSources.primarySrcList, _.toNumber).length;
                const allCommonPrimarySources = numCommonPrimarySources === useSources.primarySrcList.length;
                const singlePrimaryNc = useNcList.length === 1 && useNcList[0].count === 1;
		
		// the problem here is that i'm not ANDing or XORing with only the original clue combos, but
		// with the accumulation of previously merged used clues
		
                let valid = false;
		if (op !== Op.and) { // or, xor
		    valid = numCommonPrimarySources === 0;
		}
                if (!valid && (op !== Op.xor)) { // or, and
		    if (allCommonPrimarySources && (singlePrimaryNc || partialMatchAnyNcList(useNcList, sources.srcNcLists))) {
                        valid = true;
		    }
                }
                if (valid) {
		    // TODO: i get the feeling that is merging ncList is not working here, doubling up ncList when merging face,card
		    // need to think deeply here
		    if (1 || op === Op.xor) {
			mergedSourcesList.push(mergeSources(sources, useSources));
		    } else {
			mergedSourcesList.push(sources);
		    }
                }
                if (logging>3 || (valid && logging>2)) {
		    console.log(`  valid: ${valid}, useNcList: ${useNcList}, op: ${OpName(op)}`);
		    //### console.log(`    sources:   ${showNcLists(sources.srcNcLists)}, primary: ${sources.primaryNameSrcList}`);
		    //### console.log(`    useNcList: ${useNcList}, primary: ${useSources.primaryNameSrcList}`);
		    console.log(`    allCommon: ${allCommonPrimarySources}, singlePrimaryNc: ${singlePrimaryNc}`);
		}
            }
        }
        sourcesList = mergedSourcesList;
    }
    if (logging>3) console.log(`  mergeUsed, op: ${OpName(op)}, count: ${sourcesList.length}`);
    return sourcesList;
}

function applyUseNcListOperators (sourcesList, args, nameList) {
    // XOR first
    for (let xorNcLists of args.allXorNcLists) {
	let xorSources = sourcesList;
	if (!_.isEmpty(xorNcLists)) {
            xorSources = mergeAllUsedSources(xorSources, xorNcLists, Op.xor);
            if (logging) console.log(`  compatible with XOR: ${!_.isEmpty(xorSources)}, ${nameList}`);
	    if (_.isEmpty(xorSources)) continue;
	}
	// AND next
	for (let andNcLists of args.allAndNcLists) {
	    let andSources = xorSources;
	    if (!_.isEmpty(andNcLists)) {
		andSources = mergeAllUsedSources(andSources, andNcLists, Op.and);
		if (logging) console.log(`  compatible with AND: ${!_.isEmpty(andSources)}, ${nameList}`);
		if (_.isEmpty(andSources)) continue;
	    }
	    // OR last
	    for (let orNcLists of args.allOrNcLists) {
		let orSources = andSources;
		if (!_.isEmpty(orNcLists)) {
		    let loggy = 0||logging ; //  && !_.isUndefined(_.find(orNcLists.ncList, (o) =>  o.name === 'eye'));
		    let mergedOrSources = mergeAllUsedSources(orSources, orNcLists, Op.or);
		    if (!_.isEmpty(mergedOrSources)) {
			if (loggy) console.log(`  before OR sources: ${Stringify(orSources)}`);
			if (loggy) console.log(`  after OR sources: ${Stringify(mergedOrSources)}`);
			if (loggy) console.log(`  compatible with OR: ${!_.isEmpty(mergedOrSources)}, ${nameList}`);
		    }
		    if (_.isEmpty(mergedOrSources)) continue;
		}
		return true;
	    }
	}
    }
    return false;
}

ComboMaker.prototype.getCombosForUseNcLists = function(args, options = {}) {
    let combos = [];

    // for each sourceList in sourceListArray
    ClueManager.getClueSourceListArray({
        sum: args.sum,
        max: args.max
    }).forEach(clueSourceList => {
        //Debug(`clueSourceList: ${Stringify(clueSourceList)}`);
        let sourceIndexes = [];

        let result = this.first(clueSourceList, sourceIndexes);
        if (result.done) return; // continue; 

        // this is effectively Peco.getCombinations().forEach()
        let first = true;
        while (!result.done) {
            if (!first) {
                result = this.next(clueSourceList, sourceIndexes, options);
                if (result.done) break;
            } else {
                first = false;
            }

            Log.info(`result.nameList: ${result.nameList}`);
            Log.info(`result.ncList: ${result.ncList}`);

            //logging = result.nameList.toString() === 'dark,wood';
            //          || result.nameList.toString() === 'cardinal,smith';

            let sources = mergeAllCompatibleSources(result.ncList);
            
            if (logging) console.log(`  found compatible sources: ${!_.isEmpty(sources)}`);

            // failed to find any compatible combos
            if (_.isEmpty(sources)) continue;

	    if (applyUseNcListOperators(sources, args, result.nameList)) {
		combos.push(result.nameList.toString());
	    }
        }
    }, this);

    return combos;
};

//
// args:
//  count:   # of primary clues to combine
//  max:     max # of sources to use
//  use:     list of clue names and name:counts, also allowing pairs, e.g. ['john:1','bob','red,bird']
//  // not supported: require: required clue counts, e.g. [3,5,8]
//  // not supported: limit to these primary sources, e.g. [1,9,14]
//
// A "clueSourceList" is a list (array) where each element is a
// object that contains a list (cluelist) and a count, such as
// [ { list:clues1, count:1 },{ list:clues2, count:2 }].
//
ComboMaker.prototype.makeCombos = function(args, options = {}) {
    this.nextDupeClue = 0;
    this.nextDupeSrc = 0;
    this.nextDupeCombo = 0;

    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
    }

    // TODO USE "validateArgs" 

    let require = args.require ? _.clone(args.require) : [];
    if (!_.isEmpty(args.require)) throw new Error('require not yet supported');
    if (args.sources) throw new Error('sources not yet supported');

    this.hash = {};
    let allCombos = [];

    let comboArgs = {
        sum: args.sum,
        max: args.max,
        allXorNcLists: args.xor ? buildAllUseNcLists(args.xor) : [ [] ],
        allAndNcLists: args.and ? buildAllUseNcLists(args.and) : [ [] ],
        allOrNcLists: args.or ? buildAllUseNcLists(args.or) : [ [] ]
    };
    

    let combos = this.getCombosForUseNcLists(comboArgs, options);
    allCombos.push(...combos);

    Debug(`dupeClue(${this.nextDupeClue})` +
          `, dupeSrc(${this.nextDupeSrc})` +
          `, dupeCombo(${this.nextDupeCombo})`);

    return allCombos;
};

// As long as one final result has only primary sources from 'sources'
// array, we're good.

ComboMaker.prototype.checkPrimarySources = function(resultList, sources) {
    return resultList.some(result => {
        return NameCount.makeCountList(result.nameSrcList)
            .every(source => {
                return _.includes(sources, source);
            });
    });
};

function getKnownNcListForName (name) {
    const countList = ClueManager.getCountListForName(name);
    if (_.isEmpty(countList)) throw new Error(`not a valid clue name: '${name}'`);
    return countList.map(count => NameCount.makeNew(name, count));
}

//
// Given a list of names and/or ncStrs, convert ncStrs to an array of (1) NC
// and convert names to an array of all known NCs for that name.
// Return a list of ncLists.
//
// ex:
//  convert: [ 'billy', 'bob:1' ]
//  to: [ [ billy:1, billy:2 ], [ bob:1 ] ]
//

// ..ToListOfKnownNcLists

function nameOrNcStrListToKnownNcList (nameOrNcStrList) {
    return nameOrNcStrList.map(nameOrNcStr => NameCount.makeNew(nameOrNcStr))
        .map(nc => nc.count ? [ nc ] : getKnownNcListForName(nc.name));
}

function combinationNcList (combo, ncLists) {
    return combo.map((ncIndex, listIndex) => ncLists[listIndex][ncIndex]);
}

function ncListsToCombinations (ncLists) {
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]),       // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)
    }).getCombinations()
	.map(combo => combinationNcList(combo, ncLists));
}

function getCombinationNcLists (useArgsList) {
    return useArgsList.map(useArg => useArg.split(','))
        .map(nameOrNcStrList => nameOrNcStrListToKnownNcList(nameOrNcStrList))
        .map(knownNcLists => ncListsToCombinations(knownNcLists));
}

// This is the exact same method as ncListsToCombinations?
function combinationsToNcLists (combinationNcLists) {
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]),
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)       // sum of lengths of nclists
    }).getCombinations()
	.map(combo => combinationNcList(combo, combinationNcLists));
}

function buildAllUseNcLists (useArgsList) {
    return combinationsToNcLists(getCombinationNcLists(useArgsList));
}

//

function buildUseNcLists (useArgsList) {
    let useNcLists = [];
    useArgsList.forEach(useArg =>  {
        let args = useArg.split(',');
        let ncList = args.map(arg => {
            let nc = NameCount.makeNew(arg);
            if (!nc.count) throw new Error(`arg: ${arg} requires a :COUNT`);
            if (!_.has(ClueManager.knownClueMapArray[nc.count], nc.name)) throw new Error(`arg: ${nc} does not exist`);
            return nc;
        });
        useNcLists.push(ncList);
    });
    return useNcLists;
}

//
//
ComboMaker.prototype.hasUniqueClues = function(clueList) {
    let sourceMap = {};
    for (let clue of clueList) {
        if (isNaN(clue.count)) {
            throw new Error('bad clue count');
        }
        else if (clue.count > 1) {
            // nothing?
        }
        else if (!this.testSetKey(sourceMap, clue.src)) {
            return false; // forEach.continue... ..why?
        }
    }
    return true;
};

//

ComboMaker.prototype.testSetKey = function(map, key, value = true) {
    if (_.has(map, key)) return false;
    map[key] = value;
    return true;
};

//

ComboMaker.prototype.displaySourceListArray = function(sourceListArray) {
    console.log('-----\n');
    sourceListArray.forEach(function(sourceList) {
        sourceList.forEach(function(source) {
            source.display();
            console.log('');
        });
        console.log('-----\n');
    });
};

//

ComboMaker.prototype.first = function(clueSourceList, sourceIndexes, options = {}) {
    for (let index = 0; index < clueSourceList.length; ++index) {
        sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;
    return this.next(clueSourceList, sourceIndexes, options);
};

//

ComboMaker.prototype.next = function(clueSourceList, sourceIndexes, options = {}) {
    for (;;) {
        if (!this.nextIndex(clueSourceList, sourceIndexes, options)) {
            return { done: true };
        }
        let ncList = [];          // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
        let nameList = [];        // e.g. [ "pollock", "jackson" ]
        let srcCountStrList = []; // e.g. [ "white,fish:2", "moon,walker:4" ]
        if (!clueSourceList.every((clueSource, index) => {
            let clue = clueSource.list[sourceIndexes[index]];
            if (clue.ignore || clue.skip) {
                return false; // every.exit
            }
            nameList.push(clue.name);
            // I think this is right
            ncList.push(NameCount.makeNew(clue.name, clueSource.count));
            srcCountStrList.push(NameCount.makeCanonicalName(clue.src, clueSource.count));
            return true; // every.continue;
        })) {
            continue;
        }

        nameList.sort();
        // skip combinations we've already checked
        let skip = false;

        if (skip && !this.addComboToFoundHash(nameList.toString())) continue; // already checked

        // skip combinations that have duplicate source:count
        if (!options.allow_dupe_src) {
            if (skip && _.uniq(srcCountStrList).length !== srcCountStrList.length) {
                //Debug('skipping duplicate clue src: ' + srcCountStrList);
                ++this.nextDupeSrc;
                continue;
            }
        }

        // skip combinations that have duplicate names
        if (skip && _.sortedUniq(nameList).length !== nameList.length) {
            //Debug('skipping duplicate clue name: ' + nameList);
            ++this.nextDupeClue; // TODO: DupeName
            continue;
        }

        return {
            done:     false,
            ncList:   ncList.sort(),
            nameList: nameList
        };
    }
};

//
//
ComboMaker.prototype.addComboToFoundHash = function(nameListCsv) {
    if (this.testSetKey(this.hash, nameListCsv)) {
        this.hash[nameListCsv] = true;
        return true;
    }
    //Debug('skipping duplicate combo: ' + nameListCsv);
    this.nextDupeCombo += 1;
    return false;
};

//
//
ComboMaker.prototype.nextIndex = function(clueSourceList, sourceIndexes) {
    let index = sourceIndexes.length - 1;

    // increment last index
    ++sourceIndexes[index];

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    while (sourceIndexes[index] === clueSourceList[index].list.length) {
        sourceIndexes[index] = 0;
        index -= 1;
        if (index < 0) {
            return false;
        }
        sourceIndexes[index] += 1;
    }
    return true;
};

//
//
ComboMaker.prototype.displayCombos = function(clueListArray) {
    console.log('\n-----\n');
    let count = 0;
    clueListArray.forEach(function(clueList) {
        clueList.display();
        ++count;
    });
    console.log('total = ' + count);
};

//
//
ComboMaker.prototype.clueListToString = function(clueList) {
    let str = '';
    clueList.forEach(function(clue) {
        if (str.length > 0) {
            str += ' ';
        }
        str += clue.name;
        if (clue.src) {
            str += ':' + clue.src;
        }
    });
    return str;
};

