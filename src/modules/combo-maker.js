//
// combo-maker.js
//

'use strict';

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
const Stringify2  = require('stringify-object');
const Validator   = require('./validator');
const stringify = require('javascript-stringify').stringify;

function Stringify(val) {
    return stringify(val, (value, indent, stringify) => {
	if (typeof value == 'function') return "function";
	return stringify(value);
    }, " ");
}

let logging = 0;
let loggy = false;

const Op = { 'and':1, 'or':2, 'xor':3 };
Object.freeze(Op);

function OpName (opValue) {
    return _.findKey(Op, (v) => opValue === v);
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

function recursiveAddSrcNcLists (list, resultMap, top) {
    let keys = _.flatMap(_.keys(resultMap), key => {
        let val = resultMap[key];
        if (_.isObject(val)) {
            // A: non-array object value type: allow
            if (!_.isArray(val)) return key;
            // split multiple primary sources into separate keys
            let splitKeys = key.split(',');
            // B: comma separated key with array value type: split; TODO assert primary?
            if (splitKeys.length > 1) return splitKeys;
            // D: single top-level key with array value type: allow; TODO assert primary?
            if (top) { if (loggy) console.log(`D: ${key}`); return key; }
            // C: single nested key with array value type: ignore; TODO assert primary?
        }
        if (loggy) console.log(`F: ${key}`);
        return [];
    });
    if (loggy) console.log(keys);
    if (!_.isEmpty(keys)) {
        list.push(keys);
        keys.forEach(key => {
            let val = resultMap[key];
            if (val && !_.isArray(val)) {
		recursiveAddSrcNcLists(list, val, false);
	    }
        });
    }
    return list;
}

function buildSrcNcLists (resultMap) {
    return recursiveAddSrcNcLists([], resultMap, true);
}

function getSourcesList (nc) {
    const sources = [];
    ClueManager.getKnownSourceMapEntries(nc).forEach(entry => {
        entry.results.forEach(result => {
            ClueManager.primaryNcListToNameSrcLists(result.ncList).forEach(primaryNameSrcList => {
		let ncStr = nc.toString();
		if (0 && (ncStr == 'berry:2')) loggy = true;
                let source = {
                    ncList: result.ncList,
		    primaryNameSrcList,
                    srcNcLists: result.resultMap ? buildSrcNcLists(result.resultMap.map()) : [result.ncList]
		};
		if (NameCount.count(nc) > 1) {
		    // TODO: for completeness' sake, i suppose I should find all peers (same count) of 'nc' that have same primary sources.
		    // Achievable by just looking at resultMap?
		    if (loggy) console.log(`$(ncStr): srcNcLists: ${Stringify2(source.srcNcLists)}\n,resultMap: ${Stringify2(result.resultMap.map())}`);
                    source.srcNcLists.push([ncStr]);
		}
                if (loggy || logging > 3) {
                    console.log(`getSourcesList() ncList: ${source.ncList}, srcNcLists: ${showNcLists(source.srcNcLists)}`);
                    if (_.isEmpty(source.srcNcLists)) console.log(`empty srcNcList: ${Stringify(result.resultMap.map())}`);
                }
		loggy = false;
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

let timing = 0;
let one = 0;
let two = 0;
let three =  0;
let four =  0;

// called in inner (inner inner) loops
function mergeSources (sources1, sources2, mergedPrimaryNameSrcList, prefix = '') {
    let mergedSources = {};
    mergedSources.ncList = [...sources1.ncList, ...sources2.ncList];
    mergedSources.primaryNameSrcList = mergedPrimaryNameSrcList
	? mergedPrimaryNameSrcList
	: [...sources1.primaryNameSrcList, ...sources2.primaryNameSrcList];
    // move to getSourcesLists
    const srcNcLists1 = sources1.srcNcLists;
    const srcNcLists2 = sources2.srcNcLists;
    //if (logging>3) console.log(`${prefix} srcNcLists1: ${showNcLists(srcNcLists1)}`);
    //if (logging>3) console.log(`${prefix} srcNcLists2: ${showNcLists(srcNcLists2)}`);
    mergedSources.srcNcLists = [...srcNcLists1, ...srcNcLists2]; // TODO: _uniqBy(, _.toString)? maybe not necessary here
    //if (logging>2) console.log(`  ${prefix} merged: ${showNcLists(mergedSources.srcNcLists)}`);
    return mergedSources;
}

// make primaryNameSrcList a set, created in clue-manager
// instead of intersectionBy, use something like this:
//
// TODO: (equal)subsetOrDistinct
//
function setsEqualOrDistinct(set1, set2, prefix = '') {
    let equal = true;
    let distinct = true;

    let arr1 = [...set1].map(key => key.toString());
    let arr2 = [...set2].map(key => key.toString());
    //
    // Really nice to see this output.  i think there's an optimization here to eliminate calls
    // with set combinations where either set's total count > N in -cM,N
    //
    // console.log(`arr2: ${arr2}`);
    //
    let it = 'red:1';
    let hasit1 = set1.has(it); // _.includes(arr1, it);
    let hasit2 = set2.has(it); // _.includes(arr2, it);
    
    if ((hasit1 || hasit2) && logging) {
	console.log(`${prefix} - ${arr1}.has(it) = ${hasit1}, ${arr2}.has(it) = ${hasit2}`);
    }

    if (set1.size !== set2.size) equal = false;
    let eod = set1.forEach(nameSrc => {
	//console.log(`  ns: ${nameSrc}`);
	if (set2.has(nameSrc)) {
	    distinct = false;
	} else {
	    equal = false;
	}
	//return equal || distinct;
    });
    if (logging && (hasit1 || hasit2)) {
	console.log(`  eod: ${eod} equal: ${equal} distinct: ${distinct}`);
    }
    return { equal, distinct };
}

function mergeCompatibleSourcesLists (sources1, sources2, prefix = '') { // TODO sourcesList1, sourcesList2
    let mergedSources = []; // TODO mergedSourcesList
    for (const entry1 of sources1) { // TODO sources1 of sourcesList1
        for (const entry2 of sources2) { // TODO sources2 of sourcesList2
            if (0 || logging>2) console.log(`mergeCompat: nameSrcList1: ${entry1.primaryNameSrcList}, nameSrcList2: ${entry2.primaryNameSrcList}`);
	    
	    // Same "partial match" trap exists here as in mergeAllUsedSources below.
	    // Here, we are merging multiple comma-separated sources.
	    // Example --or pink --or card.
 	    // red:1 is common, and duplicate so should be allowed, but I believe it won't be.
	    // Also, out of curiousity, do reults for --or card include robin?

	    //const { equal, distinct } = setsEqualOrDistinct(entry1.primaryNameSrcSet, entry2.primaryNameSrcSet, prefix + '-mCSL');
	    //if (distinct) {
	    // if (logging) console.log(`  distinct: ${[...entry1.primaryNameSrcSet]}, ${[...entry2.primaryNameSrcSet]}`);

	    //let allUnique = _.isEmpty(_.intersectionBy(entry1.primaryNameSrcList, entry2.primaryNameSrcList, NameCount.count);
	    //const uniq = _.uniqBy([...entry1.primaryNameSrcList, ...entry2.primaryNameSrcList], NameCount.count);
	    //let allUnique = uniq.length === entry1.primaryNameSrcList.length + entry2.primaryNameSrcList.length;
	    const allUnique = allCountUnique(entry1.primaryNameSrcList, entry2.primaryNameSrcList);
	    if (allUnique) {
                mergedSources.push(mergeSources(entry1, entry2, false,/*combined,*/ prefix + '-mCSL'));
            }
        }
    }
    return mergedSources;
}

// see: showNcLists
let listOfNcListsToString = (listOfNcLists) => {
    if (!listOfNcLists) return _.toString(listOfNcLists);
    let result = "";
    listOfNcLists.forEach((ncList, index) => {
	if (index > 0) result += ' - ';
	result += NameCount.listToString(ncList);
    });
    return result;
};

let stringifySources = (sources) => {
    let result = "[\n";
    let first = true;
    for (let source of sources) {
	if (!first) result += ',\n';
	else first = false;
	result += '  {\n';
	result += `    ncList: ${NameCount.listToString(source.ncList)}\n`;
	result += `    primaryNameSrcList: ${NameCount.listToString(source.primaryNameSrcList)}\n`;
	result += `    srcNcLists: ${listOfNcListsToString(source.srcNcLists)}\n`;
	result += '  }';
    }
    return result + "\n]";
};

function mergeAllCompatibleSources (ncList, prefix = "") {
    Expect(ncList.length).is.above(0).and.below(3); // because broken for > 2 below
    //console.log(Stringify(ncList));
    let sources = getSourcesList(ncList[0]);
    loggy = 0 && ncList[0].name == "wood";
    if (loggy) {
	console.log('*******************MERGE ACS***********************');
	console.log(`** ncList: ${ncList}`);
	console.log('***************************************************');
	console.log(`** sources:\n${stringifySources(sources)}`);
    }
    loggy = false;
    for (let ncIndex = 1; ncIndex < ncList.length; ncIndex += 1) {
        const nextSources = getSourcesList(ncList[ncIndex]);
	// already timed, 30ms
        sources = mergeCompatibleSourcesLists(sources, nextSources, prefix + '-mACS');
	if (0) {
	    console.log(`** merging index: ${ncIndex}, ${ncList[ncIndex]} as nextSources:`);
	    console.log(`${stringifySources(nextSources)}`);
	    console.log(`** result:\n${stringifySources(sources)}`);
	}
        if (_.isEmpty(sources)) break; // this is broken for > 2; should be something like: if (sources.length !== ncIndex + 1) 
    }
    return sources;
}

function matchAnyNcList (ncList, matchNcLists) {
    for (const matchNcList of matchNcLists) {
        const matchLength = _.intersectionBy(ncList, matchNcList, _.toString).length;
        if (matchLength === ncList.length) return true;
    }
    return false;
}

function allCountUnique (nameSrcList1, nameSrcList2) {
    // Uh. use a Set? This is called from within an inner loop.
    let hash = {};
    for (let nameSrc of nameSrcList1) {
	hash[nameSrc.count] = true;
    }
    for (let nameSrc of nameSrcList2) {
	if (hash[nameSrc.count] === true) return false;
    }
    return true;
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
//   srcNcLists = result.resultMap ? buildSrcNcLists(result.resultMap.map()) : [result.ncList];
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
function mergeAllUsedSources (sourcesList, useNcDataList, op) {
    for (let useNcData of useNcDataList) {
        let mergedSourcesList = [];

	if (!useNcData.sourcesList) {
            useNcData.sourcesList = mergeAllCompatibleSources(useNcData.ncList, 'mAUS');
	}
        for (let useSources of useNcData.sourcesList) {
            for (let sources of sourcesList) {
		const allUnique = allCountUnique(sources.primaryNameSrcList, useSources.primaryNameSrcList);
                const singlePrimaryNc = useNcData.ncList.length === 1 && useNcData.ncList[0].count === 1;

                let valid = false;
                if ((op !== Op.and) && allUnique) { // or, xor
                    mergedSourcesList.push(mergeSources(sources, useSources, false, 'mAUS'));
                    valid = true;
                }
                if (!valid && (op === Op.or)) { // or, (and if:  op !== Op.xor)
		    // this appears to be an optimization.  this will match primary clues which have multiple names but same source #
		    // i.e. synonyms/homonyms/different defintions. probably a better way, but probably rare that this matters.
		    const numCommonPrimaryCount = _.intersectionBy(sources.primaryNameSrcList, useSources.primaryNameSrcList, NameCount.count).length;
		    if (numCommonPrimaryCount === useSources.primaryNameSrcList.length) {
			// technically, there is a possibility here that that --or sources
			// could split across matching/not matching primary sources, in the
			// case where there is a duplicate entry for a particular source
			// component's COUNT value.
			// Example: --or card = red:1,bird:1, matching sources red:1,anotherword:1 
			// Because red:1 is a partial match, I believe we'd fail to merge
			// the remaining source (anotherword:1).
			// I think we fall into the same trap in mergeCompatibleSourcesLists.
			if (singlePrimaryNc || matchAnyNcList(useNcData.ncList, sources.srcNcLists)) {
			    Debug(`--or match: ${useNcData.ncList} with something`);
			    mergedSourcesList.push(sources);
                            valid = true;
			}
		    }
                }
                if (logging>3 || (valid && logging>2)) {
                    console.log(`  valid: ${valid}, useNcList: ${useNcData.ncList}, op: ${OpName(op)}`);
                    console.log(`    sources:   ${showNcLists(sources.srcNcLists)}, primary: ${sources.primaryNameSrcList}`);
                    console.log(`    useNcList: ${useNcData.ncList}, primary: ${useSources.primaryNameSrcList}`);
                    //console.log(`    distinct: ${distinct}, singlePrimaryNc: ${singlePrimaryNc}`);
                }
            }
        }
        sourcesList = mergedSourcesList;
    }
    if (logging>3) console.log(`  mergeUsed, op: ${OpName(op)}, count: ${sourcesList.length}`);
    return sourcesList;
}

// TODO: call once per outer iteration (from clues.js) vs. every iteration
//
let getCompatibleUseNcDataSources = (args) => {
    let hashList = [];
    let sourcesLists = [];
    //console.log(`allxorncdatalists(${args.allXorNcDataLists.length}): ${Stringify2(args.allXorNcDataLists)}`);
    for (let [dataListIndex, xorNcDataList] of args.allXorNcDataLists.entries()) {
	for (let [sourcesListIndex, xorNcData] of xorNcDataList.entries()) {
	    if (!sourcesLists[sourcesListIndex]) sourcesLists.push([]);
	    if (!hashList[sourcesListIndex]) hashList.push({});
	    //console.log(`ncList: ${NameCount.listToString(xorNcData.ncList)}`);
	    let sourcesList = mergeAllCompatibleSources(xorNcData.ncList, 'gCuNcDS');
	    for (let sources of sourcesList) {
		let key = sources.primaryNameSrcList.map(_.toString).sort().toString();
		if (!hashList[sourcesListIndex][key]) {
		    sourcesLists[sourcesListIndex].push(sources);
		    hashList[sourcesListIndex][key] = true;
		}
	    }
	}
    }
    let listArray = sourcesLists.map(sl => [...Array(sl.length).keys()]);
    //console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourcesLists(${sourcesLists.length}): ${Stringify2(sourcesLists)}`);

    let peco = Peco.makeNew({
	listArray,
	max: 99999
    });

    let xorSourcesList = [];
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
	//console.log(`indexList: ${stringify(indexList)}`);
	let primaryNameSrcList = [];
	for (let [listIndex, sourceIndex] of indexList.entries()) {
	    let sources = sourcesLists[listIndex][sourceIndex];
	    //console.log(Stringify2(sources));
	    if (_.isEmpty(primaryNameSrcList)) {
		// or just = sources.pnsl.sort(); ?
		primaryNameSrcList.push(...sources.primaryNameSrcList); // .sort((a, b) => { return a.count - b.count; }));
	    } else {
		let combinedNameSrcList = primaryNameSrcList.concat(sources.primaryNameSrcList);
		if (_.uniqBy(combinedNameSrcList, NameCount.count).length === combinedNameSrcList.length) {
		    primaryNameSrcList = combinedNameSrcList; // .sort
		} else {
		    primaryNameSrcList = [];
		    break;
		}
	    }
	}
	if (!_.isEmpty(primaryNameSrcList)) {
	    xorSourcesList.push({ primaryNameSrcList });
	}
    }

    //console.log(`%% xorSourcesLists(${xorSourcesLists.length}): ${Stringify2(xorSourcesLists)}`);
    return xorSourcesList;
};

//
//
let nextIndex = function(clueSourceList, sourceIndexes) {
    // increment last index
    let index = sourceIndexes.length - 1;
    ++sourceIndexes[index];

    // while last index is maxed reset to zero, increment next-to-last index, etc.
    while (sourceIndexes[index] === clueSourceList[index].list.length) {
        sourceIndexes[index] = 0;
        if (--index < 0) {
            return false;
        }
        ++sourceIndexes[index];
    }
    return true;
};

//
//
let next = function(clueSourceList, sourceIndexes, options = {}) {
    for (;;) {
        if (!nextIndex(clueSourceList, sourceIndexes, options)) {
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

	/*
        // skip combinations we've already checked
        let skip = false;

        if (skip && !addComboToFoundHash(nameList.toString())) continue; // already checked

        // skip combinations that have duplicate source:count
        if (!options.allow_dupe_src) {
            if (skip && _.uniq(srcCountStrList).length !== srcCountStrList.length) {
                continue;
            }
        }
        // skip combinations that have duplicate names
        if (skip && _.sortedUniq(nameList).length !== nameList.length) {
            continue;
        }
	*/

        return {
            done:     false,
            ncList:   ncList.sort(),
            nameList: nameList
        };
    }
};

//
//
let first = function(clueSourceList, sourceIndexes, options = {}) {
    for (let index = 0; index < clueSourceList.length; ++index) {
        sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;
    return next(clueSourceList, sourceIndexes, options);
};

/*
//
//
let addComboToFoundHash = function(nameListCsv) {
    if (testSetKey(hash, nameListCsv)) {
        hash[nameListCsv] = true;
        return true;
    }
    //Debug('skipping duplicate combo: ' + nameListCsv);
    nextDupeCombo += 1;
    return false;
};
*/

let isCompatibleWithUseSources = (sources, useSources) => {
    for (let source of sources) {
	for (let useSource of useSources) {
	    const allUnique = allCountUnique(source.primaryNameSrcList, useSource.primaryNameSrcList);
	    if (allUnique) return true;
	}
    }
    return false;
};

//
// NEW NEW NEW
//
let getCombosForUseNcLists = function(args, options = {}) {
    let hash = {};
    let combos = [];

    let comboCount = 0;
    let totalVariationCount = 0;
    let cacheHitCount = 0;
    let numIncompatible = 0;
    
    let MILLY = 1000000n;
    let start = process.hrtime.bigint();

    let useSources = args.useSources;
    if (0) console.log(`compatibleUseNcDataSources: ${Stringify2(useSources)}`);

    // for each sourceList in sourceListArray
    ClueManager.getClueSourceListArray({
        sum: args.sum,
        max: args.max
    }).forEach(clueSourceList => {
	comboCount += 1;
        let sourceIndexes = [];

	//console.log(`sum(${args.sum}) max(${args.max}) clueSrcList: ${Stringify(clueSourceList)}`);

        let result = first(clueSourceList, sourceIndexes);
        if (result.done) return; // continue; 

	let variationCount = 1;

        // this is effectively Peco.getCombinations().forEach()
	let firstIter = true;
        while (!result.done) {
            if (!firstIter) {
		// TODO problem 1:
		// problem1: why is this (apparently) considering the first two entries of the same
		// clue count (e.g. red, red). It doesn't matter when the clue counts are different,
		// but when they're the same, we're wasting time. Is there some way to determine if
		// the two lists are equal at time of get'ing (getClueSourceListArray) such that
		// we could optimize this.next for this condition?
		// timed; 58s in 2
                result = next(clueSourceList, sourceIndexes, options);
                if (result.done) break;
		variationCount += 1;
            } else {
		firstIter = false;
            }
            //console.log(`result.nameList: ${result.nameList}`);
            //console.log(`result.ncList: ${result.ncList}`);

	    // TODO problem 2:
	    // wouldn't it (generally) be (a lot) faster to check for UseNcList compatability before
	    // merging all compatible sources? (We'd have to do it again after merging, presumably).

	    const key = NameCount.listToString(result.ncList);
	    let sources;
	    // TOOD: removing hash would be nice. filter "list" for uniqueness. (requires we have a list first)
	    if (!hash[key]) {
		sources = mergeAllCompatibleSources(result.ncList, 'c4UNCL');
		//console.log(`$$ sources: ${Stringify2(sources)}`);
		hash[key] = { sources };
	    } else {
		sources = hash[key].sources;
		cacheHitCount += 1;
	    }
	    logging = 0;

            if (logging) console.log(`  found compatible sources: ${!_.isEmpty(sources)}`);

            // failed to find any compatible combos
            if (_.isEmpty(sources)) continue;

	    if (_.isUndefined(hash[key].isUseNcCompatible)) {
		hash[key].isUseNcCompatible = isCompatibleWithUseSources(sources, useSources);
	    }
	    if (hash[key].isUseNcCompatible) {
                combos.push(result.nameList.toString());
            } else {
		++numIncompatible;
	    }
        }
	totalVariationCount += variationCount;
    }, this);

    let duration = (process.hrtime.bigint() - start) / MILLY;

    Debug(`combos(${comboCount}) variations(${totalVariationCount}) cacheHits(${cacheHitCount}) incompatible(${numIncompatible}) ` +
	  `actual(${totalVariationCount - cacheHitCount - numIncompatible})`);
    console.error(`combos(${comboCount}) variations(${totalVariationCount}) cacheHits(${cacheHitCount}) incompatible(${numIncompatible}) ` +
		  `actual(${totalVariationCount - cacheHitCount - numIncompatible}) ${duration}ms`);
    //process.stderr.write('.');

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
let makeCombosForSum = function(args, options = {}) {
    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
    }

    // TODO USE "validateArgs" 
    let require = args.require ? _.clone(args.require) : [];
    if (!_.isEmpty(args.require)) throw new Error('require not yet supported');
    if (args.sources) throw new Error('sources not yet supported');

    let allCombos = [];

    let comboArgs = {
        sum: args.sum,
        max: args.max,
	allXorNcDataLists: args.allXorNcDataLists,
	allAndNcDataLists: args.allAndNcDataLists,
	allOrNcDataLists: args.allOrNcDataLists,
	useSources: args.useSources
    };
    
    let combos = getCombosForUseNcLists(comboArgs, options);
    allCombos.push(...combos);

    Debug(`timing: ${PrettyMs(timing)}  one: ${PrettyMs(one)} two: ${PrettyMs(two)} three: ${PrettyMs(three)} four: ${PrettyMs(four)}`);
    return allCombos;
};

//
//
let makeCombos = (args, options) => {
    let sumRange;
    if (!_.isUndefined(args.sum)) {
	// is _chain even necessary here?
        sumRange = _.chain(args.sum).split(',').map(_.toNumber).value();
    }
    Expect(sumRange).is.an.Array().with.property('length').below(3); // of.at.most(2);
    Debug('++combos' +
          `, sum: ${sumRange}` +
          `, max: ${args.max}` +
//          `, require: ${args.require}` +
//          `, sources: ${args.sources}` +
          `, use: ${args.use}`);
    
    let total = 0;
    let known = 0;
    let reject = 0;
    let duplicate  = 0;
    let comboMap = {};
    let beginDate = new Date();

    args.allXorNcDataLists = args.xor ? buildAllUseNcDataLists(args.xor) : [ [] ];
    //console.log(`allXorNcDataLists: ${Stringify2(args.allXorNcDataLists)}`);
    args.allAndNcDataLists = args.and ? buildAllUseNcDataLists(args.and) : [ [] ];
    args.allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or) : [ [] ];
    args.useSources = getCompatibleUseNcDataSources(args);

    let lastSum = sumRange.length > 1 ? sumRange[1] : sumRange[0];
    for (let sum = sumRange[0]; sum <= lastSum; ++sum) {
        args.sum = sum;
        let max = args.max;
        if (args.max > args.sum) args.max = args.sum;
        // TODO: return # of combos filtered due to note name match
        const comboList = makeCombosForSum(args, options);
        args.max = max;
        total += comboList.length;
        const filterResult = ClueManager.filter(comboList, args.sum, comboMap);
        known += filterResult.known;
        reject += filterResult.reject;
        duplicate += filterResult.duplicate;
    }
    let d = new Duration(beginDate, new Date()).milliseconds;
    _.keys(comboMap).forEach(nameCsv => console.log(nameCsv));
    //console.log(`${Stringify(comboMap)}`);
    
    //process.stderr.write('\n');

    Debug(`total: ${total}` +
                ', filtered: ' + _.size(comboMap) +
                ', known: ' + known +
                ', reject: ' + reject +
                ', duplicate: ' + duplicate);
    Debug(`--combos: ${PrettyMs(d)}`);

    if (total !== _.size(comboMap) + known + reject + duplicate) {
        Debug('WARNING: amounts to not add up!');
    }
};

// As long as one final result has only primary sources from 'sources'
// array, we're good.

let checkPrimarySources = function(resultList, sources) {
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
        .map(nc => nc.count ? [nc] : getKnownNcListForName(nc.name));
}

function combinationNcList (combo, ncLists) {
    return combo.map((ncIndex, listIndex) => ncLists[listIndex][ncIndex]);
}

function combinationNcDataList (combo, ncLists) {
    return combo.map((ncIndex, listIndex) => Object({ ncList: ncLists[listIndex][ncIndex]}));
}

function ncListsToCombinations (ncLists) {
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]),       // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)
    }).getCombinations()
        .map(combo => combinationNcList(combo, ncLists));
}

function getCombinationNcLists (useArgsList) {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    return useArgsList.map(useArg => useArg.split(','))
        .map(nameOrNcStrList => nameOrNcStrListToKnownNcList(nameOrNcStrList))
        .map(knownNcLists => ncListsToCombinations(knownNcLists));
}

// This is the exact same method as ncListsToCombinations? except for final map method. could pass as parameter.
function combinationsToNcLists (combinationNcLists) {
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]),
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)       // sum of lengths of nclists
    }).getCombinations()
      .map(combo => combinationNcList(combo, combinationNcLists));
}

// TODO: get rid of this and combinationsToNCLists, and add extra map step in buildAllUseNCData
function combinationsToNcDataLists (combinationNcLists) {
    Debug(`combToNcDataLists() combinationNcLists: ${Stringify(combinationNcLists)}`);
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]),
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)       // sum of lengths of nclists
    }).getCombinations()
      .map(combo => combinationNcDataList(combo, combinationNcLists));
}

function buildAllUseNcLists (useArgsList) {
    return combinationsToNcLists(getCombinationNcLists(useArgsList));
}

function buildAllUseNcDataLists (useArgsList) {
    return combinationsToNcDataLists(getCombinationNcLists(useArgsList));
}

//
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
let hasUniqueClues = function(clueList) {
    let sourceMap = {};
    for (let clue of clueList) {
        if (isNaN(clue.count)) {
            throw new Error('bad clue count');
        }
        else if (clue.count > 1) {
            // nothing?
        }
        else if (!testSetKey(sourceMap, clue.src)) {
            return false; // forEach.continue... ..why?
        }
    }
    return true;
};

//
//
let testSetKey = function(map, key, value = true) {
    if (_.has(map, key)) return false;
    map[key] = value;
    return true;
};

//
//
let displaySourceListArray = function(sourceListArray) {
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
//
let displayCombos = function(clueListArray) {
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
let clueListToString = function(clueList) {
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

module.exports = {
    makeCombos
};
