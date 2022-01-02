///
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
const OS          = require('os');
const Parallel    = require('paralleljs');
const Peco        = require('./peco');
const PrettyHrTime = require('pretty-hrtime');
const PrettyMs    = require('pretty-ms');
const ResultMap   = require('../types/result-map');
//const Stringify   = require('stringify-object');
const BootstrapCM = require('./bootstrap-combo-maker');
const Validator   = require('./validator');
const stringify   = require('javascript-stringify').stringify;
//let Stringify = stringify;

function Stringify(val) {
    return stringify(val, (value, indent, stringify) => {
	if (typeof value == 'function') return "function";
	return stringify(value);
    }, " ");
}

let logging = 0;
let mcsl_timing = 0n;
let mcsl_iter = 0;
let ms_iter = 0;
let mcsl_count = 0;
let sl_timing = 0n;
let maus_timing = 0n;
const MILLY = 1000000n;

let usenc_count = 0;
let usenc_size = 0;
let usenc_sources_count = 0;
let usenc_sources_size = 0;
let sources_count = 0;
let sources_size = 0;

const Op = { 'and':1, 'or':2, 'xor':3 };
Object.freeze(Op);

function OpName (opValue) {
    return _.findKey(Op, (v) => opValue === v);
}

//

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

function getSourcesList (nc) {
    const sources = [];
    ClueManager.getKnownSourceMapEntries(nc).forEach(entry => {
        entry.results.forEach(result => {
            ClueManager.primaryNcListToNameSrcLists(result.ncList).forEach(primaryNameSrcList => {
                let source = { primaryNameSrcList };
                source.ncList = result.ncList;
                // should be able to eliminate primarySrcList using _.findBy(primaryNameSrcList, Name.count)
                //source.primarySrcList = NameCount.makeCountList(source.primaryNameSrcList);
                source.srcNcLists = result.resultMap ? buildSrcNcLists(result.resultMap.map()) : [ result.ncList ];
                source.srcNcLists.push([ nc ]); // TODO will this break anything, ? (any other use of mergeSources/checkCompatibleSources)
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

let timing = 0;
let one = 0;
let two = 0;
let three =  0;
let four =  0;

// this is the innermost function of some nasty complexity nested loops.
function mergeSources (sources1, sources2, mergedPrimaryNameSrcList, prefix = '') {
    let mergedSources = {};
    // timed, 133ms in 2
    mergedSources.ncList = [...sources1.ncList,...sources2.ncList];
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

    ms_iter += 1;

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
    /*
    let it = 'red:1';
    let hasit1 = set1.has(it); // _.includes(arr1, it);
    let hasit2 = set2.has(it); // _.includes(arr2, it);
    
    if ((hasit1 || hasit2) && logging) {
	console.log(`${prefix} - ${arr1}.has(it) = ${hasit1}, ${arr2}.has(it) = ${hasit2}`);
    }
    */
    
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
    /*
    if (logging && (hasit1 || hasit2)) {
	console.log(`  eod: ${eod} equal: ${equal} distinct: ${distinct}`);
    }
    */
    return { equal, distinct };
}

function mergeCompatibleSourcesLists (sources1, sources2, prefix = '') { // TODO sourcesList1, sourcesList2
    let mergedSources = []; // TODO mergedSourcesList
    for (const entry1 of sources1) { // TODO sources1 of sourcesList1
        for (const entry2 of sources2) { // TODO sources2 of sourcesList2
            //if (logging>2) console.log(`mergeCompat: nameSrcList1: ${entry1.primaryNameSrcList}, nameSrcList2: ${entry2.primaryNameSrcList}`);
	    
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

function mergeAllCompatibleSources (ncList, prefix = "") {
    Expect(ncList.length).is.above(0);
    //console.log(Stringify(ncList));
    let sources = getSourcesList(ncList[0]);
    for (let ncIndex = 1; ncIndex < ncList.length; ncIndex += 1) {
        const nextSources = getSourcesList(ncList[ncIndex]);
	// already timed, 30ms
        sources = mergeCompatibleSourcesLists(sources, nextSources, prefix + '-mACS');
        if (_.isEmpty(sources)) break;
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

function mergeAllUsedSources (sourcesList, useNcDataList, op, options) {
    for (let useNcData of useNcDataList) {
	if (!useNcData.sourcesList) {
            useNcData.sourcesList = mergeAllCompatibleSources(useNcData.ncList, 'mAUS');
	}
    }

    usenc_count +=1;
    usenc_size += useNcDataList.length;
    let maus_start = process.hrtime.bigint();

    for (let useNcData of useNcDataList) {
        let mergedSourcesList = [];

	/*
	if (!useNcData.sourcesList) {
	    let start = new Date();
            useNcData.sourcesList = mergeAllCompatibleSources(useNcData.ncList, 'mAUS');
	    timing += new Duration(start, new Date()).milliseconds;
	}
	*/

	usenc_sources_count +=1;
	usenc_sources_size += useNcData.sourcesList.length;

        // we can ignore this error because some useSources entries may be invalid, in particular if the sources
        // that were provided without a [:COUNT] were mapped to all possible counts.
        //if (_.isEmpty(useSourcesList)) throw new Error(`sources not compatible: ${useNcList}`);
        for (let useSources of useNcData.sourcesList) {

	    sources_count +=1;
	    sources_size += sourcesList.length;

            for (let sources of sourcesList) {
		const allUnique = allCountUnique(sources.primaryNameSrcList, useSources.primaryNameSrcList);
                const singlePrimaryNc = useNcData.ncList.length === 1 && useNcData.ncList[0].count === 1;
                
                // the problem here is that i'm not ANDing or XORing with only the original clue combos, but
                // with the accumulation of previously merged used clues
                // (actually i'm not sure that's a problem at all. that might be by design)
                
		// timed; 400ms in 2
                let valid = false;
                if ((op !== Op.and) && allUnique) { // or, xor
                    mergedSourcesList.push(mergeSources(sources, useSources, false,/*combined,*/ 'mAUS'));
                    valid = true;
                }

                if (!valid && (op === Op.or)) { // or, (and if:  op !== Op.xor)
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
                /* don't remove
                if (valid) {
                    // TODO: i get the feeling that is merging ncList is not working here, doubling up ncList when merging face,card
                    //
                    // need to think deeply here. the correct logic is an optimization
                    //
                    if (op === Op.xor || ((op === Op.or) && !allCommonPrimarySources)) {
                        mergedSourcesList.push(mergeSources(sources, useSources));
                    }
                }
                */
		/*
                if (logging>3 || (valid && logging>2)) {
                    console.log(`  valid: ${valid}, useNcList: ${useNcData.ncList}, op: ${OpName(op)}`);
                    console.log(`    sources:   ${showNcLists(sources.srcNcLists)}, primary: ${sources.primaryNameSrcList}`);
                    console.log(`    useNcList: ${useNcData.ncList}, primary: ${useSources.primaryNameSrcList}`);
                    //console.log(`    distinct: ${distinct}, singlePrimaryNc: ${singlePrimaryNc}`);
                }
		*/
            }
        }
        sourcesList = mergedSourcesList;
    }

    let maus_end = process.hrtime.bigint();
    maus_timing += maus_end - maus_start;

    if (logging>3) console.log(`  mergeUsed, op: ${OpName(op)}, count: ${sourcesList.length}`);
    return sourcesList;
}

//

function isCompatibleWithUseNcLists (sourcesList, args, nameList, options) {
    // XOR first
    for (let xorNcDataList of args.allXorNcDataLists) {
        let xorSources = sourcesList;
        if (!_.isEmpty(xorNcDataList)) {
            xorSources = mergeAllUsedSources(xorSources, xorNcDataList, Op.xor, options);
            if (logging) console.log(`  compatible with XOR: ${!_.isEmpty(xorSources)}, ${nameList}`);
            if (_.isEmpty(xorSources)) continue;
        }
        // AND next
        for (let andNcData of args.allAndNcDataLists) {
            let andSources = xorSources;
            if (!_.isEmpty(andNcData)) {
                andSources = mergeAllUsedSources(andSources, andNcData, Op.and, options);
                if (logging) console.log(`  compatible with AND: ${!_.isEmpty(andSources)}, ${nameList}`);
                if (_.isEmpty(andSources)) continue;
            }
            // OR last
            for (let orNcData of args.allOrNcDataLists) {
                let orSources = andSources;
                if (!_.isEmpty(orNcData)) {
                    let mergedOrSources = mergeAllUsedSources(orSources, orNcData, Op.or, options);
                    if (!_.isEmpty(mergedOrSources)) {
                        if (logging>2) console.log(`  before OR sources: ${Stringify(orSources)}`);
                        if (logging>2) console.log(`  after OR sources: ${Stringify(mergedOrSources)}`);
                        if (logging>2) console.log(`  compatible with OR: ${!_.isEmpty(mergedOrSources)}, ${nameList}`);
                    }
                    if (_.isEmpty(mergedOrSources)) continue;
                }
                return true;
            }
        }
    }
    return false;
}

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

//
//
let nextIndex = function(clueSourceList, sourceIndexes) {
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

let next = (clueSourceList, sourceIndexes, options = {}) => {
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
                //Debug('skipping duplicate clue src: ' + srcCountStrList);
                ++nextDupeSrc;
                continue;
            }
        }

        // skip combinations that have duplicate names
        if (skip && _.sortedUniq(nameList).length !== nameList.length) {
            //Debug('skipping duplicate clue name: ' + nameList);
            ++nextDupeClue; // TODO: DupeName
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

let first = (clueSourceList, sourceIndexes, options = {}) => {
    for (let index = 0; index < clueSourceList.length; ++index) {
        sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;
    return next(clueSourceList, sourceIndexes, options);
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

// This is the exact same method as ncListsToCombinations?
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

let testSetKey = function(map, key, value = true) {
    if (_.has(map, key)) return false;
    map[key] = value;
    return true;
};

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

let hash = {};

let getCombosForUseNcLists = function(args, options = {}) {
    let combos = [];

    let comboCount = 0;
    let totalVariationCount = 0;
    let cacheHits = 0;
    
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
	let notfirst = false;
        while (!result.done) {
            if (notfirst) {
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
                notfirst = true;
            }
            //console.log(`result.nameList: ${result.nameList}`);
            //console.log(`result.ncList: ${result.ncList}`);

	    //if (result.ncList.length == 2 && result.ncList[0].name == "red" && result.ncList[1].name == "red") logging = 5;

	    // TODO problem 2:
	    // wouldn't it (generally) be (a lot) faster to check for UseNcList compatability before
	    // merging all compatible sources? (We'd have to do it again after merging, presumably).


	    const strList = NameCount.listToString(result.ncList); // result.ncList.toString();
	    //const strList = _.sortBy(result.ncList, _.toString).toString();
	    let sources;
	    if (!hash[strList]) {
		sources = mergeAllCompatibleSources(result.ncList, 'c4UNCL');
		hash[strList] = { sources };
	    } else {
		++cacheHits;
		sources = hash[strList].sources;
	    }

            // failed to find any compatible combos
            if (_.isEmpty(sources)) continue;

	    if (_.isUndefined(hash[strList].useNcCompatible)) {
		hash[strList].useNcCompatible = isCompatibleWithUseNcLists(sources, args, result.nameList);
	    }
	    if (hash[strList].useNcCompatible) {
                combos.push(result.nameList.toString());
            }
        }
	totalVariationCount += variationCount;
    });

//    Debug(`combos(${comboCount}), variations(${totalVariationCount}), AVG variations/combo(${totalVariationCount/comboCount}), cacheHits(${cacheHits})`);
    console.error(`combos(${comboCount}), variations(${totalVariationCount}), cacheHits(${cacheHits}), merges(${totalVariationCount - cacheHits})`);

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
let makeCombos = function(args, options = {}) {
    // TODO USE "validateArgs" 

    /*
    let require = args.require ? _.clone(args.require) : [];
    if (!_.isEmpty(args.require)) throw new Error('require not yet supported');
    if (args.sources) throw new Error('sources not yet supported');
    */

    //this.hash = {};
    let allCombos = [];

    // TODO: push out parallelization a level, compute these once per process

    let allXorNcDataLists = args.xor ? buildAllUseNcDataLists(args.xor) : [ [] ];
    let allAndNcDataLists = args.and ? buildAllUseNcDataLists(args.and) : [ [] ];
    let allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or) : [ [] ];

    //let allXorNcLists = args.xor ? buildAllUseNcLists(args.xor) : [ [] ];
    //console.log(`allXorNcDataLists: ${Stringify(allXorNcDataLists)}`);
    //console.log(`allXorNcDataLists[0]: ${Stringify(allXorNcDataLists[0])}`);

    let comboArgs = {
        sum: args.sum,
        max: args.max,
	allXorNcDataLists,
	allAndNcDataLists,
	allOrNcDataLists
    };
    
    let combos = getCombosForUseNcLists(comboArgs, options);
    allCombos.push(...combos);

//    console.error(`timing: ${PrettyMs(timing)} mcsl: ${mcsl_timing/MILLY}ms sl: ${sl_timing/MILLY}ms maus: ${maus_timing/MILLY}ms`);
//    console.error(`mcsl iter: ${mcsl_iter} ms iter: ${ms_iter} avg ms/mscl: ${_.toInteger(ms_iter / mcsl_iter)}`);
//    console.error(`usenc count: ${usenc_count} size: ${usenc_size} avg: ${_.toInteger(usenc_size/usenc_count)}`);
//    console.error(`usenc sources count: ${usenc_sources_count} size: ${usenc_sources_size} avg: ${_.toInteger(usenc_sources_size/usenc_sources_count)}`);
//    console.error(`sources count: ${sources_count} size: ${sources_size} avg: ${_.toInteger(sources_size/sources_count)}`);

    return allCombos;
};

let makeCombosForRange = (first, last, args, options) => {
    let range = [...Array(last + 1).keys()].slice(first)
	.map(sum => { return {
	    apple: args.apple,
	    final: args.final,
	    meta:  args.meta,
	    sum,
            max: (args.max > sum) ? sum : args.max,
	    xor: args.xor,
	    and: args.and,
	    or: args.or
	};});

    let cpus = OS.cpus().length;
    let cpus_used = cpus <= 6 ? cpus: cpus / 2;
    console.error(`cpus: ${cpus} used: ${cpus_used}`);
    let p = new Parallel(range, {
	maxWorkers: cpus_used,
    	evalPath: '${__dirname}/../../modules/bootstrap-combo-maker.js'
    });
    let entrypoint = BootstrapCM.entrypoint;
    console.log('makeCombos++');
    let beginDate = new Date();
    p.map(entrypoint).then(data => {
	console.log(`chunks: ${data.length} combos: ${data[0].length}`);
	let d = new Duration(beginDate, new Date()).milliseconds;
	console.error(`--makeCombos: ${PrettyMs(d)}`);
    });
    // TODO: call filter on each data element, return map
    // check if range == data and /or if .then(return) passes thru
    //const filterResult = ClueManager.filter(data[i], args.sum, comboMap);
};

//
//
//
/*
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
    makeCombos,
    makeCombosForRange,
};
