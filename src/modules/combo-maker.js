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
const stringify   = require('javascript-stringify').stringify;
const Stringify2  = require('stringify-object');
const Validator   = require('./validator');

function Stringify(val) {
    return stringify(val, (value, indent, stringify) => {
        if (typeof value == 'function') return "function";
        return stringify(value);
    }, " ");
}

let logging = 0;
let loggy = false;

const Op = {
    'and':1,
    'or':2,
    'xor':3
};
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
                if (0 && (nc == 'bat:3')) loggy = true;
                let source = {
                    ncList: [nc],
                    primaryNameSrcList,
                    srcNcLists: result.resultMap ? buildSrcNcLists(result.resultMap.map()) : [result.ncList]
                };
                if (NameCount.count(nc) > 1) {
                    if (loggy) console.log(`(${nc}): srcNcLists: ${showNcLists(source.srcNcLists)}\n,resultMap: ${Stringify2(result.resultMap.map())}`);
                    // TODO: there is possibly some operator (different than --or) where I should add all peers
		    // (same count) of 'nc' that have same primary sources. Achievable by just looking at resultMap? 
                    source.srcNcLists.push([nc.toString()]);
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

let mergeSources = (sources1, sources2) => {
    let mergedSources = {};
    mergedSources.ncList = [...sources1.ncList, ...sources2.ncList];
    mergedSources.primaryNameSrcList = [...sources1.primaryNameSrcList, ...sources2.primaryNameSrcList];
    mergedSources.srcNcLists = [...sources1.srcNcLists, ...sources2.srcNcLists];
    return mergedSources;
};

//
//
let mergeCompatibleSources = (sources1, sources2) => {
    if (0 || logging>2) console.log(`mergeCompat: nameSrcList1: ${sources1.primaryNameSrcList}, nameSrcList2: ${sources2.primaryNameSrcList}`);
    const allUnique = allCountUnique(sources1.primaryNameSrcList, sources2.primaryNameSrcList);
    // wrap one element in an array to simplify !allUnique failure/null condition at caller site
    return allUnique ? [mergeSources(sources1, sources2, false)] : [];
};

//
//
function mergeCompatibleSourcesLists (sourcesList1, sourcesList2) {
    let mergedSourcesList = [];
    for (const sources1 of sourcesList1) {
        for (const sources2 of sourcesList2) {
            mergedSourcesList.push(...mergeCompatibleSources(sources1, sources2));
        }
    }
    return mergedSourcesList;
}

//
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

//
//
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

//
//
let mergeAllCompatibleSources = (ncList) => {
    Expect(ncList.length).is.above(0).and.below(3); // because broken for > 2 below
    //console.log(Stringify(ncList));
    let sources = getSourcesList(ncList[0]);
    loggy = false;
    if (loggy) {
        console.log('*******************MERGE ACS***********************');
        console.log(`** ncList: ${ncList}`);
        console.log('***************************************************');
        console.log(`** sources:\n${stringifySources(sources)}`);
    }
    for (let ncIndex = 1; ncIndex < ncList.length; ncIndex += 1) {
        const nextSources = getSourcesList(ncList[ncIndex]);
        // already timed, 30ms
        sources = mergeCompatibleSourcesLists(sources, nextSources);
        if (loggy) {
            console.log(`** merging index: ${ncIndex}, ${ncList[ncIndex]} as nextSources:`);
            console.log(`${stringifySources(nextSources)}`);
            console.log(`** result:\n${stringifySources(sources)}`);
        }
        if (_.isEmpty(sources)) break; // this is broken for > 2; should be something like: if (sources.length !== ncIndex + 1) 
    }
    loggy = false;
    return sources;
};

//
//
function matchAnyNcList (ncList, matchNcLists) {
    for (const matchNcList of matchNcLists) {
        const matchCount = _.intersectionBy(ncList, matchNcList, _.toString).length;
        if (matchCount === ncList.length) return true;
    }
    return false;
}

//
//
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

// sourcesList, list of:
//   ncList = NC list
//   primaryNameSrcLists = list of all possible primary NameSrcLists which ncList resolves to
//   srcNcLists = result.resultMap ? buildSrcNcLists(result.resultMap.map()) : [result.ncList];
//
function mergeAllUsedSources (sourcesList, useNcDataList, op) {
    for (let useNcData of useNcDataList) {
        let mergedSourcesList = [];

        if (!useNcData.sourcesList) {
            useNcData.sourcesList = mergeAllCompatibleSources(useNcData.ncList);
        }
        for (let useSources of useNcData.sourcesList) {
            for (let sources of sourcesList) {
                const allUnique = allCountUnique(sources.primaryNameSrcList, useSources.primaryNameSrcList);
                const singlePrimaryNc = useNcData.ncList.length === 1 && useNcData.ncList[0].count === 1;

                let valid = false;
                if ((op !== Op.and) && allUnique) { // or, xor
                    mergedSourcesList.push(mergeSources(sources, useSources, false));
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

//
//
let buildUseSourcesLists = (useNcDataLists) => {
    let sourcesLists = [];
    let hashList = [];
    //console.log(`useNcDataLists(${useNcDataLists.length}): ${Stringify2(useNcDataLists)}`);
    for (let [dataListIndex, useNcDataList] of useNcDataLists.entries()) {
        for (let [sourcesListIndex, useNcData] of useNcDataList.entries()) {
            if (!sourcesLists[sourcesListIndex]) sourcesLists.push([]);
            if (!hashList[sourcesListIndex]) hashList.push({});
            //console.log(`ncList: ${NameCount.listToString(useNcData.ncList)}`);
            let sourcesList = mergeAllCompatibleSources(useNcData.ncList);
            //console.log(`sourcesList(${sourcesList.length}): ${Stringify2(sourcesList)}`);
            for (let sources of sourcesList) {
                let key = sources.primaryNameSrcList.map(_.toString).sort().toString();
                if (!hashList[sourcesListIndex][key]) {
                    sourcesLists[sourcesListIndex].push(sources);
                    hashList[sourcesListIndex][key] = true;
                }
            }
        }
    }
    return sourcesLists;
};

//
//
let mergeCompatibleUseSources = (sourcesLists, op) => {
    ///
    // TODO: so I think i can share a lot of this
    // Pass OpType here, if Xor, same as always, if Or:
    // *X  add 1 to each array;
    // *X  index 0 = continue
    // *X  index sourcesList array by sourceIndex - 1
    // No:? It's looking like Yes Maybe
    // *  call mergeSources on successful combinedNameSrcList
    // *    and add those mergedSources to the final result entry
    // Sorta:? Yah Maybe
    // *  in effect "primaryNameSrcList" now becomes "sources"
    // *    as a result of calling mergeSources (which does the 
    // *    primary merge for us
    //
    // return list of primaryNameSrc+sources lists as always.

    // TODO: sometimes a sourcesList is empty, like if doing $(cat required) with a
    // low clue count range (e.g. -c2,4). should that even be allowed?
    let pad = (op === Op.or) ? 1 : 0;
    let listArray = sourcesLists.map(sl => [...Array(sl.length + pad).keys()]);
    //console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourcesLists(${sourcesLists.length}): ${Stringify2(sourcesLists)}`);

    let peco = Peco.makeNew({
        listArray,
        max: 99999
    });

    // resultList? 
    let iter = 0;
    let sourcesList = [];
    let orSourcesListsArray = [];
//    let last
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
        //console.log(`indexList: ${stringify(indexList)}`);
        let primaryNameSrcList = [];
	//
	// TODO: list of sourcesLists outside of this loop. 
	// assign result.sourceLists inside indexList.entries() loop. 
	//
        let result = {
            primaryNameSrcList: []
        };
        for (let [listIndex, sourceIndex] of indexList.entries()) {
	    if (!orSourcesListsArray[listIndex]) orSourcesListsArray.push([]);
	    result.sourcesLists = orSourcesListsArray[listIndex];
	    //console.log(`iter(${iter}) list(${listIndex}) source(${sourceIndex}), result.sourcesLists(${result.sourcesLists.length})`);
            if (op === Op.or) {
                if (sourceIndex === 0) {
		    //console.log(`  adding orSources: ${Stringify2(sourcesLists[listIndex])}`);
                    result.sourcesLists.push(sourcesLists[listIndex]);
		    //console.log(`  iter(${iter}) list(${listIndex}) source(${sourceIndex}), result.sourcesLists(${result.sourcesLists.length})`);
                    continue;
                }
                --sourceIndex;
            }
            let sources = sourcesLists[listIndex][sourceIndex];
            //console.log(Stringify2(sources));

            //
            // TODO: I think i can combine these easily, just separating to not break --xor for now
            //
            if (op === Op.xor) {
                if (_.isEmpty(primaryNameSrcList)) {
                    // or just = sources.pnsl.sort(); ?
                    primaryNameSrcList.push(...sources.primaryNameSrcList); // .sort((a, b) => { return a.count - b.count; }));
                } else {
                    // TODO: hash of primary sources would be faster here.  inside inner loop.
                    let combinedNameSrcList = primaryNameSrcList.concat(sources.primaryNameSrcList);
                    if (_.uniqBy(combinedNameSrcList, NameCount.count).length === combinedNameSrcList.length) {
                        primaryNameSrcList = combinedNameSrcList; // .sort
                    } else {
                        primaryNameSrcList = []; // needed for _.isEmpty() below
                        break;
                    }
                }
            } else {
                if (_.isEmpty(result.primaryNameSrcList)) {
                    // or just = sources.pnsl.sort(); ?
                    result.primaryNameSrcList.push(...sources.primaryNameSrcList); // .sort((a, b) => { return a.count - b.count; }));
		    //console.log(`  result.primaryNameSrcList: ${NameCount.listToString(result.primaryNameSrcList)}`);
                } else {
                    //
                    // TODO: you know, i'm not sure any of the below is relevant. flip flopping.
		    //
		    // it seemed to stem from the idea that when using this branch for both --xor and --or,
		    // i need to save the srcNcList stuff during the --xor pass so that i can use it on the
		    // --or pass. (i think).
		    //
		    // but i'm just not seeing it now. all the srcNcList stuff is *only* for --or, and it
		    // already exists in the sourcesLists supplied for --or.
		    //
		    // Ok, so, later, we're going to merge XorSourcesList with OrSourcesList. There, we do copy
		    // over the orSourcesLists to the merged results. So I think that is all we need.
		    //
		    // ---
		    //
                    // add/merge the rest of the sources stuff here too, specifically srcNameList for --or at least
                    // can maybe be achieved via mergeSources, or a mergeCompatibleSources wrapper around it.
                    // (the inner loop code of mergeCompatibleSourcesLists)
                    // something like:
                    //
                    // result = mergeCompatibleSource(result, sources);
                    //
                    // NOTE: result.primaryNameSrcList will not exist in this case.  should change logic
                    // in this function to either not require a primaryNameSrcList as the "success" indicator
                    // (better) or to include an empty-array primaryNameSrcList in the mergeCompatibleSources
                    // failure return value (worse)

                    // TODO: hash of primary sources would be faster here.  inside inner loop.
                    let combinedNameSrcList = result.primaryNameSrcList.concat(sources.primaryNameSrcList);
                    if (_.uniqBy(combinedNameSrcList, NameCount.count).length === combinedNameSrcList.length) {
			//console.log('  valid');
                        result.primaryNameSrcList = combinedNameSrcList; // .sort
                    } else {
			//console.log('  invalid');
                        result.primaryNameSrcList = []; // needed for _.isEmpty() below
                        break;
                    }
                }
            }
        }
	// ugh. getting uglier by the minute.
	let pnsl = (op === Op.or) ? result.primaryNameSrcList : primaryNameSrcList;
        if (!_.isEmpty(pnsl)) {
            sourcesList.push((op === Op.or) ? result : { primaryNameSrcList });
        }
	++iter;
    }
    if (0 && op === Op.or) console.log(`%% sourcesList(${sourcesList.length}): ${Stringify2(sourcesList)}`);
    return sourcesList;
};

//
//
let getUseSourcesList = (ncDataLists, op) => {
    let loggy = 0; // op === Op.or;
    if (loggy) console.log(`ncDataLists: ${Stringify2(ncDataLists)}`);
    if (_.isEmpty(ncDataLists[0])) return [];
    let sourcesLists = buildUseSourcesLists(ncDataLists);
    if (loggy) console.log(`buildUseSourcesLists: ${Stringify2(sourcesLists)}`);
    return mergeCompatibleUseSources(sourcesLists, op);
};

// nested loops over XorSources, OrSources primaryNameSrcLists,
// looking for compatible lists
//
let mergeOrSourcesList = (sourcesList, orSourcesList) => {
    // NOTE: optimization, can be implemented with separate loop, 
    // (can start with LAST item in list as that should be the one with all
    // --or options, and if that fails, we can bail)
    let mergedSourcesList = [];
    for (let sources of sourcesList) {
        for (let orSources of orSourcesList) {
            //
            // TODO:  call mergeCompatibleSources
            //
            let combinedNameSrcList = sources.primaryNameSrcList.concat(orSources.primaryNameSrcList);
            // possible faulty (rarish) optimization, only checking clue count
            if (_.uniqBy(combinedNameSrcList, NameCount.count).length === combinedNameSrcList.length) {
                mergedSourcesList.push({
                    primaryNameSrcList: combinedNameSrcList,
                    orSourcesLists: orSources.sourcesLists  // yeah this terminology will confuse pretty much anymore
                });
            } else {
		console.error(`not unique, sources: ${NameCount.listToString(sources.primaryNameSrcList)}, ` +
			      `orSources: ${NameCount.listToString(orSources.primaryNameSrcList)}`);
	    }
        }
    }
    return mergedSourcesList;
};

//
//
let getCompatibleUseSourcesFromNcData = (args) => {
    // XOR first
    let sourcesList = getUseSourcesList(args.allXorNcDataLists, Op.xor);
    //console.log(`xorSourcesList(${xorSourcesList.length): ${Stringify2(xorSourcesList)}`);

    // NOTE: orSourcesList is currently a little different than xor
    // OR next
    let orSourcesList = getUseSourcesList(args.allOrNcDataLists, Op.or);
    //console.log(`orSourcesList(${orSourcesList.length}) ${Stringify2(orSourcesList)}`);

    // final: merge or with xor
    if (!_.isEmpty(orSourcesList)) {
	sourcesList = mergeOrSourcesList(sourcesList, orSourcesList);
	//console.log(`orSourcesList(${orSourcesList.length}), mergedSources(${sourcesList.length}): ${Stringify2(sourcesList)}`);
    }
    return sourcesList; // xorSourcesList;
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

//
//
let isCompatibleWithOrSourcesLists = (sources, orSourcesLists) => {
    // TODO: yes this happens. why I don't know.
    //if (_.isEmpty(orSourcesLists)) console.error(`empty orSourcesList!`);
    if (!orSourcesLists || _.isEmpty(orSourcesLists)) {
	// Return false because the caller's presumption is that sources is not
	// wholly compatible with the useSources currently under consideration,
	// and thus called this function seeking an exception due to an orSources
	// match. If there are no orSources that match, we revert to the original
	// presumption (incompatible). The semantics could be made a little cleaner
	// by e.g. returning an empty array here, and a populated array for a match.
	return false;
    }
    //console.log(`orSourcesLists(${orSourcesLists.length}): ${Stringify2(orSourcesLists)}`);
    const singlePrimaryNc = sources.ncList.length === 1 && sources.ncList[0].count === 1;
    for (let orSourcesList of orSourcesLists) {
	let match = false;
	for (let orSources of orSourcesList) {
	    const numCommonPrimarySources = _.intersectionBy(sources.primaryNameSrcList, orSources.primaryNameSrcList, NameCount.count).length;
	    if (numCommonPrimarySources === orSources.primaryNameSrcList.length) {
		if (loggy) {
		    console.log(`--or matching ${NameCount.listToString(sources.ncList)} to ${orSources.ncList}`);
		    console.log(`     srcNcLists ${showNcLists(sources.srcNcLists)}`);
		}
		if (singlePrimaryNc || matchAnyNcList(orSources.ncList, sources.srcNcLists)) {
		    if (loggy) console.log(`     MATCHED: ${orSources.ncList} with something`);
		    match = true;
		    break;
		}
	    }
	}
	if (!match) return false;
    }
    return true;
};

let isCompatibleWithUseSourcesList = (sources, useSourcesList) => {
    let hasOrSources = false;
    for (let source of sources) {
        for (let useSources of useSourcesList) {
            const allUnique = allCountUnique(source.primaryNameSrcList, useSources.primaryNameSrcList);
            if (allUnique || isCompatibleWithOrSourcesLists(source, useSources.orSourcesLists)) {
		//if (!allUnique) console.error('--or match!');
		return true;
	    }
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

    let useSourcesList = args.useSourcesList;
    if (0) console.log(`compatibleUseNcDataSources: ${Stringify2(useSourcesList)}`);

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
                sources = mergeAllCompatibleSources(result.ncList);
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
                hash[key].isUseNcCompatible = isCompatibleWithUseSourcesList(sources, useSourcesList);
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
    if (1) {
	console.error(`combos(${comboCount}) variations(${totalVariationCount}) cacheHits(${cacheHitCount}) incompatible(${numIncompatible}) ` +
                      `actual(${totalVariationCount - cacheHitCount - numIncompatible}) ${duration}ms`);
    } else {
	process.stderr.write('.');
    }

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

    let comboArgs = {
        sum: args.sum,
        max: args.max,
        allXorNcDataLists: args.allXorNcDataLists,
        allAndNcDataLists: args.allAndNcDataLists,
        allOrNcDataLists: args.allOrNcDataLists,
        useSourcesList: args.useSourcesList
    };
    let combos = getCombosForUseNcLists(comboArgs, options);
    return combos;
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
    args.useSourcesList = getCompatibleUseSourcesFromNcData(args);
    if (_.isEmpty(args.useSourcesList)) {
        console.error('incompatible --xor/--or params');
        process.exit(-1);
    }

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
    console.error(`--combos: ${PrettyMs(d)}`);

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
