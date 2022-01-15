//
// combo-maker.js
//

'use strict';

const _           = require('lodash');
const BootstrapComboMaker = require('./bootstrap-combo-maker');
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

// key types:
//{
// A:
//  'jack:3': {
//    'card:2': {
// B:
//      'bird:1,red:1': [   // multiple primary NCs with array value type, split them
//        'bird:2,red:8'
//      ]
//    },
//    'face:1': {
// C:
//      'face:1': [         // single primary NC with array value type, ignore
//        'face:10'
//      ]
//    }
//  }
//}
//
//{
// D:
//  'face:1': [              // single top-level primary NC with array value type, allow
//    'face:10'
//  ]
//}
let recursiveAddSrcNcLists = (obj, resultMap, top = true) => {
    let keys = _.flatMap(_.keys(resultMap), key => {
        let val = resultMap[key];
        if (_.isObject(val)) {
            // A: non-array object value type: allow
            if (!_.isArray(val)) return key;
            // split multiple primary NCs into separate keys
            let splitKeys = key.split(',');
            // B: comma separated key with array value type: split; TODO assert primary?
            if (splitKeys.length > 1) return splitKeys;
            // D: single top-level key with array value type: allow; TODO assert primary?
            if (top) {
		if (loggy) console.log(`D: ${key}`);
		return key;
	    }
            // C: single nested key with array value type: ignore; TODO assert primary?
        }
        if (loggy) console.log(`F: ${key}`);
        return [];
    });
    if (loggy) console.log(keys);
    if (!_.isEmpty(keys)) {
	// push combined sorted keys for multi-key case
	if (keys.length > 1) {
	    let sortedKeys = keys.sort().toString();
            obj.list.push(sortedKeys);
	    obj.map[sortedKeys] = true;
	}
        keys.forEach(key => {
	    // push individual keys
            obj.list.push(key);
	    obj.map[key] = true;
            let val = resultMap[key];
            if (val && !_.isArray(val)) {
                recursiveAddSrcNcLists(obj, val, false);
            }
        });
    }
    return obj;
};

function buildSrcNcLists (resultMap) {
    return recursiveAddSrcNcLists({ list: [], map: {} }, resultMap);
}

function getSourcesList (nc) {
    const sources = [];
    ClueManager.getKnownSourceMapEntries(nc).forEach(entry => {
        entry.results.forEach(result => {
            ClueManager.primaryNcListToNameSrcLists(result.ncList).forEach(primaryNameSrcList => {
		let srcNcLists;
		let srcNcMap = {};
		if (result.resultMap) {
		    let srcNcData = buildSrcNcLists(result.resultMap.map());
		    srcNcLists = srcNcData.list;
		    srcNcMap = srcNcData.map;
		} else {
		    if (result.ncList.length !== 1 || result.ncList[0].count !== 1) throw new Error("wrong assumption");
		    srcNcLists = [result.ncList.toString()];
		    srcNcMap[result.ncList.toString()] = true;
		}
                let source = {
                    ncList: [nc],
                    primaryNameSrcList,
                    srcNcLists,
		    srcNcMap
                };
                if (nc.count > 1) {
                    if (loggy) console.log(`(${nc}): srcNcLists: ${showNcLists(source.srcNcLists)}\n,resultMap: ${Stringify2(result.resultMap.map())}`);
                    // TODO: there is possibly some operator (different than --or) where I should add all peers
		    // (same count) of 'nc' that have same primary sources. Achievable by just looking at resultMap? 
                    source.srcNcLists.push(nc.toString());
                    source.srcNcMap[nc.toString()] = true;
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

//
//
let mergeSources = (sources1, sources2) => {
    let mergedSources = {};
    mergedSources.ncList = [...sources1.ncList, ...sources2.ncList];
    mergedSources.ncCsv = mergedSources.ncList.sort().toString();
    mergedSources.primaryNameSrcList = [...sources1.primaryNameSrcList, ...sources2.primaryNameSrcList];
    mergedSources.srcNcLists = [...sources1.srcNcLists, ...sources2.srcNcLists];
    mergedSources.srcNcMap = {};
    _.keys(sources1.srcNcMap).forEach(key => { mergedSources.srcNcMap[key] = true; });
    _.keys(sources2.srcNcMap).forEach(key => { mergedSources.srcNcMap[key] = true; });
    mergedSources.srcNcMap[mergedSources.ncCsv] = true;
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
        result += `    ncList: ${source.ncList}\n`;
        result += `    primaryNameSrcList: ${source.primaryNameSrcList}\n`;
        result += `    srcNcLists: ${Stringify2(source.srcNcLists)}\n`;
        result += '  }';
    }
    return result + "\n]";
};

//
//
let mergeAllCompatibleSources = (ncList) => {
    if (ncList.length > 2) { // because broken for > 2 below
	console.log(ncList.toString());
	throw new Error(`ncList.length > 2 (${ncList.length})`);
    }
    let sources = getSourcesList(ncList[0]);
    for (let ncIndex = 1; ncIndex < ncList.length; ncIndex += 1) {
        const nextSources = getSourcesList(ncList[ncIndex]);
        sources = mergeCompatibleSourcesLists(sources, nextSources);
        if (loggy) {
            console.log(`** merging index: ${ncIndex}, ${ncList[ncIndex]} as nextSources:`);
            console.log(`${stringifySources(nextSources)}`);
            console.log(`** result:\n${stringifySources(sources)}`);
        }
        if (_.isEmpty(sources)) break; // TODO BUG this is broken for > 2; should be something like: if (sources.length !== ncIndex + 1) 
    }
    loggy = false;
    return sources;
};

// TODO: use new Set() here for god's sake
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
                //let key = sources.primaryNameSrcList.map(_.toString).sort().toString();
                let key = sources.primaryNameSrcList.sort().toString();
                if (!hashList[sourcesListIndex][key]) {
                    sourcesLists[sourcesListIndex].push(sources);
                    hashList[sourcesListIndex][key] = true;
                }
            }
        }
    }
    return sourcesLists;
};

// Here we have primaryNameSrcList which is the combined compatible primary sources of
// one or more --or arguments.
//
// And we have orSourcesList, which is a list of sources for each --or argument that is
// *not* included in primaryNameSrcList.
//	    
// Whittle down these lists of separate --or argument sources into lists of combined
// compatible sources, with each result list containing one element from each source list.
//
// So for [ [ a, b], [b, d] ] the candidates would be [a, b], [a, d], [b, b], [b, d],
// and [b, b] would be filtered out as incompatible.
//
let getCompatibleOrSourcesLists = (primaryNameSrcList, orSourcesLists) => {
    if (_.isEmpty(orSourcesLists)) return [];

    let listArray = orSourcesLists.map(sourcesList => [...Array(sourcesList.length).keys()]);
    //console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourcesLists(${orSourcesLists.length}): ${Stringify2(orSourcesLists)}`);

    let peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    
    let sourcesLists = [];
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
        //console.log(`indexList: ${stringify(indexList)}`);
	let sourcesList = [];
	let primarySrcSet = new Set(primaryNameSrcList.map(nameSrc => NameCount.count));
        for (let [listIndex, sourceIndex] of indexList.entries()) {
	    let sources = orSourcesLists[listIndex][sourceIndex];
	    //console.log(`sources: ${Stringify2(sources)}`);
	    // TODO: if (!allArrayElemsInSet(sources.primaryNameSrcList, primarySrcSet))
	    let prevSetSize = primarySrcSet.size;
	    for (let nameSrc of sources.primaryNameSrcList) {
		primarySrcSet.add(nameSrc.count);
	    }
	    if (primarySrcSet.size !== prevSetSize + sources.primaryNameSrcList.length) {
		sourcesList = [];
		break;
	    }
	    sourcesList.push(sources);
	}
	if (!_.isEmpty(sourcesList)) {
	    sourcesLists.push(sourcesList);
	}
    }
    return sourcesLists;
};

// Here we have 'orSourcesLists', created from getUseSourcesList(Op.or).
//
// Generate a sorted ncCsv from the combined NCs of each ncList across all sources
// in each sourcesList. Return a list of ncCsvs.
//
// It'd be preferable to embed this ncCsv within each sourcesList itself. I'd need to
// wrap it in an object like { sourcesList, ncCsv }.
//
let buildSourcesNcCsvList = (orSourcesLists) => {
    return orSourcesLists.map(sourcesList => 
	    _.flatMap(sourcesList.map(sources => sources.ncList)).sort().toString());
};

//
//
let ZZ = 0;
let mergeCompatibleUseSources = (sourcesLists, op) => {
    // TODO: sometimes a sourcesList is empty, like if doing $(cat required) with a
    // low clue count range (e.g. -c2,4). should that even be allowed?
    let pad = (op === Op.or) ? 1 : 0;
    let listArray = sourcesLists.map(sourcesList => [...Array(sourcesList.length + pad).keys()]);
    //ZZ = (op === Op.or);
    if (ZZ) console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourcesLists(${sourcesLists.length}): ${Stringify2(sourcesLists)}`);

    let peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    
    let iter = 0;
    let sourcesList = [];  // resultList? 
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
        if (ZZ) console.log(`indexList: ${stringify(indexList)}`);
	//
	// TODO: list of sourcesLists outside of this loop. 
	// assign result.sourceLists inside indexList.entries() loop. 
	//
	let primaryNameSrcList = [];
	let orSourcesLists = [];
	let success = true;
        for (let [listIndex, sourceIndex] of indexList.entries()) {
	    if (!orSourcesLists[listIndex]) orSourcesLists.push([]);
	    const orSourcesList = orSourcesLists[listIndex];
            if (ZZ) console.log(`iter(${iter}) listIndex(${listIndex}) sourceIndex(${sourceIndex}), orSourcesList(${orSourcesList.length})`);
            if (op === Op.or) {
                if (sourceIndex === 0) {
		    if (ZZ) console.log(`adding to orSourcesLists @ index(${listIndex}) length(${orSourcesList.length}) count(${sourcesLists[listIndex].length})`);
		    if (ZZ) console.log(`  sourcesLists[listIndex][0].ncList: ${sourcesLists[listIndex][0].ncList}`);
                    orSourcesLists.push(sourcesLists[listIndex]);
		    // TODO: probably can remove this (and all references to sources.ncCsv) at some point
		    sourcesLists[listIndex].forEach(source => { source.ncCsv = source.ncList.sort().toString(); });
                    continue;
                }
                --sourceIndex;
            }
	    let sources = sourcesLists[listIndex][sourceIndex];
            if (_.isEmpty(primaryNameSrcList)) {
                primaryNameSrcList.push(...sources.primaryNameSrcList);
		if (ZZ) console.log(`pnsl, initial: ${primaryNameSrcList}`);
            } else {
		// 
                // TODO: hash of primary sources would be faster here.  inside inner loop.
		// TODO: or use push instead of concat
		//
                let combinedNameSrcList = primaryNameSrcList.concat(sources.primaryNameSrcList);
                if (_.uniqBy(combinedNameSrcList, NameCount.count).length === combinedNameSrcList.length) {
                    primaryNameSrcList = combinedNameSrcList;
		    if (ZZ) console.log(`pnsl, combined: ${primaryNameSrcList}`);
                } else {
		    if (ZZ) console.log(`pnsl, emptied: ${primaryNameSrcList}`);
		    success = false;
                    break;
                }
            }
        }
	if (success) {
	    if (ZZ) console.log(`pnsl, final: ${primaryNameSrcList}`);
	    let result = { primaryNameSrcList };
	    if (op === Op.or) {
		let nonEmptyOrSourcesLists = orSourcesLists.filter(sourcesList => !_.isEmpty(sourcesList));
		result.sourcesLists = getCompatibleOrSourcesLists(primaryNameSrcList, nonEmptyOrSourcesLists);
		result.sourcesNcCsvList = buildSourcesNcCsvList(result.sourcesLists);
		if (ZZ && _.isEmpty(result.sourcesLists) && !_.isEmpty(nonEmptyOrSourcesLists)) {
		    console.log(`before: orSourcesLists(${orSourcesLists.length}): ${Stringify2(orSourcesLists)}`);
		    console.log(`after: result.sourcesLists(${result.sourcesLists.length}): ${Stringify2(result.sourcesLists)}`);
		    ZZ = false;
		}
	    }
            sourcesList.push(result);
        }
	++iter;
    }
    return sourcesList;
};

//
//
let getUseSourcesList = (ncDataLists, op) => {
    //console.log(`ncDataLists: ${Stringify2(ncDataLists)}`);
    if (_.isEmpty(ncDataLists[0])) return [];
    let sourcesLists = buildUseSourcesLists(ncDataLists);
    //console.log(`buildUseSourcesLists: ${Stringify2(sourcesLists)}`);
    return mergeCompatibleUseSources(sourcesLists, op);
};

//
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
            // TODO:  call mergeCompatibleSources.  still? or..
            //
            let combinedNameSrcList = sources.primaryNameSrcList.concat(orSources.primaryNameSrcList);
	    // 
            // TODO: hash of primary sources faster here?
	    //
            // possible faulty (rarish) optimization, only checking clue count
	    // TODO: not 100% sure this change is correct...
	    // if (_.uniqBy(combinedNameSrcList, NameCount.count).length === combinedNameSrcList.length) {
            const numUnique = _.uniqBy(combinedNameSrcList, NameCount.count).length;
	    if (numUnique === orSources.primaryNameSrcList.length) {
		console.error('an --or value is implicitly compatible with an --xor value, making this --or value unnecessary');
	    } else if (numUnique === combinedNameSrcList.length) {
                mergedSourcesList.push({
                    primaryNameSrcList: combinedNameSrcList,
                    orSourcesLists: orSources.sourcesLists,  // yeah this terminology will confuse pretty much anymore
		    orSourcesNcCsvList: orSources.sourcesNcCsvList
                });
            } else if (0) {
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

    // OR next
    let orSourcesList = getUseSourcesList(args.allOrNcDataLists, Op.or);
    //console.log(`orSourcesList(${orSourcesList.length}) ${Stringify2(orSourcesList)}`);

    // final: merge or with xor
    if (!_.isEmpty(orSourcesList)) {
	// NOTE: orSourcesList is currently a little different than xor
	sourcesList = mergeOrSourcesList(sourcesList, orSourcesList);
	//console.log(`orSourcesList(${orSourcesList.length}), mergedSources(${sourcesList.length}): ${Stringify2(sourcesList)}`);
    }
    console.error(`orSourcesList(${orSourcesList.length})`);

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
let next = (clueSourceList, sourceIndexes) => {
    for (;;) {
        if (!nextIndex(clueSourceList, sourceIndexes)) {
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
let first = (clueSourceList, sourceIndexes) => {
    for (let index = 0; index < clueSourceList.length; ++index) {
        sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;
    return next(clueSourceList, sourceIndexes);
};

//
//
let XX = 0;
let isCompatibleWithOrSources = (sources, useSources) => {
    // TODO: yes this happens. why I don't know.
    const orSourcesLists = useSources.orSourcesLists;
    //if (_.isEmpty(orSourcesLists)) console.error(`empty orSourcesList!`);
    if (!orSourcesLists || _.isEmpty(orSourcesLists)) {
	// Return false because the caller's presumption is that sources is not
	// wholly compatible with the useSources currently under consideration,
	// and thus called this function seeking an exception due to an orSources
	// match. If there are no orSources that match, we revert to the original
	// presumption (incompatible). The semantics could be made a little cleaner
	// by e.g. returning an empty array here, and a populated array for a match.
	// or maybe just a method name change.
	return false;
    }
    //console.log(`orSourcesLists(${orSourcesLists.length}): ${Stringify2(orSourcesLists)}`);
    for (let [listIndex, orSourcesList] of orSourcesLists.entries()) {
	let ncCsv = useSources.orSourcesNcCsvList[listIndex];
	if (sources.srcNcMap[ncCsv]) {
	    //:: orSources ncCsv matches sources ncCsv

	    // essentially, subtract all orSources.primaryNameSrcList entries from from sources.primaryNameSrcList
	    // based on primary source. probably not fast.
	    // TODO: profile, speed this shit up.
	    const orSourcesPrimaryNameSrcList = _.flatten(orSourcesList.map(orSources => orSources.primaryNameSrcList));
	    const xorPrimaryNameSrcList = _.xorBy(sources.primaryNameSrcList, orSourcesPrimaryNameSrcList, NameCount.count);
	    if (XX) {
		console.log(`sources: ${sources.primaryNameSrcList}`);
		console.log(`or: ${orSourcesPrimaryNameSrcList}`);
		console.log(`xor: ${xorPrimaryNameSrcList}`);
		XX = false;
	    }
	    if (xorPrimaryNameSrcList.length === sources.primaryNameSrcList.length - orSourcesPrimaryNameSrcList.length) {
		//:: orSources primary sources match sources primary sources
		if (allCountUnique(xorPrimaryNameSrcList, useSources.primaryNameSrcList)) {
		    //:: sources' remaining primary sources are compatible with useSources' primary sources
		    return true;
		}
	    }
	}
    }
    return false;
};

//
//
let isCompatibleWithUseSources = (sourcesList, useSourcesList) => {
    if (_.isEmpty(useSourcesList)) return true;
    for (let sources of sourcesList) {
        for (let useSources of useSourcesList) {
            const allUnique = allCountUnique(sources.primaryNameSrcList, useSources.primaryNameSrcList);
            if (allUnique || isCompatibleWithOrSources(sources, useSources)) {
		return true;
	    }
        }
    }
    return false;
};

//
//
let getCombosForUseNcLists = (sum, max, args) => {
    let hash = {};
    let combos = [];

    let comboCount = 0;
    let totalVariations = 0;
    let numCacheHits = 0;
    let numIncompatible = 0;
    
    let MILLY = 1000000n;
    let start = process.hrtime.bigint();

    let useSourcesList = args.useSourcesList;
    if (0) console.log(`useSourcesList: ${Stringify2(useSourcesList)}`);

    // for each sourceList in sourceListArray
    ClueManager.getClueSourceListArray({ sum, max }).forEach(clueSourceList => {
        comboCount += 1;

        //console.log(`sum(${sum}) max(${max}) clueSrcList: ${Stringify(clueSourceList)}`);
        let sourceIndexes = [];
        let result = first(clueSourceList, sourceIndexes);
        if (result.done) return; // continue; 

        let numVariations = 1;

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
                result = next(clueSourceList, sourceIndexes);
                if (result.done) break;
                numVariations += 1;
            } else {
                firstIter = false;
            }
            //console.log(`result.nameList: ${result.nameList}`);
            //console.log(`result.ncList: ${result.ncList}`);

            //const key = NameCount.listToString(result.ncList);
	    const key = result.ncList.sort().toString();
	    let cacheHit = false;
            let sourcesList;
            if (!hash[key]) {
                sourcesList = mergeAllCompatibleSources(result.ncList);
                //console.log(`$$ sources: ${Stringify2(sources)}`);
                hash[key] = { sourcesList };
            } else {
                sourcesList = hash[key].sources;
		cacheHit = true;
		numCacheHits += 1;
            }

            if (logging) console.log(`  found compatible sources: ${!_.isEmpty(sourcesList)}`);

            // failed to find any compatible combos
            if (_.isEmpty(sourcesList)) continue;

            if (_.isUndefined(hash[key].isCompatible)) {
                hash[key].isCompatible = isCompatibleWithUseSources(sourcesList, useSourcesList);
            }
            if (hash[key].isCompatible) {
                combos.push(result.nameList.toString());
            } else if (!cacheHit) {
                numIncompatible += 1;
	    }
        }
        totalVariations += numVariations;
    }, this);

    let duration = (process.hrtime.bigint() - start) / MILLY;
    Debug(`combos(${comboCount}) variations(${totalVariations}) cacheHits(${numCacheHits}) incompatible(${numIncompatible}) ` +
          `actual(${totalVariations - numCacheHits - numIncompatible}) ${duration}ms`);

    if (1) {
	console.error(`combos(${comboCount}) variations(${totalVariations}) cacheHits(${numCacheHits}) incompatible(${numIncompatible}) ` +
                      `actual(${totalVariations - numCacheHits - numIncompatible}) ${duration}ms`);
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
let makeCombosForSum = (sum, max, args) => {
    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
    }

    // TODO move this a layer or two out; use "validateArgs" 
    if (!_.isEmpty(args.require)) throw new Error('require not yet supported');
    if (args.sources) throw new Error('sources not yet supported');

    let combos = getCombosForUseNcLists(sum, max, args);
    return combos;
};

//
//
let parallel_makeCombosForRange = (first, last, args) => {
    let range = [...Array(last + 1).keys()].slice(first)
	.map(sum => Object({
	    apple: args.apple,
	    final: args.final,
	    meta:  args.meta,
	    sum,
            max: (args.max > sum) ? sum : args.max,
	    xor: args.xor,
	    //and: args.and,
	    or: args.or,
	    useSourcesList: args.useSourcesList,
	    parallel: true
	}));

    let cpus = OS.cpus().length;
    let cpus_used = cpus <= 6 ? cpus: cpus / 2;
    console.error(`cpus: ${cpus} max used: ${cpus_used}`);
    let p = new Parallel(range, {
	maxWorkers: cpus_used,
    	evalPath: '${__dirname}/../../modules/bootstrap-combo-maker.js'
    });
    let entrypoint = BootstrapComboMaker.entrypoint;
    //console.error('++makeCombosForRange');
    let beginDate = new Date();
    return p.map(entrypoint).then(data => {
	//console.log(`data = ${typeof data} array: ${_.isArray(data)}, data[${0}] = ${Stringify(data[0])}`);
	let d = new Duration(beginDate, new Date()).milliseconds;
	console.error(`time: ${PrettyMs(d)} chunks: ${data.length}`);
    });
    // check if range == data and /or if .then(return) passes thru
    //const filterResult = ClueManager.filter(data[i], args.sum, comboMap);
};

//
//
let test = (sum, max, args) => {
    // buildOrSourcesSet (part of buildUseSourcesList perhaps)
    
    let useSourcesList = args.useSourcesList;
    let numUseSources = useSourcesList.length;
    let numOrSourcesLists = 0;
    let numOrSources = 0;

    let show = true;

    for (let useSources of useSourcesList) {
	if (show) {
	    //console.log(`${Stringify2(useSources)}`);
	    show = false;
	}
	let orSourcesLists = useSources.orSourcesLists;
	numOrSourcesLists += orSourcesLists.length;
	if (!orSourcesLists || _.isEmpty(orSourcesLists)) continue;
	for (let orSourcesList of orSourcesLists) {
	    numOrSources += orSourcesList.length;
	    /*
	    console.log('----------------');
	    for (let orSources of orSourcesList) {
		console.log(`${orSources.ncList}`);
	    }
	    */
	}
    }
    console.error(`useSources(${numUseSources}) orSourcesLists(${numOrSourcesLists}) orSources(${numOrSources})`);
};

//
//
let makeCombos = (args) => {
    let sumRange;
    if (!_.isUndefined(args.sum)) {
        // is _chain even necessary here?
        sumRange = _.chain(args.sum).split(',').map(_.toNumber).value();
    }
    Expect(sumRange).is.an.Array().with.property('length').below(3); // of.at.most(2);
    Debug('++combos' +
          `, sum: ${sumRange}` +
          `, max: ${args.max}` +
	  //`, require: ${args.require}` +
	  //`, sources: ${args.sources}` +
          `, use: ${args.use}`);
    
    let begin = new Date();
    args.allXorNcDataLists = args.xor ? buildAllUseNcDataLists(args.xor) : [ [] ];
    //console.log(`allXorNcDataLists: ${Stringify2(args.allXorNcDataLists)}`);
    //args.allAndNcDataLists = args.and ? buildAllUseNcDataLists(args.and) : [ [] ];
    args.allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or) : [ [] ];
    args.useSourcesList = getCompatibleUseSourcesFromNcData(args);

    // test
    //let lastSum = sumRange.length > 1 ? sumRange[1] : sumRange[0];
    //test(sumRange[0], lastSum, args);

    let d = new Duration(begin, new Date()).milliseconds;
    console.error(`Precompute(${PrettyMs(d)})`);

    if (_.isEmpty(args.useSourcesList)) {
	if (args.xor || args.or) {
            console.error('incompatible --xor/--or params');
            process.exit(-1);
	}
    }

    let total = 0;
    begin = new Date();
    if (args.parallel) {
	let first = sumRange[0];
	let last = sumRange.length > 1 ? sumRange[1] : first;
	parallel_makeCombosForRange(first, last, args).then(data => {
	    let comboSet = new Set();
	    for (let arr of data) {
		arr.forEach(comboStr => comboSet.add(comboStr));
	    }
	    for (let combo of comboSet.keys()) {
		console.log(combo);
	    }
	});
    } else {
	let comboMap = {};
	let lastSum = sumRange.length > 1 ? sumRange[1] : sumRange[0];
	for (let sum = sumRange[0]; sum <= lastSum; ++sum) {
	    // TODO: Fix this abomination
            args.sum = sum;
            let max = args.max;
	    if (args.max > args.sum) args.max = args.sum;
            // TODO: return # of combos filtered due to note name match
            const comboList = makeCombosForSum(sum, args.max, args);
            args.max = max;
            total += comboList.length;
            const filterResult = ClueManager.filter(comboList, sum, comboMap);
	}
	d = new Duration(begin, new Date()).milliseconds;
	console.error(`--combos: ${PrettyMs(d)}`);
	Debug(`total: ${total}, filtered(${_.size(comboMap)})`);
	_.keys(comboMap).forEach(nameCsv => console.log(nameCsv));
	//console.log(`${Stringify(comboMap)}`);
	//process.stderr.write('\n');
    }
    return 1;
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
    makeCombos,
    makeCombosForSum
};
