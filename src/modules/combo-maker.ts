//
// combo-maker.ts
//

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

//
//
interface NameCount {
    name: string;
    count: number;
}
type NCList = NameCount[];

interface StringBoolMap {
    [key: string]: boolean; // for now; eventually maybe array of string (sorted primary nameSrcCsv)
}

interface StringAnyMap {
    [key: string]: any;
}

// TODO
//
type ValidateResult = any;

//
//
interface NCData {
    ncList: NCList;
}
type NCDataList = NCData[];

//
//
interface SourceBase {
    primaryNameSrcList: NCList;
}

//
//
interface LazySourceData extends SourceBase {
    ncList: NameCount[];
    validateResultList: ValidateResult[];
}

//
//
interface SourceData extends SourceBase {
    ncList: NCList;
    srcNcLists: string[];
    srcNcMap: StringBoolMap;
    ncCsv?: string;
}
type SourceList = SourceData[];
type AnySourceData = LazySourceData | SourceData;

//
//
interface XorSource extends SourceBase {
}

interface OrSource extends SourceBase {
    sourceLists: SourceList[];
    sourceNcCsvList: string[];
}
type OrSourceList = OrSource[];

// optional properties until I can think of a better way.  mergeOrSourcesList.
//
interface UseSource extends SourceBase {
    orSourceLists?: SourceList[];
    orSourcesNcCsvList?: string[];
}

function Stringify(val: any) {
    return stringify(val, (value: any, indent: any, stringify: any) => {
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

function OpName (opValue: number): string {
    return _.findKey(Op, (v: number) => opValue === v);
}


//
// see: showNcLists
let listOfNcListsToString = (listOfNcLists: NCList[]): string => {
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
let stringifySourceList = (sourceList: SourceList): string => {
    let result = "[\n";
    let first = true;
    for (let source of sourceList) {
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

function showNcLists (ncLists: NCList[]): string {
    let str = "";
    let first = true;
    for (let ncList of ncLists) {
        if (!first) str += ' - ';
        str += ncList;
        first = false;
    }
    return _.isEmpty(str) ? "[]" : str;
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
let recursiveAddSrcNcLists = (obj: any, resultMap: any, top = true): any => {
    let keys: string[] = _.flatMap(_.keys(resultMap), (key: string) => {
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

//
//
function buildSrcNcLists (resultMap: any): any {
    return recursiveAddSrcNcLists({ list: [], map: {} }, resultMap);
}

//
//
let populateSourceData = (lazySource: SourceBase, nc: NameCount, validateResult: ValidateResult): SourceData => {
    let source: SourceData = lazySource /*as SourceBase*/ as SourceData;
    if (validateResult.resultMap) {
        let srcNcData = buildSrcNcLists(validateResult.resultMap.map());
        source.srcNcLists = srcNcData.list;
        source.srcNcMap = srcNcData.map;
    } else {
        if (validateResult.ncList.length !== 1 || validateResult.ncList[0].count !== 1) throw new Error("wrong assumption");
	let ncCsv = validateResult.ncList.toString();
        source.srcNcLists = [ncCsv];
        source.srcNcMap = { "${ncCsv}": true };
    }
    if (nc.count > 1) {
	let ncStr = nc.toString();
        source.srcNcLists.push(ncStr);
        source.srcNcMap[ncStr] = true;
    }
    source.ncList = [nc]; // TODO i could try getting rid of "LazySource.nc" and just make this part of LazySouceData
    //source.srcNcLists = srcNcLists;
    //source.srcNcMap = srcNcMap;
    
    /*
    let source: SourceData = {
        primaryNameSrcList,
        ncList: [nc],
        srcNcLists,
        srcNcMap
    };
    */
    if (loggy || logging > 3) {
        console.log(`getSourceList() ncList: ${source.ncList}, srcNcLists: ${source.srcNcLists}`);
        if (_.isEmpty(source.srcNcLists)) console.log(`empty srcNcList: ${Stringify(validateResult.resultMap.map())}`);
    }
    return source;
};

//
//
let getSourceData = (nc: NameCount, validateResult: ValidateResult, lazy: boolean): AnySourceData => {
    const primaryNameSrcList: NCList = validateResult.nameSrcList;
    return lazy
	? { primaryNameSrcList,	ncList: [nc], validateResultList: [validateResult] }
	: populateSourceData({ primaryNameSrcList }, nc, validateResult);
};

//
//
let getSourceList = (nc: NameCount, lazy: boolean): AnySourceData[] => {
    const sourceList: AnySourceData[] = [];
    ClueManager.getKnownSourceMapEntries(nc).forEach((entry: any) => {
        entry.results.forEach((result: ValidateResult) => {
            sourceList.push(getSourceData(nc, result, lazy));
        });
    });
    return sourceList;
};

//
//
let mergeSources = (source1: AnySourceData, source2: AnySourceData, lazy: boolean): AnySourceData => {
    let primaryNameSrcList = [...source1.primaryNameSrcList, ...source2.primaryNameSrcList];
    let ncList = [...source1.ncList, ...source2.ncList];
    if (lazy) {
	if (ncList.length !== 2) throw new Error(`ncList.length(${ncList.length})`);
	source1 = source1 as LazySourceData;
	source2 = source2 as LazySourceData;
	let result: LazySourceData = {
	    primaryNameSrcList,
	    ncList,
	    validateResultList: [source1.validateResultList[0], source2.validateResultList[0]]
	};
	return result;
    }
    source1 = source1 as SourceData;
    source2 = source2 as SourceData;
    let mergedSource: SourceData = {
	primaryNameSrcList,
	ncList,
        srcNcLists: [...source1.srcNcLists, ...source2.srcNcLists],
        srcNcMap: {}
    };
    mergedSource.ncCsv = mergedSource.ncList.sort().toString();
    _.keys(source1.srcNcMap).forEach((key: string) => { mergedSource.srcNcMap[key] = true; });
    _.keys(source2.srcNcMap).forEach((key: string) => { mergedSource.srcNcMap[key] = true; });
    mergedSource.srcNcMap[mergedSource.ncCsv] = true;
    return mergedSource;
};

//
//
let mergeCompatibleSources = (source1: AnySourceData, source2: AnySourceData, lazy: boolean): AnySourceData[] => {
    if (0 || logging>2) console.log(`mergeCompat: nameSrcList1: ${source1.primaryNameSrcList}, nameSrcList2: ${source2.primaryNameSrcList}`);
    // TODO: this logic could be part of mergeSources
    return allCountUnique(source1.primaryNameSrcList, source2.primaryNameSrcList)
	? [mergeSources(source1, source2, lazy)]
	: [];
};

//
//
let mergeCompatibleSourceLists = (sourceList1: AnySourceData[], sourceList2: AnySourceData[], lazy: boolean): AnySourceData[] => {
    let mergedSourcesList: AnySourceData[] = [];
    for (const source1 of sourceList1) {
        for (const source2 of sourceList2) {
            mergedSourcesList.push(...mergeCompatibleSources(source1, source2, lazy));
        }
    }
    return mergedSourcesList;
};

//
//
let mergeAllCompatibleSources = (ncList: NCList, lazy = false): AnySourceData[] => {
    if (ncList.length > 2) { // because **maybe** broken for > 2 below
        throw new Error(`${ncList} length > 2 (${ncList.length})`);
    }
    // TODO:
    // Take a lazy flag parameter to this function.
    // **if Lazy**, Don't populate all the fields in each source here,
    // just primaryNameSrcList (and probably primarySrcArray), and
    // **result** - that is, attach the result object in to the source
    // in getSourceList. So we can do a post-compatible-check merge.
    // Need to figure out how to parameterize the SourceList type to
    // either contain all of the data or just the "fast" data.
    // Need to pass lazy flag to mergeCompatibleSourceLists/Sources/mergeSources
    // Probably can use narrowing using ReturnType(this) != SourceBase
    // to copy all the other shit.
    // 
    // reduce might work well here I think as well.

    let sourceList = getSourceList(ncList[0], lazy);
    for (let ncIndex = 1; ncIndex < ncList.length; ncIndex += 1) {
        const nextSourceList = getSourceList(ncList[ncIndex], lazy);
        sourceList = mergeCompatibleSourceLists(sourceList, nextSourceList, lazy);
        if (0 || loggy) {
            console.log(`** merging index: ${ncIndex}, ${ncList[ncIndex]} as nextSourceList:`);
            //console.log(`${stringifySourceList(nextSourceList)}`);
            //console.log(`** result:\n${stringifySourceList(sourceList)}`);
        }
        if (_.isEmpty(sourceList)) break; // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
    }
    loggy = false;
    return sourceList;
};

//
//
let allCountUnique = (nameSrcList1: NCList, nameSrcList2: NCList): boolean => {
    let set: Set<number> = new Set<number>();
    for (let nameSrc of nameSrcList1) {
        set.add(nameSrc.count);
    }
    for (let nameSrc of nameSrcList2) {
        if (set.has(nameSrc.count)) return false;
    }
    return true;
};

//
//
let buildUseSourcesLists = (useNcDataLists: NCDataList[]): SourceList[] => {
    let sourceLists: SourceList[] = [];
    let hashList: StringBoolMap[] = [];
    //console.log(`useNcDataLists(${useNcDataLists.length}): ${Stringify2(useNcDataLists)}`);
    for (let [dataListIndex, useNcDataList] of useNcDataLists.entries()) {
        for (let [sourceListIndex, useNcData] of useNcDataList.entries()) {
            if (!sourceLists[sourceListIndex]) sourceLists.push([]);
            if (!hashList[sourceListIndex]) hashList.push({});
            //console.log(`ncList: ${NameCount.listToString(useNcData.ncList)}`);
            let sourceList = mergeAllCompatibleSources(useNcData.ncList);
            //console.log(`sourceList(${sourceList.length}): ${Stringify2(sourceList)}`);
            for (let source of sourceList) {
                //let key = sources.primaryNameSrcList.map(_.toString).sort().toString();
                let key = source.primaryNameSrcList.sort().toString();
                if (!hashList[sourceListIndex][key]) {
                    sourceLists[sourceListIndex].push(source as SourceData);
                    hashList[sourceListIndex][key] = true;
                }
            }
        }
    }
    return sourceLists;
};

// Here we have primaryNameSrcList which is the combined compatible primary sources of
// one or more --or arguments.
//
// And we have orSourceLists, which is a list of sourceLists: one list for each --or
// argument that is *not* included in primaryNameSrcList.
//          
// Whittle down these lists of separate --or argument sources into lists of combined
// compatible sources, with each result list containing one element from each source list.
//
// So for [ [ a, b], [b, d] ] the candidates would be [a, b], [a, d], [b, b], [b, d],
// and [b, b] would be filtered out as incompatible.
//
let getCompatibleOrSourcesLists = (primaryNameSrcList: NCList, orSourceLists: SourceList[]): SourceList[] => {
    if (_.isEmpty(orSourceLists)) return [];

    let listArray = orSourceLists.map(sourceList => [...Array(sourceList.length).keys()]);
    //console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourceLists(${orSourceLists.length}): ${Stringify2(orSourceLists)}`);

    let peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    
    let sourceLists: SourceList[] = [];
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
        //console.log(`indexList: ${stringify(indexList)}`);
        let sourceList: SourceList = [];
        let primarySrcSet = new Set(primaryNameSrcList.map(nameSrc => NameCount.count));
        for (let [listIndex, sourceIndex] of indexList.entries()) {
            let sources = orSourceLists[listIndex][sourceIndex];
            //console.log(`sources: ${Stringify2(sources)}`);
            // TODO: if (!allArrayElemsInSet(sources.primaryNameSrcList, primarySrcSet))
            let prevSetSize = primarySrcSet.size;
            for (let nameSrc of sources.primaryNameSrcList) {
                primarySrcSet.add(nameSrc.count);
            }
            if (primarySrcSet.size !== prevSetSize + sources.primaryNameSrcList.length) {
                sourceList = [];
                break;
            }
            sourceList.push(sources);
        }
        if (!_.isEmpty(sourceList)) {
            sourceLists.push(sourceList);
        }
    }
    return sourceLists;
};

// Here we have 'orSourceLists', created from getUseSourcesList(Op.or).
//
// Generate a sorted ncCsv from the combined NCs of each ncList across all sources
// in each sourceList. Return a list of ncCsvs.
//
// It'd be preferable to embed this ncCsv within each sourceList itself. I'd need to
// wrap it in an object like { sourceList, ncCsv }.
//
let buildSourcesNcCsvList = (orSourceLists: SourceList[]): string[] => {
    return orSourceLists.map(sourceList => 
            _.flatMap(sourceList.map(sources => sources.ncList)).sort().toString());
};

//
//
// use generic here

let ZZ = 0;
let mergeCompatibleUseSources = <SourceType extends SourceBase>(sourceLists: SourceList[], op: any): SourceType[] => {
    // TODO: sometimes a sourceList is empty, like if doing $(cat required) with a
    // low clue count range (e.g. -c2,4). should that even be allowed?
    let pad = (op === Op.or) ? 1 : 0;
    let listArray = sourceLists.map(sourceList => [...Array(sourceList.length + pad).keys()]);
    //ZZ = (op === Op.or);
    if (ZZ) console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourceLists(${sourceLists.length}): ${Stringify2(sourceLists)}`);

    let peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    
    let iter = 0;
    let sourceList: SourceType[] = [];
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
        if (ZZ) console.log(`indexList: ${stringify(indexList)}`);
        //
        // TODO: list of sourceLists outside of this loop. 
        // assign result.sourceLists inside indexList.entries() loop. 
        //
        let primaryNameSrcList: NCList = [];
        let orSourceLists: SourceList[] = [];
        let success = true;
        for (let [listIndex, sourceIndex] of indexList.entries()) {
            if (!orSourceLists[listIndex]) orSourceLists.push([]);
            const orSourceList = orSourceLists[listIndex];
            if (ZZ) console.log(`iter(${iter}) listIndex(${listIndex}) sourceIndex(${sourceIndex}), orSourceList(${orSourceList.length})`);
            if (op === Op.or) {
                if (sourceIndex === 0) {
                    if (ZZ) console.log(`adding to orSourceLists @ index(${listIndex}) length(${orSourceList.length}) count(${sourceLists[listIndex].length})`);
                    if (ZZ) console.log(`  sourceLists[listIndex][0].ncList: ${sourceLists[listIndex][0].ncList}`);
                    orSourceLists.push(sourceLists[listIndex]);
                    // TODO: probably can remove this (and all references to sources.ncCsv) at some point
                    sourceLists[listIndex].forEach(source => { source.ncCsv = source.ncList.sort().toString(); });
                    continue;
                }
                --sourceIndex;
            }
            let source = sourceLists[listIndex][sourceIndex];
            if (_.isEmpty(primaryNameSrcList)) {
                primaryNameSrcList.push(...source.primaryNameSrcList);
                if (ZZ) console.log(`pnsl, initial: ${primaryNameSrcList}`);
            } else {
                // 
                // TODO: hash of primary sources would be faster here.  inside inner loop.
                // TODO: or use push instead of concat
                //
                let combinedNameSrcList = primaryNameSrcList.concat(source.primaryNameSrcList);
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
            let result: SourceBase = { primaryNameSrcList }; // ugly
            if (op === Op.or) {
                let nonEmptyOrSourcesLists = orSourceLists.filter(sourceList => !_.isEmpty(sourceList));
		let orResult: OrSource = result as OrSource;
                orResult.sourceLists = getCompatibleOrSourcesLists(primaryNameSrcList, nonEmptyOrSourcesLists);
                orResult.sourceNcCsvList = buildSourcesNcCsvList(orResult.sourceLists);
                if (ZZ && _.isEmpty(orResult.sourceLists) && !_.isEmpty(nonEmptyOrSourcesLists)) {
                    console.log(`before: orSourceLists(${orSourceLists.length}): ${Stringify2(orSourceLists)}`);
                    console.log(`after: result.sourceLists(${orResult.sourceLists.length}): ${Stringify2(orResult.sourceLists)}`);
                    ZZ = 0;
                }
            }
            sourceList.push(result as SourceType);
        }
        ++iter;
    }
    return sourceList;
};

//
//
// use generic here

let getUseSourcesList = <SourceType extends SourceBase>(ncDataLists: NCDataList[], op: any): SourceType[] => {
    //console.log(`ncDataLists: ${Stringify2(ncDataLists)}`);
    if (_.isEmpty(ncDataLists[0])) return [];
    let sourceLists = buildUseSourcesLists(ncDataLists);
    //console.log(`buildUseSourcesLists: ${Stringify2(sourceLists)}`);
    return mergeCompatibleUseSources<SourceType>(sourceLists, op);
};

//
// nested loops over XorSources, OrSources primaryNameSrcLists,
// looking for compatible lists
//
let mergeOrSourcesList = (sourceList: XorSource[], orSourceList: OrSource[]): UseSource[] => {
    // NOTE: optimization, can be implemented with separate loop, 
    // (can start with LAST item in list as that should be the one with all
    // --or options, and if that fails, we can bail)
    let mergedSourcesList: UseSource[] = [];
    for (let sources of sourceList) {
        for (let orSources of orSourceList) {
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
                    orSourceLists: orSources.sourceLists,  // yeah this terminology will confuse pretty much anymore
                    orSourcesNcCsvList: orSources.sourceNcCsvList
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
let getCompatibleUseSourcesFromNcData = (args: any): UseSource[] => {
    // XOR first
    let sourceList = getUseSourcesList<XorSource>(args.allXorNcDataLists, Op.xor);
    //console.log(`xorSourceList(${xorSourceList.length): ${Stringify2(xorSourceList)}`);

    // OR next
    let orSourceList = getUseSourcesList<OrSource>(args.allOrNcDataLists, Op.or);
    //console.log(`orSourceList(${orSourceList.length}) ${Stringify2(orSourceList)}`);

    // final: merge or with xor
    if (!_.isEmpty(orSourceList)) {
        sourceList = mergeOrSourcesList(sourceList, orSourceList);
        //console.log(`orSourceList(${orSourceList.length}), mergedSources(${sourceList.length}): ${Stringify2(sourceList)}`);
    }
    console.error(`orSourceList(${orSourceList.length})`);

    return sourceList; // xorSourceList;
};

//
//
let nextIndex = function(clueSourceList: any, sourceIndexes: any): boolean {
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

interface FirstNextResult {
    done: boolean;
    ncList?: NCList;
    nameList?: string[];
}

//
//
let next = (clueSourceList: any, sourceIndexes: number[]): FirstNextResult => {
    for (;;) {
        if (!nextIndex(clueSourceList, sourceIndexes)) {
            return { done: true };
        }
        let ncList: NCList = [];            // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
        let nameList: string[] = [];        // e.g. [ "pollock", "jackson" ]
        let srcCountStrList: string[] = []; // e.g. [ "white,fish:2", "moon,walker:4" ]
        if (!clueSourceList.every((clueSource: any, index: number) => {
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
let first = (clueSourceList: any, sourceIndexes: number[]): FirstNextResult => {
    for (let index = 0; index < clueSourceList.length; ++index) {
        sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;
    return next(clueSourceList, sourceIndexes);
};

//
//
let XX = 0;
let isCompatibleWithOrSources = (sources: SourceData, useSources: UseSource): boolean => {
    // TODO: yes this happens. why I don't know.
    const orSourceLists = useSources.orSourceLists;
    //if (_.isEmpty(orSourceLists)) console.error(`empty orSourceList!`);
    if (!orSourceLists || _.isEmpty(orSourceLists)) {
        // Return false because the caller's presumption is that sources is not
        // wholly compatible with the useSources currently under consideration,
        // and thus called this function seeking an exception due to an orSources
        // match. If there are no orSources that match, we revert to the original
        // presumption (incompatible). The semantics could be made a little cleaner
        // by e.g. returning an empty array here, and a populated array for a match.
        // or maybe just a method name change.
        return false;
    }
    //console.log(`orSourceLists(${orSourceLists.length}): ${Stringify2(orSourceLists)}`);
    for (let [listIndex, orSourceList] of orSourceLists.entries()) {
        let ncCsv = useSources.orSourcesNcCsvList![listIndex];
        if (sources.srcNcMap[ncCsv]) {
            //:: orSources ncCsv matches sources ncCsv

            // essentially, subtract all orSources.primaryNameSrcList entries from from sources.primaryNameSrcList
            // based on primary source. probably not fast.
            // TODO: profile, speed this shit up.
            const orSourcesPrimaryNameSrcList = _.flatten(orSourceList.map(orSources => orSources.primaryNameSrcList));
            const xorPrimaryNameSrcList = _.xorBy(sources.primaryNameSrcList, orSourcesPrimaryNameSrcList, NameCount.count);
            if (XX) {
                console.log(`sources: ${sources.primaryNameSrcList}`);
                console.log(`or: ${orSourcesPrimaryNameSrcList}`);
                console.log(`xor: ${xorPrimaryNameSrcList}`);
                XX = 0;
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
let isCompatibleWithUseSources = (sourceList: SourceList, useSourcesList: SourceList): boolean => {
    if (_.isEmpty(useSourcesList)) return true;
    for (let sources of sourceList) {
        for (let useSources of useSourcesList) {
            const allUnique = allCountUnique(sources.primaryNameSrcList, useSources.primaryNameSrcList);
            if (allUnique || isCompatibleWithOrSources(sources, useSources)) {
                return true;
            }
        }
    }
    return false;
};

// Lazy-load and merge all of the sources in lazySourceList.
//
let loadAndMergeSourceList = (lazySourceList: LazySourceData[]): SourceList => {
    let sourceList: SourceList = []
    for (let lazySource of lazySourceList) {
	if (lazySource.ncList.length !== 2) throw new Error(`lazySource.ncList.length(${lazySource.ncList.length})`);
	if (lazySource.validateResultList.length !== 2) throw new Error(`lazySource.validateResultList.length(${lazySource.validateResultList.length})`);
	let sourcesToMerge: AnySourceData[] = []; // TODO: not ideal, would prefer SourceData here
	for (let index = 0; index < 2; ++index) {
	    sourcesToMerge.push(getSourceData(lazySource.ncList[index], lazySource.validateResultList[index], false));
	}
	sourceList.push(mergeSources(sourcesToMerge[0], sourcesToMerge[1], false) as SourceData);
    }
    return sourceList;
};

//
//
let getCombosForUseNcLists = (sum: number, max: number, args: any): any => {
    let hash: StringAnyMap = {};
    let combos: string[] = [];

    let comboCount = 0;
    let totalVariations = 0;
    let numCacheHits = 0;
    let numMergeIncompatible = 0;
    let numUseIncompatible = 0;
    
    let MILLY = 1000000n;
    let start = process.hrtime.bigint();

    let useSourcesList = args.useSourcesList;
    if (0) console.log(`useSourcesList: ${Stringify2(useSourcesList)}`);

    // for each sourceList in sourceListArray
    ClueManager.getClueSourceListArray({ sum, max }).forEach((clueSourceList: any) => {
        comboCount += 1;

        //console.log(`sum(${sum}) max(${max}) clueSrcList: ${Stringify(clueSourceList)}`);
        let sourceIndexes: number[] = [];
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
            const key: string = result.ncList!.sort().toString();
            let cacheHit = false;
            let sourceList: LazySourceData[];
            if (!hash[key]) {
                sourceList = mergeAllCompatibleSources(result.ncList!, true) as LazySourceData[];
		if (_.isEmpty(sourceList)) ++numMergeIncompatible;
                //console.log(`$$ sources: ${Stringify2(sourceList)}`);
                hash[key] = { sourceList };
            } else {
                sourceList = hash[key].sourceList;
                cacheHit = true;
                numCacheHits += 1;
            }

            if (logging) console.log(`  found compatible sources: ${!_.isEmpty(sourceList)}`);

            // failed to find any compatible combos
            if (_.isEmpty(sourceList)) continue;

            if (_.isUndefined(hash[key].isCompatible)) {
                hash[key].isCompatible = isCompatibleWithUseSources(loadAndMergeSourceList(sourceList), useSourcesList);
            }
            if (hash[key].isCompatible) {
                combos.push(result.nameList!.toString());
            } else if (!cacheHit) {
                numUseIncompatible += 1;
            }
        }
        totalVariations += numVariations;
    });

    let duration = (process.hrtime.bigint() - start) / MILLY;
    Debug(`combos(${comboCount}) variations(${totalVariations}) cacheHits(${numCacheHits}) ` +
	`merge-incompatible(${numMergeIncompatible}) use-incompatible(${numUseIncompatible}) ` +
        `actual(${totalVariations - numCacheHits - numUseIncompatible}) ${duration}ms`);

    if (1) {
        console.error(`combos(${comboCount}) variations(${totalVariations}) cacheHits(${numCacheHits}) ` +
	    `merge-incompatible(${numMergeIncompatible}) use-incompatible(${numUseIncompatible}) ` +
            `actual(${totalVariations - numCacheHits - numUseIncompatible}) ${duration}ms`);
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
let makeCombosForSum = (sum: number, max: number, args: any): any => {
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
let parallel_makeCombosForRange = (first: number, last: number, args: any): any => {
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
    return p.map(entrypoint).then((data: any[]) => {
        //console.log(`data = ${typeof data} array: ${_.isArray(data)}, data[${0}] = ${Stringify(data[0])}`);
        let d = new Duration(beginDate, new Date()).milliseconds;
        console.error(`time: ${PrettyMs(d)} chunks: ${data.length}`);
    });
    // check if range == data and /or if .then(return) passes thru
    //const filterResult = ClueManager.filter(data[i], args.sum, comboMap);
};

//
//
let test = (sum: number, max: number, args: any): any => {
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
        let orSourceLists = useSources.orSourceLists;
        numOrSourcesLists += orSourceLists.length;
        if (!orSourceLists || _.isEmpty(orSourceLists)) continue;
        for (let orSourceList of orSourceLists) {
            numOrSources += orSourceList.length;
            /*
            console.log('----------------');
            for (let orSources of orSourceList) {
                console.log(`${orSources.ncList}`);
            }
            */
        }
    }
    console.error(`useSources(${numUseSources}) orSourceLists(${numOrSourcesLists}) orSources(${numOrSources})`);
};

//
//
let makeCombos = (args: any): any => {
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
        parallel_makeCombosForRange(first, last, args).then((data: any[]) => {
            let comboSet = new Set();
            for (let arr of data) {
                arr.forEach((comboStr: string) => comboSet.add(comboStr));
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
        _.keys(comboMap).forEach((nameCsv: string) => console.log(nameCsv));
        //console.log(`${Stringify(comboMap)}`);
        //process.stderr.write('\n');
    }
    return 1;
};

function getKnownNcListForName (name: string): NCList {
    const countList: number[] = ClueManager.getCountListForName(name);
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

function nameOrNcStrListToKnownNcLists (nameOrNcStrList: string[]): NCList[] {
    return nameOrNcStrList.map(nameOrNcStr => NameCount.makeNew(nameOrNcStr))
        .map(nc => nc.count ? [nc] : getKnownNcListForName(nc.name));
}

function combinationNcList (combo: any, ncLists: NCList[]): NCList {
    return combo.map((ncIndex: number, listIndex: number) => ncLists[listIndex][ncIndex]);
}

function combinationNcDataList (combo: any, ncLists: NCList[]): NCDataList {
    return combo.map((ncIndex: number, listIndex: number) => Object({ ncList: ncLists[listIndex][ncIndex]}));
}

function ncListsToCombinations (ncLists: NCList[]): any {
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]),       // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)
    }).getCombinations()
        .map((combo: any) => combinationNcList(combo, ncLists));
}

function getCombinationNcLists (useArgsList: string[]): NCList[] {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    return useArgsList.map(useArg => useArg.split(','))
        .map(nameOrNcStrList => nameOrNcStrListToKnownNcLists(nameOrNcStrList))
        .map(knownNcLists => ncListsToCombinations(knownNcLists));
}

// This is the exact same method as ncListsToCombinations? except for final map method. could pass as parameter.
function combinationsToNcLists (combinationNcLists: NCList[]): NCList[] {
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]),
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)       // sum of lengths of nclists
    }).getCombinations()
	.map((combo: any) => combinationNcList(combo, combinationNcLists));
}

// TODO: get rid of this and combinationsToNCLists, and add extra map step in buildAllUseNCData
function combinationsToNcDataLists (combinationNcLists: NCList[]): NCDataList[] {
    Debug(`combToNcDataLists() combinationNcLists: ${Stringify(combinationNcLists)}`);
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]),
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)       // sum of lengths of nclists
    }).getCombinations()
	.map((combo: any) => combinationNcDataList(combo, combinationNcLists));
}

function buildAllUseNcLists (useArgsList: string[]): NCList[] {
    return combinationsToNcLists(getCombinationNcLists(useArgsList));
}

function buildAllUseNcDataLists (useArgsList: string[]): NCDataList[] {
    return combinationsToNcDataLists(getCombinationNcLists(useArgsList));
}

//
//
function buildUseNcLists (useArgsList: string[]): NCList[] {
    let useNcLists: NCList[] = [];
    useArgsList.forEach((useArg: string) =>  {
        let args = useArg.split(',');
        let ncList: NCList = args.map(arg => {
            let nc = NameCount.makeNew(arg);
            if (!nc.count) throw new Error(`arg: ${arg} requires a :COUNT`);
            if (!_.has(ClueManager.knownClueMapArray[nc.count], nc.name)) throw new Error(`arg: ${nc} does not exist`);
            return nc;
        });
        useNcLists.push(ncList);
    });
    return useNcLists;
}

module.exports = {
    makeCombos,
    makeCombosForSum
};
