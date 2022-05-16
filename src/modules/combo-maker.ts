//
// combo-maker.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const BootstrapComboMaker = require('../../modules/bootstrap-combo-maker');
const ResultMap   = require('../../types/result-map');
const Peco        = require('../../modules/peco');
const Log         = require('../../modules/log')('combo-maker');

const Assert      = require('assert');
const Debug       = require('debug')('combo-maker');
const Duration    = require('duration');
const Expect      = require('should/as-function');
const OS          = require('os');
const Parallel    = require('paralleljs');
const PrettyMs    = require('pretty-ms');
const stringify   = require('javascript-stringify').stringify;
const Stringify2  = require('stringify-object');

import * as Clue from '../types/clue';
import * as ClueManager from './clue-manager';
import * as NameCount from '../types/name-count';
import { ValidateResult } from './validator';

interface StringBoolMap {
    [key: string]: boolean; // for now; eventually maybe array of string (sorted primary nameSrcCsv)
}

//
//
interface NCData {
    ncList: NameCount.List;
}
type NCDataList = NCData[];

// TODO: renaming this "Source" stuff would be welcome. isn't it close to ValidateResult?

//
// TODO: find case where we use this without ncList and comment why ncList isn't here
//       even though all derived interfaces have an ncList.
//
interface SourceBase {
    primaryNameSrcList: NameCount.List;
}

//
//
interface LazySourceData extends SourceBase {
    ncList: NameCount.List;
    validateResultList: ValidateResult[];
}

//
//
interface SourceData extends SourceBase {
    ncList: NameCount.List;
    sourceNcCsvList: string[];
    ncCsv?: string;
}
type SourceList = SourceData[];
type AnySourceData = LazySourceData | SourceData;


/* TODO:
interface LazySourceHashEntry { 
    sourceList: LazySourceData[];
    isCompatible: boolean;
}

interface LazySourceHashMap {
    [key: string]: LazySourceHashEntry;
}

replaces: StringAnyMap
*/

interface StringAnyMap {
    [key: string]: any;
}

type MergeArgs = {
    min_synonyms: number;
    max_synonyms: number;
    lazy: boolean | undefined;
};

//
//

type CountArray = Int32Array;

interface CountArrayAndSize {
    array: CountArray;
    size: number;
}

interface UseSourceBase extends SourceBase {
    primarySrcArray: CountArray;
    ncList: NameCount.List;
}

interface XorSource extends UseSourceBase {
}

type SourceNcCsvMap = Record<string, number[]>;

interface OrSource extends UseSourceBase {
    sourceLists: SourceList[];
    sourceNcCsvMap: SourceNcCsvMap;
    primarySrcArrayAndSizeList: CountArrayAndSize[];
}

interface UseSource extends UseSourceBase {
    orSource?: OrSource;
}

interface PreComputedData {
    orSourcesNcCsvMap: Map<string, number>;
    useSourcesList: UseSource[];
}

let PCD: PreComputedData | undefined = undefined;

//
//
function Stringify(val: any) {
    return stringify(val, (value: any, indent: any, stringify: any) => {
        if (typeof value == 'function') return "function";
        return stringify(value);
    }, " ");
}

let logging = 0;
let loggy = false;
let ZZ = false;
let WW = false;
let AA = false;


// TODO: as const;
const Op = {
    and: 1,
    or:  2,
    xor: 3
};
Object.freeze(Op);

function OpName (opValue: number): string | undefined {
    return _.findKey(Op, (v: number) => opValue === v);
}

//
// see: showNcLists
let listOfNcListsToString = (listOfNcLists: NameCount.List[]): string => {
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
        result += `    sourcNcCsvList: ${Stringify2(source.sourceNcCsvList)}\n`;
        result += '  }';
    }
    return result + "\n]";
};

function showNcLists (ncLists: NameCount.List[]): string {
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
let allCountUnique = (nameSrcList1: NameCount.List, nameSrcList2: NameCount.List): boolean => {
    let set: Set<number> = new Set<number>();
    for (let nameSrc of nameSrcList1) {
        set.add(nameSrc.count);
    }
    // TODO: some
    for (let nameSrc of nameSrcList2) {
        if (set.has(nameSrc.count)) return false;
    }
    return true;
};

// 
//
let anyCountInArray = (ncList: NameCount.List, countArray: CountArray): boolean => {
    return ncList.some(nc => countArray[nc.count] === nc.count);
};

//
//
let anyNumberInArray = (numberList: number[], countArray: CountArray): boolean => {
    return numberList.some(num => countArray[num] === num);
};

//
//
let getCountArray = (ncList: NameCount.List): CountArray => {
    return ncList.reduce((array, nc) => {
        array[nc.count] = nc.count;
        return array;
    }, new Int32Array(ClueManager.getNumPrimarySources()));
};

//
//
let getCountArrayAndSize = (sourceList: SourceList): CountArrayAndSize => {
    // NO: reduce
    let array = new Int32Array(ClueManager.getNumPrimarySources());
    let size = 0;
    for (let source of sourceList) {
        for (let nc of source.primaryNameSrcList) {
            array[nc.count] = nc.count;
        }
        size += source.primaryNameSrcList.length;
    }
    return { array, size };
};

//
//
let getCountListNotInArray = (ncList: NameCount.List, countArray: CountArray): number[] => {
    return ncList.map(nc => nc.count).filter(count => countArray[count] !== count);
};

//
//
let  getNumCountsInArray = (ncList: NameCount.List, countArray: CountArray): number => {
    // TODO: reduce
    let count = 0;
    for (let nc of ncList) {
        if (countArray[nc.count] === nc.count) ++count;
    }
    return count;
};

//
//
/*
let noCountsNotInOneAreInTwo = (ncList: NameCount.List, xorCountArrayAndSize: CountArrayAndSize, uniqCountArray: CountArray): boolean => {
    let xorCount = 0;
    for (let nc of ncList) {
        if (xorCountArrayAndSize.array[nc.count] === nc.count) {
            ++xorCount;
        } else if (uniqCountArray[nc.count] === nc.count) {
            return false;
        }
    }
    return xorCount === xorCountArrayAndSize.size;
}
*/

// key types:
//{
// A:
//  'jack:3': {             // non-array object value type
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
let recursiveAddSrcNcLists = (list: string[], resultMap: any, top = true): string[] => {
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
            list.push(sortedKeys);
            //obj.map[sortedKeys] = true;
        }
        keys.forEach(key => {
            // push individual keys
            list.push(key);
            //obj.map[key] = true;
            let val = resultMap[key];
            if (val && !_.isArray(val)) {
                recursiveAddSrcNcLists(list, val, false);
            }
        });
    }
    return list;
};

// NOTE: resultMap here is a not actually a ResultMap, it's a resultMap.map().
//
function buildSrcNcList (resultMap: Object): string[] {
    return recursiveAddSrcNcLists([], resultMap);
}

let PP = false;

//
//
let populateSourceData = (lazySource: SourceBase, nc: NameCount.Type, validateResult: ValidateResult,
                          orSourcesNcCsvMap?: Map<string, number>): SourceData => {
    let source: SourceData = lazySource /*as SourceBase*/ as SourceData;
    if (validateResult.resultMap) {
        source.sourceNcCsvList = buildSrcNcList(validateResult.resultMap.map());
    } else {
        Assert(validateResult.ncList.length === 1 && validateResult.ncList[0].count === 1, 'wrong assumption');
        source.sourceNcCsvList = [NameCount.listToString(validateResult.ncList)];
    }
    if (nc.count > 1) {
        source.sourceNcCsvList.push(NameCount.toString(nc));
    }
    if (PP) console.log(`before: ${Stringify2(source.sourceNcCsvList)}`);
    if (orSourcesNcCsvMap) {
        source.sourceNcCsvList = source.sourceNcCsvList.filter(ncCsv => orSourcesNcCsvMap.has(ncCsv));
        //.filter(ncCsv => orSourcesNcCsvMap ? orSourcesNcCsvMap.has(ncCsv) : true);
    }
    if (PP) console.log(`after: ${Stringify2(source.sourceNcCsvList)}`);

    source.ncList = [nc]; // TODO i could try getting rid of "LazySource.nc" and just make this part of LazySouceData
    if (loggy || logging > 3) {
        console.log(`getSourceList() ncList: ${source.ncList}, sourceNcCsvList: ${source.sourceNcCsvList}`);
        if (_.isEmpty(source.sourceNcCsvList)) console.log(`empty sourceNcCsvList: ${Stringify(validateResult.resultMap.map())}`);
    }
    return source;
};

//
//
let getSourceData = (nc: NameCount.Type, validateResult: ValidateResult, lazy: boolean | undefined,
                     orSourcesNcCsvMap?: Map<string, number>): AnySourceData => {
    const primaryNameSrcList: NameCount.List = validateResult.nameSrcList;
    return lazy
        ? { primaryNameSrcList, ncList: [nc], validateResultList: [validateResult] }
        : populateSourceData({ primaryNameSrcList }, nc, validateResult, orSourcesNcCsvMap);
};

let oob = 0;

//
//
let filterPropertyCountsOutOfBounds = (result: ValidateResult, args: MergeArgs): boolean => {
    let propertyCounts: Clue.PropertyCounts.Map;
    if (result.nameSrcList.length === 1) {
        // primary clue: propertyCounts are attached to clue
        const nameSrc = result.nameSrcList[0];
        const clue = _.find(ClueManager.getClueList(1), { name: nameSrc.name, src: _.toString(nameSrc.count) }) as Clue.Primary;
        propertyCounts = clue.propertyCounts!;
    } else {
        // compoundClue: propertyCounts are attached to each ValidateResult
        propertyCounts = result.propertyCounts!;
    }

    if (propertyCounts[Clue.PropertyName.Synonym].total > args.max_synonyms) {
        oob++;
        return false;
    }
    return true;
};

//
//
let getSourceList = (nc: NameCount.Type, args: MergeArgs): AnySourceData[] => {
    const sourceList: AnySourceData[] = [];
    ClueManager.getKnownSourceMapEntries(nc)
        .forEach(entry => {
            sourceList.push(...entry.results
                .filter((result: ValidateResult) => filterPropertyCountsOutOfBounds(result, args))
                .map((result: ValidateResult) => getSourceData(nc, result, args.lazy)));
        });
    if (AA) {
        console.log(`getSourceList ${NameCount.toString(nc)} (${sourceList.length}):`);
        for (let source of sourceList) console.log(` ncList: ${NameCount.listToString(source.ncList)}`);
    }
    return sourceList;
};

//
//
let mergeSources = (source1: AnySourceData, source2: AnySourceData, lazy: boolean | undefined): AnySourceData => {
    let primaryNameSrcList = [...source1.primaryNameSrcList, ...source2.primaryNameSrcList];
    let ncList = [...source1.ncList, ...source2.ncList];
    if (lazy) {
        Assert(ncList.length === 2, `ncList.length(${ncList.length})`);
        let result: LazySourceData = {
            primaryNameSrcList,
            ncList,
            validateResultList: [
                (source1 as LazySourceData).validateResultList[0],
                (source2 as LazySourceData).validateResultList[0]
            ]
        };
        return result;
    }
    source1 = source1 as SourceData;
    source2 = source2 as SourceData;
    let mergedSource: SourceData = {
        primaryNameSrcList,
        ncList,
        sourceNcCsvList: [...source1.sourceNcCsvList, ...source2.sourceNcCsvList]
    };
    // TODO: still used?
    mergedSource.ncCsv = NameCount.listToSortedString(mergedSource.ncList);
    return mergedSource;
};

//
//
let mergeCompatibleSources = (source1: AnySourceData, source2: AnySourceData, args: MergeArgs): AnySourceData[] => {
    // TODO: this logic could be part of mergeSources
    // also, uh, isn't there are primarySrcArray I can be using here?
    return allCountUnique(source1.primaryNameSrcList, source2.primaryNameSrcList)
        ? [mergeSources(source1, source2, args.lazy)]
        : [];
};

//
//
let mergeCompatibleSourceLists = (sourceList1: AnySourceData[], sourceList2: AnySourceData[], args: MergeArgs): AnySourceData[] => {
    let mergedSourcesList: AnySourceData[] = [];
    // TODO: reduce
    for (const source1 of sourceList1) {
        for (const source2 of sourceList2) {
            mergedSourcesList.push(...mergeCompatibleSources(source1, source2, args));
        }
    }
    return mergedSourcesList;
};

//
//
let mergeAllCompatibleSources = (ncList: NameCount.List, args: MergeArgs): AnySourceData[] => {
    // because **maybe** broken for > 2 below
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    // TODO: reduce (or some) here
    let sourceList = getSourceList(ncList[0], args);
    for (let ncIndex = 1; ncIndex < ncList.length; ncIndex += 1) {
        const nextSourceList = getSourceList(ncList[ncIndex], args);
        sourceList = mergeCompatibleSourceLists(sourceList, nextSourceList, args);
        // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
        if (_.isEmpty(sourceList)) break;
    }
    return sourceList;
};

//
//
let buildSourceListsForUseNcData = (useNcDataLists: NCDataList[], args: MergeArgs): SourceList[] => {
    let sourceLists: SourceList[] = [];
    let hashList: StringBoolMap[] = [];
    // TODO: forEach
    for (let [dataListIndex, useNcDataList] of useNcDataLists.entries()) {
        for (let [sourceListIndex, useNcData] of useNcDataList.entries()) {
            if (!sourceLists[sourceListIndex]) sourceLists.push([]);
            if (!hashList[sourceListIndex]) hashList.push({});
            let sourceList = mergeAllCompatibleSources(useNcData.ncList, args);
            for (let source of sourceList) {
                let key = NameCount.listToString(_.sortBy(source.primaryNameSrcList, NameCount.count));
                if (!hashList[sourceListIndex][key]) {
                    sourceLists[sourceListIndex].push(source as SourceData);
                    hashList[sourceListIndex][key] = true;
                }
            }
        }
    }
    return sourceLists;
};

//
// NOTE MUCH OF THIS COULD BE PRECOMPUTED, SPECIFICALLY WHICH --OR SOURCE COMBINATIONS
// ARE COMPATIBLE. THOSE COULD BE PRECOMPUTED ONCE. IN FACT, BY DOING SO, WE COULD ALERT
// THE USER TO INCOMPATIBLE --OR SOURCE COMBINATIONS.
// I SHOULD COUNT HOW MANY TIME THIS IS CALLED IN THE PRECOMPUTE PHASE. IF I WANT TO
// SPEED UP PRECOMPUTE, I COULD PROBABLY ELIMINATE A LOT OF CALLS HERE.
//
// Here primaryNameSrcList is the combined compatible primary NameSrc's of one or more
// --or arguments.
//
// And orSourceLists is a list of sourceLists: one list for each --or argument that is
// *not* included in primaryNameSrcList.
//          
// Whittle down these latter lists of separate --or argument sources into lists of combined
// compatible sources, with each result list containing one element from each source list.
//
// So for [ [a, b], [b, d] ] the candidates would be [a, b], [a, d], [b, b], [b, d],
// and [b, b] would be filtered out as incompatible.
//
let getCompatibleOrSourceLists = (primaryNameSrcList: NameCount.List, orSourceLists: SourceList[]): SourceList[] => {
    if (_.isEmpty(orSourceLists)) return [];

    let listArray = orSourceLists.map(sourceList => [...Array(sourceList.length).keys()]);
    let peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    
    let sourceLists: SourceList[] = [];
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
        let sourceList: SourceList = [];
        let primarySrcSet: Set<number> = new Set(primaryNameSrcList.map(nameSrc => nameSrc.count));
        // TODO: list.some(), see below
        for (let [listIndex, sourceIndex] of indexList.entries()) {
            let source = orSourceLists[listIndex][sourceIndex];
            // TODO: if (!allArrayElemsInSet(sources.primaryNameSrcList, primarySrcSet))
            let prevSetSize = primarySrcSet.size;
            // TODO: set.addAll(source.primaryNameSrcList.map(nameSrc => nameSrc.count));?
            for (let nameSrc of source.primaryNameSrcList) {
                primarySrcSet.add(nameSrc.count);
            }
            if (primarySrcSet.size !== prevSetSize + source.primaryNameSrcList.length) {
                // TODO: list.some() likely fixes this abomination
                sourceList = [];
                break;
            }
            sourceList.push(source);
        }
        if (!_.isEmpty(sourceList)) {
            sourceLists.push(sourceList);
        }
    }
    return sourceLists;
};

// Here we have 'orSourceLists', created from getUseSourcesList(Op.or).
//
// Generate a sorted ncCsv using the combined NCs of each sources's ncList,
// in each sourceList. Return a map of ncCsvs : [sourceList index(es)].
//
// Exmample sourceList's ncList's, stringified: [ [ b:1, a:2 ], [ c:3, d:4 ] ]
// Flattened ncList, stringified: [ b:1, a:2, c:3, d:4 ]
// sorted ncCsv: 'a:2,b:1,c:3,d:4'
//
// It'd be preferable to embed this ncCsv within each sourceList itself. I'd need to
// wrap it in an object like { sourceList, ncCsv }.
//
let buildOrSourceNcCsvMap = (orSourceLists: SourceList[]): SourceNcCsvMap => {
    return orSourceLists.reduce((map: SourceNcCsvMap, sourceList: SourceList, index: number) => {
        // ++INSTRUMENTATION
        if (0) {
            let text = `orSourceList(${index}):`;
            for (const source of sourceList) {
                text += ` [${NameCount.listToSortedString(source.ncList)}]`;
            }
            console.log(text);
        }
        // --INSTRUMENTATION

        const key = NameCount.listToSortedString(_.flatMap(sourceList, source => source.ncList));
        if (!map[key]) map[key] = [];
        map[key].push(index);
        return map;
    }, {});
};

//
//
let buildUseSourceNcCsvMap = (useSourceList: UseSource[]): SourceNcCsvMap => {
    return useSourceList.reduce((map: SourceNcCsvMap, source: UseSource, index: number) => {
        if (AA) console.log(Stringify2(source));
        const key = NameCount.listToSortedString(source.ncList);
        if (!map[key]) map[key] = [];
        map[key].push(index);
        return map;
    }, {});
};

//
// TODO: questionable how well I'm using the parameterized type here
//
let mergeCompatibleUseSources = <SourceType extends UseSourceBase>(sourceLists: SourceList[], op: any, args: MergeArgs): SourceType[] => {
    // TODO: sometimes a sourceList is empty, like if doing $(cat required) with a
    // low clue count range (e.g. -c2,4). should that even be allowed?
    let pad = (op === Op.or) ? 1 : 0;
    let listArray = sourceLists.map(sourceList => [...Array(sourceList.length + pad).keys()]);
    ZZ = (op === Op.or);
    if (ZZ) console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourceLists(${sourceLists.length}): ${Stringify2(sourceLists)}`);

    let peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    
    let iter = 0;
    let sourceList: SourceType[] = [];
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
        //
        // TODO: list of sourceLists outside of this loop. 
        // assign result.sourceLists inside indexList.entries() loop. 
        //
        let primaryNameSrcList: NameCount.List = [];
        let ncList: NameCount.List = []; // TODO: xor only
        let orSourceLists: SourceList[] = [];
        let compatible = true;
        //ZZ = indexList.every(index => index === 0);
        if (ZZ) console.log(`iter (${iter}), indexList: ${stringify(indexList)}`);
        // TODO: indexList.some()
        for (let [sourceListIndex, sourceIndex] of indexList.entries()) {
            // TODO: move inside op === Op.or block?
            if (!orSourceLists[sourceListIndex]) orSourceLists.push([]);
            const orSourceList = orSourceLists[sourceListIndex];
            if (ZZ) console.log(`sourceListIndex(${sourceListIndex}) sourceIndex(${sourceIndex}), orSourceList(${orSourceList.length})`);
            // TODO: Here be dragons. To introduce some semblance of sanity here, perhaps I should add a
            // initializeOrSourceListsForMerge(sourceLists, orSourceLists, indexList);
            // as initializer to orSourceLists if (op === Op.or)
            // no, we can't just initialize this at time of declaration, it's dependent on sourceIndex
            // so must be initialized inside loop. could still be a separate function though
            // child function?
            if (op === Op.or) {
                if (sourceIndex === 0) {
                    if (ZZ) {
                        console.log(`adding to orSourceLists @ index(${sourceListIndex}) length(${orSourceList.length}) count(${sourceLists[sourceListIndex].length})`);
                        console.log(`  sourceLists[${sourceListIndex}][0].ncList: ${NameCount.listToString(sourceLists[sourceListIndex][0].ncList)}`);
                    }
                    orSourceLists.push(sourceLists[sourceListIndex]);
                    // TODO: This appears it is being done way too many times in this nested loop, and I don't see
                    // how the data ever changes. It could be done once outside of all loops.
                    // Questionable whether it even belongs in this function honestly
                    sourceLists[sourceListIndex].forEach(source => {
                        source.ncCsv = NameCount.listToSortedString(source.ncList);
                    });
                    continue;
                }
                --sourceIndex;
            }
            const source = sourceLists[sourceListIndex][sourceIndex];
            if (_.isEmpty(primaryNameSrcList)) {
                primaryNameSrcList.push(...source.primaryNameSrcList);
                ncList.push(...source.ncList);
                if (ZZ) console.log(`pnsl, initial: ${primaryNameSrcList}`);
            } else {
                // TODO: hash of primary sources would be faster here. inside inner loop.
                let combinedNameSrcList = primaryNameSrcList.concat(source.primaryNameSrcList);
                // TODO: uniqBy da debil
                if (_.uniqBy(combinedNameSrcList, NameCount.count).length === combinedNameSrcList.length) {
                    primaryNameSrcList = combinedNameSrcList;
                    ncList.push(...source.ncList);
                    if (ZZ) console.log(`pnsl, combined: ${primaryNameSrcList}`);
                } else {
                    if (ZZ) console.log(`pnsl, emptied: ${primaryNameSrcList}`);
                    compatible = false;
                    break;
                }
            }
        }
        if (compatible) {
            if (0 || ZZ) console.log(`pnsl, final: ${primaryNameSrcList}, indexList(${indexList.length}): [${indexList}]`);
            if (ZZ && _.isEmpty(primaryNameSrcList)) console.log(`empty pnsl`);

            //Assert(!_.isEmpty(primaryNameSrcList), 'empty primaryNameSrcList');
            let result: UseSourceBase = {
                primaryNameSrcList,
                primarySrcArray: getCountArray(primaryNameSrcList),
                ncList
            };
            if (op === Op.or) {
                const orResult: OrSource = result as OrSource;
                // TODO: remind me, why are there empty orSourceLists to begin with?
                const nonEmptyOrSourceLists = orSourceLists.filter(sourceList => !_.isEmpty(sourceList));

                // ++DEBUG
                //console.log(`orSourceLists(${orSourceLists.length})`);
                if (ZZ && nonEmptyOrSourceLists.length !== orSourceLists.length) {
                    console.log(`  nonEmpty(${nonEmptyOrSourceLists.length})` +
                        `, empty?(${orSourceLists.length - nonEmptyOrSourceLists.length})`);
                }
                // --DEBUG
                
                // This can return an empty list and that should seem odd to you at first glance,
                // because all of these OrSources were already determined to be compatible with
                // each other.
                //
                // When you throw primaryNameSrcList into the equation, it is possible that the
                // the primary sources it uses results in no valid combination of the remaining
                // OrSources, because they can no longer use sources in use by primaryNameSrcList.
                const compatibleOrSourceLists = getCompatibleOrSourceLists(primaryNameSrcList, nonEmptyOrSourceLists);
                compatible = _.isEmpty(nonEmptyOrSourceLists) || !_.isEmpty(compatibleOrSourceLists);
                if (compatible) {
                    orResult.sourceLists = compatibleOrSourceLists;
                    orResult.sourceNcCsvMap = buildOrSourceNcCsvMap(orResult.sourceLists);
                    orResult.primarySrcArrayAndSizeList = orResult.sourceLists.map(sourceList => getCountArrayAndSize(sourceList));
                    if (ZZ && _.isEmpty(orResult.sourceLists) && !_.isEmpty(nonEmptyOrSourceLists)) {
                        console.log(`before: orSourceLists(${orSourceLists.length}): ${Stringify2(orSourceLists)}`);
                        console.log(`after: result.sourceLists(${orResult.sourceLists.length}): ${Stringify2(orResult.sourceLists)}`);
                        ZZ = false;
                    }
                }
            }
            if (compatible) {
                sourceList.push(result as SourceType);
            }
        }
        ++iter;
    }
    return sourceList;
};

//
//
let getUseSourcesList = <SourceType extends UseSourceBase>(ncDataLists: NCDataList[], op: any, args: any): SourceType[] => {
    if (_.isEmpty(ncDataLists[0])) return [];
    const mergeArgs: MergeArgs = {
        min_synonyms: args.min_synonyms,
        max_synonyms: args.max_synonyms,
        lazy: false
    };
    const sourceLists = buildSourceListsForUseNcData(ncDataLists, mergeArgs);
    return mergeCompatibleUseSources<SourceType>(sourceLists, op, mergeArgs);
};

//
// nested loops over XorSources, OrSources primaryNameSrcLists,
// looking for compatible lists
//
let mergeOrSourceList = (sourceList: XorSource[], orSourceList: OrSource[]): UseSource[] => {
    // NOTE: optimization, can be implemented with separate loop, 
    // (can start with LAST item in list as that should be the one with all
    // --or options, and if that fails, we can bail)
    let mergedSourceList: UseSource[] = [];
    for (let source of sourceList) {
        for (let orSource of orSourceList) {
            //
            // TODO:  call mergeCompatibleSources.  still? or..
            //
            let combinedNameSrcList = source.primaryNameSrcList.concat(orSource.primaryNameSrcList);
            // 
            // TODO: hash of primary sources faster here?
            //
            // possible faulty (rarish) optimization, only checking clue count
            // TODO: uniqBy da debil
            let numUnique = _.uniqBy(combinedNameSrcList, NameCount.count).length;
            if (numUnique === combinedNameSrcList.length) {
                mergedSourceList.push({
                    primaryNameSrcList: combinedNameSrcList,
                    primarySrcArray: getCountArray(combinedNameSrcList),
                    ncList: source.ncList.concat(orSource.ncList),
                    orSource
                });
            } else if (numUnique === orSource.primaryNameSrcList.length) {
                console.error('an --or value is implicitly compatible with an --xor value, making this --or value unnecessary');
            } else if (0) {
                console.error(`not unique, source: ${NameCount.listToString(source.primaryNameSrcList)}, ` +
                              `orSource: ${NameCount.listToString(orSource.primaryNameSrcList)}`);
            }
        }
    }
    return mergedSourceList;
};

//
//
let nextIndex = function(countList: number[], sourceIndexes: number[]): boolean {
    // increment last index
    let index = sourceIndexes.length - 1;
    ++sourceIndexes[index];

    // while last index is maxed: reset to zero, increment next-to-last index, etc.
    while (sourceIndexes[index] === ClueManager.getClueList(countList[index]).length) { // clueSourceList[index].list.length) {
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
    ncList?: NameCount.List;
    nameList?: string[];
}

//
//
let next = (countList: number[], sourceIndexes: number[]): FirstNextResult => {
    for (;;) {
        if (!nextIndex(countList, sourceIndexes)) {
            return { done: true };
        }
        let ncList: NameCount.List = [];    // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
        let nameList: string[] = [];        // e.g. [ "pollock", "jackson" ]
        let srcCountStrList: string[] = []; // e.g. [ "white,fish:2", "moon,walker:4" ]
        if (!countList.every((count, index) => {
            let clue = ClueManager.getClueList(count)[sourceIndexes[index]];
            if (clue.ignore || clue.skip) {
                return false; // every.exit
            }
            nameList.push(clue.name);
            // TODO: to remove NameCount.makeNew here, we must add and call
            // NameCount.sort(list: NameCount.List) below
            ncList.push(NameCount.makeNew(clue.name, count));
            srcCountStrList.push(NameCount.makeCanonicalName(clue.src, count));
            return true; // every.continue;
        })) {
            continue;
        }
        nameList.sort();
        // TODO: NameCount.sort(ncList), in sortBy func: convert NC to string
        // (NameCount.toString(nc: Type), use the standard string compare fn.
        NameCount.sortList(ncList);
        return { done: false, ncList, nameList };
    }
};

//
//
let first = (countList: number[], sourceIndexes: number[]): FirstNextResult => {
    // TODO: _.fill?
    for (let index = 0; index < countList.length; ++index) {
        sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;
    return next(countList, sourceIndexes);
};
 
//
//                
const isSourceANDCompatibleWithOrSource = (source: SourceData, orSourceData: OrSource,
                                           orSourceIndex: number): boolean => {
    // check if all primary source counts of orSource are included in source
    const primarySrcArrayAndSize = orSourceData.primarySrcArrayAndSizeList[orSourceIndex];
    const numCountsInArray = getNumCountsInArray(source.primaryNameSrcList, primarySrcArrayAndSize.array);
    const result = numCountsInArray === primarySrcArrayAndSize.size;
    if (AA) console.log(` AND compatible: ${result}, matching counts (${numCountsInArray}) out of (${primarySrcArrayAndSize.size})`);
    return result;
}

//
//                
const isSourceXORCompatibleWithOrSource = (source: SourceData, orSourceData: OrSource): boolean => {
    // comment
    const countList = NameCount.listToCountList(source.primaryNameSrcList);
    if (AA) console.log(` start: (${source.primaryNameSrcList.length}), ${NameCount.listToStringList(source.primaryNameSrcList)}`);
    const result = orSourceData.sourceLists.some(sourceList => {
        let countSet = new Set(countList);
        let size = countSet.size;
        for (const orSource of sourceList) {
            if (AA) {
                console.log(` before add: actual(${countSet.size}), expected(${size}), adding(${orSource.primaryNameSrcList.length})` +
                    `, ncList: ${NameCount.listToStringList(orSource.primaryNameSrcList)}`);
            }
            orSource.primaryNameSrcList.forEach(nameSrc => countSet.add(nameSrc.count));
            size += orSource.primaryNameSrcList.length;
            if (AA) console.log(` after add: actual(${countSet.size}), expected(${size})`);
        }
        return countSet.size === size;
    });
    if (AA) console.log(` XOR compatible: ${result}, sourceLists(${orSourceData.sourceLists.length})`);
    return result;
};

//
//
let isSourceCompatibleWithAnyOrSource = (source: SourceData, useSource: UseSource,
                                         orSourcesNcCsvMap: Map<string, number>): boolean => {
    const orSource = useSource.orSource!;
    if (AA) {
        console.log(` source.sourceNcCsvList(${source.sourceNcCsvList.length})`);
        if (1) console.log(` orSource: ${Stringify2(orSource)}`);
    }

    // source.sourceNcCsvList is a list of ncCsvs that represent all of the component sources
    // of a particular source, like a flattened ResultMap, as well the source itself.
    //
    // Example: ['bat:3','night:2,dark:1','night:2','eden:1,woman:1','eden:1','woman:1']
    //
    // orSource.sourceNcCsvMap maps ncCsvs to the  ????????
    //
    // It seems to me like there are two (or three) problems here.
    //
    // One, is that the logic seems to be "match any of the first to any of the second".
    //    This seems wrong. I don't think the outer .some can be used here. Confirm that,
    //    for example, a match on "eden:1" wouldn't succeed, or if it does, how does that
    //    confirm compatibility?
    //
    // Two, source.sourceNcCsvList is often empty and returning false. Shouldn't it be true?
    // Three, orSource.sourceNcCsvMap[ncCsv] check is returning false. Shouldn't it be true?
    //
    // One: Fixed I think.
    // Two: Worth revisiting.
    // Three: Worth revisiting
    // 
    return source.sourceNcCsvList.some(ncCsv => {
        // TODO: comment this
        if (!orSource.sourceNcCsvMap[ncCsv]) {
            if (AA) console.log( `  no ${ncCsv} in orSourceMap`);
            return false; // some.continue
        }
        if (AA) console.log(` found ${ncCsv} in map: [${_.toString(orSource.sourceNcCsvMap[ncCsv])}]`);
        return orSource.sourceNcCsvMap[ncCsv].some(index => {
            return isSourceANDCompatibleWithOrSource(source, orSource, index) &&
                isSourceXORCompatibleWithOrSource(source, orSource);
        });
    });
                
    /* this was the old "xor compatible" check.  I think it's wrong.  leaving it for until
       I'm more I know what I'm doing.
        // take the remaining primary source counts from source that are not part of orSource,
        // and ensure none of them are present in useSource.
        // TODO: I'm actually not sure why I need to do this check, because it seems this is
        // the first compatibility check in isSourceCompatibleWithUseSoruce. But I'm not going
        // to second guess myself without more investigation.
        const uniqPrimarySrcList = getCountListNotInArray(source.primaryNameSrcList, primarySrcArrayAndSize.array);
        const allUniqueSrc = !anyNumberInArray(uniqPrimarySrcList, useSource.primarySrcArray);
        if (AA) console.log(` allUnique: ${allUniqueSrc}`);
        if (allUniqueSrc) return true; // some.exit
        return false; // some.continue
    */
};

//
//
let isSourceCompatibleWithUseSource = (source: SourceData, useSource: UseSource,
                                       orSourcesNcCsvMap: Map<string, number>): boolean => {
    let compatible: boolean =
        !anyCountInArray(source.primaryNameSrcList, useSource.primarySrcArray);
    if (AA) console.log(` -xorCompatible: ${compatible}, orSourceLists(${useSource.orSource?.sourceLists.length})`);

    // Base case: --xor arguments only. Wether we're compatible or not is known at this point.
    // --or case: if orSource.sourceLists is non-empty, we must check them for --or compatibility.
    if (compatible) {
        // ++DEBUG
        //const ncStr: string = source.sourceNcCsvList[source.sourceNcCsvList.length - 1];
        const ncStr: string = NameCount.listToString(source.ncList);
        //WW = ncStr === 'buffalo spring:1,not:1';
        if (0 && WW) console.log(`${ncStr} orSource: ${Stringify2(useSource.orSource)}`);
        // --DEBUG

        if (useSource.orSource) { // ?.sourceLists.length) {
            compatible = isSourceCompatibleWithAnyOrSource(source, useSource, orSourcesNcCsvMap);
            if (AA) console.log(` -orCompatible: ${compatible}, ${ncStr}`);
        }
    }

    WW = false;

    return compatible;
}

let isCWUS_calls = 0;
let isCWUS_comps = 0;
type CwusHashEntry = {
    count: number;
//    useSource: SourceData;
};

let CwusHash: Record<string, CwusHashEntry> = {};

// TODO: isAnySourceCompatibleWithAnyUseSource

// The inner loop of this function can be executed a billion+ times with
// a reasonably sized --xor list. About 45k outer loops and 25k inner.
//
let isCompatibleWithUseSources = (sourceList: SourceList, useSourceList: UseSource[],
                                  orSourcesNcCsvMap: Map<string, number>): boolean => {
    if (_.isEmpty(useSourceList)) return true;
    isCWUS_calls += 1;

    /*
    // run through previously compatible useSources in hot-hash first
    for (const entry of Object.entries(CwusHash)) {
        let [sourceIndex, useSourceIndex] = entry[0].split(',').map(str => _.toNumber(str));
        if (sourceIndex === 3 && useSourceIndex === 3) {
            console.log(Object.keys(sourceList[sourceIndex]));
        }

        if (isSourceCompatibleWithUseSource(sourceList[sourceIndex], useSourceList[useSourceIndex], orSourcesNcCsvMap)) {
            CwusHash[entry[0]].count += 1;
            return true;
        }
    }
    */

    // TODO: some
    for (let sourceIndex = 0, sourceListLength = sourceList.length; sourceIndex < sourceListLength; ++sourceIndex) {
        let source = sourceList[sourceIndex];
        if (AA) console.log(`source nc: ${NameCount.listToString(source.ncList)}`);
        for (let useSourceIndex = 0, useSourceListLength = useSourceList.length; useSourceIndex < useSourceListLength; ++useSourceIndex) {
            let useSource = useSourceList[useSourceIndex];
            isCWUS_comps += 1;
            if (isSourceCompatibleWithUseSource(source, useSource, orSourcesNcCsvMap)) {
                if (!CwusHash[useSourceIndex]) {
                    CwusHash[useSourceIndex] = { count: 0 }; // , useSource };
                }
                CwusHash[useSourceIndex].count += 1;
                return true;
            }
        }
    }
    return false;
};

// Here lazySourceList is a list of lazy (i.e., not yet) merged sources.
//
// Construct and fully populate exactly 2 new component sources for each lazy-merged
// source, then perform a full merge on those sources.
//
// Return a list of fully merged sources.
//
let loadAndMergeSourceList = (lazySourceList: LazySourceData[], orSourcesNcCsvMap: Map<string, number>): SourceList => {
    let sourceList: SourceList = [];
    for (let lazySource of lazySourceList) {
        Assert(lazySource.ncList.length === 2, `lazySource.ncList.length(${lazySource.ncList.length})`);
        Assert(lazySource.validateResultList.length === 2, `lazySource.validateResultList.length(${lazySource.validateResultList.length})`);
        let sourcesToMerge: AnySourceData[] = []; // TODO: not ideal, would prefer SourceData here
        //PP = true;
        for (let index = 0; index < 2; ++index) {
            sourcesToMerge.push(getSourceData(lazySource.ncList[index], lazySource.validateResultList[index], false, orSourcesNcCsvMap));
        }
        PP = false;
        sourceList.push(mergeSources(sourcesToMerge[0], sourcesToMerge[1], false) as SourceData);
    }
    return sourceList;
};

//
// args:
//   min_synonyms
//   max_synonyms
//
let getCombosForUseNcLists = (sum: number, max: number, pcd: PreComputedData, args: any): string[] => {
    let hash: StringAnyMap = {};
    let combos: string[] = [];

    let comboCount = 0;
    let totalVariations = 0;
    let numCacheHits = 0;
    let numMergeIncompatible = 0;
    let numUseIncompatible = 0;
    
    let MILLY = 1000000n;
    let start = process.hrtime.bigint();

    let useSourcesList: UseSource[] = pcd.useSourcesList;
    if (0) console.log(`useSourcesList: ${Stringify2(useSourcesList)}`);

    const mergeArgs = {
        min_synonyms: args.min_synonyms,
        max_synonyms: args.max_synonyms,
        lazy: true
    };

    // Given a sum, such as 3, generate an array of addend arrays ("count lists") that
    // that add up to that sum, such as [ [1, 2], [2, 1] ]
    let countListArray: number[][] = Peco.makeNew({ sum, max }).getCombinations(); 

    // for each countList
    countListArray.forEach((countList: number[]) => {
        AA = false;
        comboCount += 1;

        //console.log(`sum(${sum}) max(${max}) countList: ${Stringify(countList)}`);
        let sourceIndexes: number[] = [];
        let result = first(countList, sourceIndexes);
        if (result.done) return; // continue; 

        let numVariations = 1;

        // this is effectively Peco.getCombinations().forEach()
        let firstIter = true;
        while (!result.done) {
            AA = false;
            if (!firstIter) {
                // TODO problem 1:
                // problem1: why is this (apparently) considering the first two entries of the same
                // clue count (e.g. red, red). It doesn't matter when the clue counts are different,
                // but when they're the same, we're wasting time. Is there some way to determine if
                // the two lists are equal at time of get'ing (getClueSourceListArray) such that
                // we could optimize this.next for this condition?
                // timed; 58s in 2
                result = next(countList, sourceIndexes);
                if (result.done) break;
                numVariations += 1;
            } else {
                firstIter = false;
            }

            
            if (result.nameList!.includes('wallet')) {
                console.log(`hit: ${result.nameList}`);
                ZZ = true;
                AA = true
                WW = true
            } else {
                ZZ = false;
                AA = false;
                WW = false;
            }

            //console.log(`result.nameList: ${result.nameList}`);
            //console.log(`result.ncList: ${result.ncList}`);

            //const key = NameCount.listToString(result.ncList);
            const key: string = NameCount.listToSortedString(result.ncList!);
            let cacheHit = false;
            let sourceList: LazySourceData[];
            if (!hash[key]) {
                sourceList = mergeAllCompatibleSources(result.ncList!, mergeArgs) as LazySourceData[];
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
                hash[key].isCompatible = isCompatibleWithUseSources(
                    loadAndMergeSourceList(sourceList, pcd.orSourcesNcCsvMap),
                    useSourcesList, pcd.orSourcesNcCsvMap);
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
// [{ list:clues1, count:1 },{ list:clues2, count:2 }].
//
let makeCombosForSum = (sum: number, max: number, args: any): string[] => {
    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
        // TODO: whereever this is actually enforced:
        // console.error(`Enforcing max results: ${args.maxResults}`);
    }
    // TODO move this a layer or two out; use "validateArgs" 
    Assert(_.isEmpty(args.require), 'require not yet supported');
    Assert(!args.sources, 'sources not yet supported');
    if (!PCD) {
        PCD = preCompute(args);
    }
    return getCombosForUseNcLists(sum, max, PCD, args);
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
            or: args.or,
            fast: args.fast,
            load_max: ClueManager.getNumPrimarySources(),
            parallel: true,
            min_synonyms: args.min_synonyms,
            max_synonyms: args.max_synonyms,
        }));

    let cpus = OS.cpus().length;
    let cpus_used = cpus <= 6 ? cpus: cpus / 2;
    console.error(`cpus: ${cpus} max used: ${cpus_used}`);
    let p = new Parallel(range, {
        maxWorkers: cpus_used,
        evalPath: '${__dirname}/../../modules/bootstrap-combo-maker.js'
    });
    let entrypoint = BootstrapComboMaker.entrypoint;
    let beginDate = new Date();
    return p.map(entrypoint).then((data: any[]) => { // TODO: StringArray[] ?
        let d = new Duration(beginDate, new Date()).milliseconds;
        console.error(`time: ${PrettyMs(d)} chunks: ${data.length}`);
    });
    // check if range == data and /or if .then(return) passes thru
    //const filterResult = ClueManager.filter(data[i], args.sum, comboMap);
};

//
//
let getAllPrimarySrcCombos = (orSource: SourceBase, size: number): string[] => {
    let combos: string[] = [];
    let list = [...Array(orSource.primaryNameSrcList.length).keys()];
    Peco.makeNew({
        listArray: [list, list], // TODO: fix for arbitrary "size"
        max: 99999,
        noDuplicates: true
    }).getCombinations().forEach((indexList: number[]) => {
        if (indexList[0] === indexList[1]) return; // TODO: fix for arbitrary "size"
        let combo = indexList.map(index => orSource.primaryNameSrcList[index].count).sort().toString();
        combos.push(combo);
    });
    return combos;
}

//
//
let makeCombos = (args: any): any => {
    Assert(args.sum, 'missing sum');
    let sumRange: number[] = args.sum.split(',').map(_.toNumber);
    Assert(sumRange.length <= 2, `funny sum (${sumRange.length})`);

    Debug(`++combos, sum: ${sumRange} max: ${args.max} use: ${args.use}`);
    
    let total = 0;
    let begin = new Date();
    let first = sumRange[0];
    let last = sumRange.length > 1 ? sumRange[1] : first;
    if (args.parallel) {
        parallel_makeCombosForRange(first, last, args)
            .then((data: any[]) => {
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
        for (let sum = first; sum <= last; ++sum) {
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
        let d = new Duration(begin, new Date()).milliseconds;
        console.error(`--combos: ${PrettyMs(d)}, oob: ${oob}, ` +
            `isCWUS calls: ${isCWUS_calls} avg comps: ${_.toInteger(isCWUS_comps / isCWUS_calls)} `);

        
        // ++ INSTRUMENTATION
        let cwusEntries = [...Object.entries(CwusHash)].sort((a, b) => b[1].count - a[1].count);
        for (let x = 0; x < 100; ++x) {
            if (cwusEntries[x]) console.error(cwusEntries[x]);
        }
        console.error(`total: ${cwusEntries.length}`);
        // -- INSTRUMENTATION


        Debug(`total: ${total}, filtered(${_.size(comboMap)})`);
        _.keys(comboMap).forEach((nameCsv: string) => console.log(nameCsv));
    }
    return 1;
};

function getKnownNcListForName (name: string): NameCount.List {
    const countList: number[] = ClueManager.getCountListForName(name);
    Assert(!_.isEmpty(countList), `not a valid clue name: '${name}'`);
    return countList.map(count => Object({ name, count }));
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

function nameOrNcStrListToKnownNcLists (nameOrNcStrList: string[]): NameCount.List[] {
    return nameOrNcStrList.map(nameOrNcStr => NameCount.makeNew(nameOrNcStr))
        .map(nc => nc.count ? [nc] : getKnownNcListForName(nc.name));
}

function combinationNcList (combo: number[], ncLists: NameCount.List[]): NameCount.List {
    return combo.map((ncIndex: number, listIndex: number) => ncLists[listIndex][ncIndex]);
}

function combinationNcDataList (ncListIndexes: number[], ncLists: NameCount.List[]): NCDataList {
    return ncListIndexes.map((ncListIndex: number, listIndex: number) => Object({ ncList: ncLists[listIndex][ncListIndex]}));
}

function ncListsToCombinations (ncLists: NameCount.List[]): any {
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]),       // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)
    }).getCombinations()
        .map((combo: any) => combinationNcList(combo, ncLists));
}

function getCombinationNcLists (useArgsList: string[]): NameCount.List[] {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    return useArgsList.map(useArg => useArg.split(','))
        .map(nameOrNcStrList => nameOrNcStrListToKnownNcLists(nameOrNcStrList))
        .map(knownNcLists => ncListsToCombinations(knownNcLists));
}

// This is the exact same method as ncListsToCombinations? except for final map method. could pass as parameter.
function combinationsToNcLists (combinationNcLists: NameCount.List[]): NameCount.List[] {
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]),
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)       // sum of lengths of nclists
    }).getCombinations()
        .map((countList: number[]) => combinationNcList(countList, combinationNcLists));
}

// TODO: get rid of this and combinationsToNcLists, and add extra map step in buildAllUseNCData
function combinationsToNcDataLists (combinationNcLists: NameCount.List[]): NCDataList[] {
    Debug(`combToNcDataLists() combinationNcLists: ${Stringify(combinationNcLists)}`);
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]),
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)       // sum of lengths of nclists
    }).getCombinations()
        .map((ncListIndexes: number[]) => combinationNcDataList(ncListIndexes, combinationNcLists));
}

//
//
function buildAllUseNcLists (useArgsList: string[]): NameCount.List[] {
    return combinationsToNcLists(getCombinationNcLists(useArgsList));
}

//
//
function buildAllUseNcDataLists (useArgsList: string[]): NCDataList[] {
    return combinationsToNcDataLists(getCombinationNcLists(useArgsList));
}

//
//
let getCompatibleUseSourcesFromNcData = (args: any): UseSource[] => {
    // XOR first
    let sourceList = getUseSourcesList<XorSource>(args.allXorNcDataLists, Op.xor, args);

    // OR next
    let orSourceList = getUseSourcesList<OrSource>(args.allOrNcDataLists, Op.or, args);

    // final: merge OR with XOR
    if (!_.isEmpty(orSourceList)) {
        sourceList = mergeOrSourceList(sourceList, orSourceList);
    }
    console.error(`useSourceList(${sourceList.length})`);
    if (0) {
        sourceList.forEach((source, index) => {
            console.log(`${source.primaryNameSrcList}`);
        });
    }
    console.error(` orSourceList(${orSourceList.length})`);
    return sourceList;
};

//
//
let getOrSourcesNcCsvCountMap = (useSourcesList: UseSource[]): Map<string, number> => {
    let map = new Map<string, number>();
    for (let useSource of useSourcesList) {
        if (!useSource.orSource) continue;
        for (let ncCsv of _.keys(useSource.orSource.sourceNcCsvMap)) {
            let value = map.get(ncCsv) || 0;
            map.set(ncCsv, value + 1);
        }
    }
    return map;
};

//
//
let preCompute = (args: any): PreComputedData => {
   let begin = new Date();
    args.allXorNcDataLists = args.xor ? buildAllUseNcDataLists(args.xor) : [ [] ];
    //console.error(`allXorNcDataLists(${args.allXorNcDataLists.length})`);
    args.allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or) : [ [] ];
    
    let useSourcesList = getCompatibleUseSourcesFromNcData(args);
    // TODO: there is a faster way to generate this map, in mergeOrSources or something.
    let orSourcesNcCsvMap = getOrSourcesNcCsvCountMap(useSourcesList);

    // ++INSTRUMENTATION
    if (1) {
        let list: [string, number][] = [...orSourcesNcCsvMap.entries()]; //.sort((a, b) => b[1] - a[1]);
        console.error(`orSourcesNcCsvCount(${list.length})`);
    }
    // --INSTRUMENTATION
    
    let d = new Duration(begin, new Date()).milliseconds;
    console.error(`Precompute(${PrettyMs(d)})`);

    if (_.isEmpty(useSourcesList)) {
        if (args.xor || args.or) {
            console.error('incompatible --xor/--or params');
            process.exit(-1);
        }
    }
    if (0) process.exit(0);
    return { useSourcesList, orSourcesNcCsvMap };
};

//
//
function buildUseNcLists (useArgsList: string[]): NameCount.List[] {
    let useNcLists: NameCount.List[] = [];
    useArgsList.forEach((useArg: string) =>  {
        let args = useArg.split(',');
        let ncList: NameCount.List = args.map(arg => {
            let nc = NameCount.makeNew(arg);
            Assert(nc.count, `arg: ${arg} requires a :COUNT`);
            Assert(_.has(ClueManager.getKnownClueMap(nc.count), nc.name), `arg: ${nc} does not exist`);
            return nc;
        });
        useNcLists.push(ncList);
    });
    return useNcLists;
}

let formatListOfListCounts = (listOfLists: any[][]): string => {
    let total = listOfLists.reduce((sum, list) => {
        sum += list.length;
        return sum;
    }, 0);
    return `(${listOfLists.length}) total(${total})`;
}

let formatListOfNcDataCounts = (ncDataList: NCDataList): string => {
    let total = ncDataList.reduce((sum, ncData) => {
        sum += ncData.ncList.length;
        return sum;
    }, 0);
    return `(${ncDataList.length}) total(${total})`;
}

module.exports = {
    makeCombos,
    makeCombosForSum
};
