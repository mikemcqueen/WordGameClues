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
import * as MinMax from '../types/min-max';
import * as NameCount from '../types/name-count';
import { ValidateResult } from './validator';

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

interface StringBoolMap {
    [key: string]: boolean; // for now; eventually maybe array of string (sorted primary nameSrcCsv)
}

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
interface NCData {
    ncList: NameCount.List;
    synonym?: MinMax.Type;
}
type NCDataList = NCData[];

// TODO: renaming this "Source" stuff would be welcome. isn't it close to ValidateResult?

//
// TODO: find case where we use this without ncList and comment why ncList isn't here
//       even though all derived interfaces have an ncList.
// also synonyms.
//
interface SourceBase {
    primaryNameSrcList: NameCount.List;
}

//
//
interface LazySourceData extends SourceBase {
    ncList: NameCount.List;
    synonym: Clue.PropertyCounts.Type;
    validateResultList: ValidateResult[];
}

//
//
interface SourceData extends SourceBase {
    ncList: NameCount.List;
    synonym: Clue.PropertyCounts.Type;
    sourceNcCsvList: string[];
    ncCsv?: string;
}
type SourceList = SourceData[];
type AnySourceData = LazySourceData | SourceData;

//
//

type CountArray = Int32Array;

interface CountArrayAndSize {
    array: CountArray;
    size: number;
}

// TODO: or just XorSource here/
interface UseSourceBase extends SourceBase {
    // TODO: use a set.
    primarySrcArray: CountArray;
    ncList: NameCount.List;
}

type XorSource = UseSourceBase;

type SourceNcCsvMap = Record<string, number[]>;

namespace CompatibleOrSource {
    //
    //
    export interface Type {
        source: SourceData;
        xorCompatible: boolean;
        andCompatible: boolean;
    }

    export function init (source: SourceData): Type {
        return {
            source,
            xorCompatible: false,
            andCompatible: false
        };
    }

    //
    //
    export interface ListContainer {
        compatibleSourceList: Type[];
        compatible: boolean;
    }

    export function listContainerInit (): ListContainer {
        return {
            compatibleSourceList: [],
            compatible: false
        };
    }
}

// One OrSource contains all of the data for a single --or argument.
//
interface OrSource {
    sourceListContainer: CompatibleOrSource.ListContainer;
}

interface UseSourceLists {
    xor: XorSource[];
    or: OrSource[];
}

interface PreComputedData {
    useSourceLists: UseSourceLists;
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

let PCLog = false;//true;
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
let listIsEmpty = (list: any[]): boolean => {
    return list.length === 0;
}

//
//
let countArrayToNumberList = (countArray: CountArray): number[] => {
    let result: number[] = [];
    for (let index = 1; index < countArray.length; ++index) {
	if (countArray[index] === index) {
	    result.push(index);
	}
    }
    return result;
}

// 
//
let anyCountInArray = (ncList: NameCount.List, countArray: CountArray): boolean => {
    return ncList.some(nc => countArray[nc.count] === nc.count);
};

// 
//
let everyCountInArray = (ncList: NameCount.List, countArray: CountArray): boolean => {
    return ncList.every(nc => countArray[nc.count] === nc.count);
};

/*
//
//
let anyNumberInArray = (numberList: number[], countArray: CountArray): boolean => {
    return numberList.some(num => countArray[num] === num);
};
*/

//
//
let listToCountArray = (ncList: NameCount.List): CountArray => {
    return ncList.reduce((array, nc) => {
        array[nc.count] = nc.count;
        return array;
    }, new Int32Array(ClueManager.getNumPrimarySources()));
};

/*
//
//
let listToCountArrayAndSize = (ncList: NameCount.List): CountArrayAndSize => {
    let array = new Int32Array(ClueManager.getNumPrimarySources());
    let size = 0;
    for (let nc of ncList) {
        array[nc.count] = nc.count;
        size++;
    }
    return { array, size };
};
*/

/*
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
*/

/*
//
//
let getCountListNotInArray = (ncList: NameCount.List, countArray: CountArray): number[] => {
    return ncList.map(nc => nc.count).filter(count => countArray[count] !== count);
};
*/

/*
//
//
let getNumCountsInArray = (ncList: NameCount.List, countArray: CountArray): number => {
    // TODO: reduce
    let count = 0;
    for (let nc of ncList) {
        if (countArray[nc.count] === nc.count) ++count;
    }
    return count;
};
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

//
//
let getPropertyCounts = (result: ValidateResult): Clue.PropertyCounts.Map => {
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
    return propertyCounts;
}

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
    if (orSourcesNcCsvMap) {
        source.sourceNcCsvList = source.sourceNcCsvList.filter(ncCsv => orSourcesNcCsvMap.has(ncCsv));
    }

    source.ncList = [nc]; // TODO i could try getting rid of "LazySource.nc" and just make this part of LazySouceData
    source.synonym = getPropertyCounts(validateResult)[Clue.PropertyName.Synonym];
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
    return lazy ? {
        primaryNameSrcList,
        ncList: [nc],
        synonym: getPropertyCounts(validateResult)[Clue.PropertyName.Synonym],
        validateResultList: [validateResult]
    } : populateSourceData({ primaryNameSrcList }, nc, validateResult, orSourcesNcCsvMap);
};

// out of bounds
let oob = 0;

//
//
let propertyCountIsInBounds = (propertyCount: Clue.PropertyCounts.Type, min: number, max: number): boolean => {
    const total = propertyCount.total;
    return min <= total && total <= max;
};

//
//
let filterPropertyCountsOutOfBounds = (result: ValidateResult, args: MergeArgs): boolean => {
    const propertyCounts = getPropertyCounts(result);
    const inBounds = propertyCountIsInBounds(propertyCounts[Clue.PropertyName.Synonym], args.min_synonyms, args.max_synonyms);
    if (!inBounds) oob++;
    return inBounds;
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
    const primaryNameSrcList = [...source1.primaryNameSrcList, ...source2.primaryNameSrcList];
    const ncList = [...source1.ncList, ...source2.ncList];
    if (lazy) {
        Assert(ncList.length === 2, `ncList.length(${ncList.length})`);
        source1 = source1 as LazySourceData;
        source2 = source2 as LazySourceData;
        const synonym = Clue.PropertyCounts.merge(
            getPropertyCounts(source1.validateResultList[0])[Clue.PropertyName.Synonym],
            getPropertyCounts(source2.validateResultList[0])[Clue.PropertyName.Synonym]);
        const result: LazySourceData = {
            primaryNameSrcList,
            ncList,
            synonym,
            validateResultList: [
                (source1 as LazySourceData).validateResultList[0],
                (source2 as LazySourceData).validateResultList[0]
            ]
        };
        return result;
    }
    source1 = source1 as SourceData;
    source2 = source2 as SourceData;
    const mergedSource: SourceData = {
        primaryNameSrcList,
        synonym: Clue.PropertyCounts.merge(source1.synonym, source2.synonym),
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
    // also, uh, isn't there a primarySrcArray I can be using here?
    return allCountUnique(source1.primaryNameSrcList, source2.primaryNameSrcList)
        ? [mergeSources(source1, source2, args.lazy)]
        : [];
};

//
//
let mergeCompatibleSourceLists = (sourceList1: AnySourceData[], sourceList2: AnySourceData[], args: MergeArgs): AnySourceData[] => {
    let mergedSourcesList: AnySourceData[] = [];
    for (const source1 of sourceList1) {
        for (const source2 of sourceList2) {
            mergedSourcesList.push(...mergeCompatibleSources(source1, source2, args))
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
        if (listIsEmpty(sourceList)) break;
    }
    return sourceList;
};

// TODO: revisit this hash and it's purpose
//
let buildSourceListsForUseNcData = (useNcDataLists: NCDataList[], args: MergeArgs): SourceList[] => {
    let sourceLists: SourceList[] = [];
    let hashList: StringBoolMap[] = [];
    // TODO: forEach
    for (let [dataListIndex, useNcDataList] of useNcDataLists.entries()) {
        for (let [sourceListIndex, useNcData] of useNcDataList.entries()) {
            if (!sourceLists[sourceListIndex]) sourceLists.push([]);
            if (!hashList[sourceListIndex]) hashList.push({});
            let sourceList = mergeAllCompatibleSources(useNcData.ncList, args) as SourceList;
            for (let source of sourceList) {
                if (!propertyCountIsInBounds(source.synonym, args.min_synonyms, args.max_synonyms)) {
                    // never did get this to fire, but appears to be working, shut it off when i see it
                    console.error(`oob: [${NameCount.listToNameList(source.ncList)}], syn-total(${source.synonym.total})`);
                    continue;
                }
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
// NOTE: not used, but may use something similar at some point. maybe.
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
/*
  let buildUseSourceNcCsvMap = (useSourceList: UseSourceLists): SourceNcCsvMap => {
  return useSourceList.reduce((map: SourceNcCsvMap, source: UseSource, index: number) => {
  if (AA) console.log(Stringify2(source));
  const key = NameCount.listToSortedString(source.ncList);
  if (!map[key]) map[key] = [];
  map[key].push(index);
  return map;
  }, {});
  };
*/

// TODO: function name
//
const mergeCompatibleXorSources = (indexList: number[], sourceLists: SourceList[]): XorSource[] => {
    //
    // TODO: list of sourceLists outside of this loop. 
    // assign result.sourceLists inside indexList.entries() loop. 
    //
    let primaryNameSrcList: NameCount.List = [];
    let ncList: NameCount.List = []; // TODO: xor only
    let compatible = true;
    // TODO: indexList.some()
    for (let [sourceListIndex, sourceIndex] of indexList.entries()) {
        if (ZZ) console.log(`XOR sourceListIndex(${sourceListIndex}) sourceIndex(${sourceIndex})`);
        const source = sourceLists[sourceListIndex][sourceIndex];
        if (listIsEmpty(primaryNameSrcList)) {
            primaryNameSrcList.push(...source.primaryNameSrcList);
            ncList.push(...source.ncList);
            if (ZZ) console.log(` XOR pnsl, initial: ${primaryNameSrcList}`);
        } else {
            // TODO: hash of primary sources would be faster here. inside inner loop.
            let combinedNameSrcList = primaryNameSrcList.concat(source.primaryNameSrcList);
            // TODO: uniqBy da debil
            if (_.uniqBy(combinedNameSrcList, NameCount.count).length === combinedNameSrcList.length) {
                primaryNameSrcList = combinedNameSrcList;
                ncList.push(...source.ncList);
                if (ZZ) console.log(` XOR pnsl, combined: ${primaryNameSrcList}`);
            } else {
                if (ZZ) console.log(` XOR pnsl, emptied: ${primaryNameSrcList}`);
                compatible = false;
                break;
            }
        }
    }
    if (compatible) {
        if (0 || ZZ) console.log(` XOR pnsl, final: ${primaryNameSrcList}, indexList(${indexList.length}): [${indexList}]`);
        if (ZZ && listIsEmpty(primaryNameSrcList)) console.log(`empty pnsl`);
        
        //Assert(!_.isEmpty(primaryNameSrcList), 'empty primaryNameSrcList');
        let result: XorSource = {
            primaryNameSrcList,
            primarySrcArray: listToCountArray(primaryNameSrcList),
            ncList
        };
	if (ZZ) {
	    console.log(` XOR compatible: true, adding` +
		` pnsl[${NameCount.listToString(result.primaryNameSrcList)}]`);
	}
        return [result];
    } else {
	if (ZZ) console.log(` XOR compatible: false`);
    }
    return [];
};

//
// TODO function name,
let mergeCompatibleXorSourceCombinations = (sourceLists: SourceList[]/*, args: MergeArgs*/): XorSource[] => {
    // TODO: sometimes a sourceList is empty, like if doing $(cat required) with a
    // low clue count range (e.g. -c2,4). should that even be allowed?
    let listArray = sourceLists.map(sourceList => [...Array(sourceList.length).keys()]);
    if (ZZ) console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourceLists(${sourceLists.length}): ${Stringify2(sourceLists)}`);
    const peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    let sourceList: XorSource[] = [];
    for (let indexList = peco.firstCombination(); indexList; indexList = peco.nextCombination()) {
        //if (ZZ) console.log(`iter (${iter}), indexList: ${stringify(indexList)}`);
        const mergedSources: XorSource[] = mergeCompatibleXorSources(indexList, sourceLists);
        sourceList.push(...mergedSources);
    }
    return sourceList;
};

//
//
let getUseSourceLists = (ncDataLists: NCDataList[], args: any): SourceList[] => {
    if (listIsEmpty(ncDataLists[0])) return [];
    const mergeArgs: MergeArgs = {
        min_synonyms: args.min_synonyms,
        max_synonyms: args.max_synonyms,
        lazy: false
    };
    const sourceLists = buildSourceListsForUseNcData(ncDataLists, mergeArgs);
    return sourceLists;
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
let isSourceArrayXORCompatibleWithSourceList = (primarySrcArray: CountArray, sourceList: SourceList): boolean => {
    /*
      if (0 && AA) {
      console.log(` XOR: pnsl[${NameCount.listToCountList(source.primaryNameSrcList)}]` +
      `, primarySrcArray[${primarySrcArray)}]`);
      }
    */
    Assert(!listIsEmpty(sourceList));
    let compatible = true;
    for (let source of sourceList) {
        compatible = !anyCountInArray(source.primaryNameSrcList, primarySrcArray);
        if (!compatible) break;
    }
    if (AA) {
        console.log(` isSourceArrayXORCompatibileWithSourceList: ${compatible}`);
    }
    return true;
}

//
//
let isAnyCompatibleOrSourceANDCompatibleWithSourceArray = (compatibleSourceList: CompatibleOrSource.Type[],
                                                           primarySrcArray: CountArray): boolean => {
    let compatible = false;
    for (let compatibleSource of compatibleSourceList) {
        // this should never happen because AND compatibility should have propagated up to the
        // container level, and we never should have been called if container is compatible.
        Assert(!compatibleSource.andCompatible);
        compatible = everyCountInArray(compatibleSource.source.primaryNameSrcList, primarySrcArray);
        if (compatible) break;
    }
    return compatible;
}

//
//
let isAnyCompatibleOrSourceXORCompatibleWithSourceArray = (compatibleSourceList: CompatibleOrSource.Type[],
                                                           primarySrcArray: CountArray): boolean => {
    let compatible = false;
    for (let compatibleSource of compatibleSourceList) {
        // skip any sources that were already determined to be XOR incompatible or AND compatible
        // with supplied --xor sources.
        if (!compatibleSource.xorCompatible || compatibleSource.andCompatible) continue;
        compatible = !anyCountInArray(compatibleSource.source.primaryNameSrcList, primarySrcArray);
        if (compatible) break;
    }
    return compatible;
}

/*
//
//
let isSourceArrayANDCompatibleWithSourceList = (primarySrcArrayAndSize: CountArrayAndSize, sourceList: SourceList): boolean => {
    //
    //
    // TODO: I think I should eliminate .and sourceLists if any one of them is compatible
    // with a supplied XorSource, in filterXOR... (or add a filterAND.. method as well).
    //
    //
    Assert(!listIsEmpty(sourceList));
    let compatible = true;
    for (const source of sourceList) {
        const numCountsInArray = getNumCountsInArray(source.primaryNameSrcList, primarySrcArrayAndSize.array);
        compatible = numCountsInArray === primarySrcArrayAndSize.size;
        if (!compatible) break;
          if (0 && AA) {
          console.log(` AND: (${index}), match(${numCountsInArray})` +
          `, psnl[${NameCount.listToCountList(source.primaryNameSrcList)})]` +
          `, primarySrcArray[${countArrayToNumberList(primarySrcArrayAndSize.array)}]`);
          }
    }
    if (AA) {
        console.log(` isSourceArrayANDCompatibileWithSourceList: ${compatible}`);
    }
    return compatible;
}
*/

// OR == XOR || AND
//
let isSourceArrayCompatibleWithEveryOrSource = (primarySrcArray: CountArray, orSourceList: OrSource[]) : boolean => {
    CWOS_calls++;
    let compatible = true; // if no --or sources specified, compatible == true
    for (let orSource of orSourceList) {
        CWOS_comps++;
        
        //
        //
        //
        // TODO
        //
        //
        // skip calls to here if container.compatible = true  which may have been
        // determined in Precompute phase @ markAllANDCompatibleOrSources()
        // and skip the XOR check as well in this case.

        // First check for XOR compatibility
        compatible = isAnyCompatibleOrSourceXORCompatibleWithSourceArray(
            orSource.sourceListContainer.compatibleSourceList, primarySrcArray);
        // Any XOR compatible sources, means "OR compatibility" was achieved with this OrSource
        if (compatible) continue;

        // Next check for AND compatibility, our last hope at achieving "OR compatibility"
        compatible = isAnyCompatibleOrSourceANDCompatibleWithSourceArray(
            orSource.sourceListContainer.compatibleSourceList, primarySrcArray);
        if (!compatible) break;
    }
    if (AA) {
        console.log(` -orCompatible: ${compatible}, primarySrcArray:[${primarySrcArray}]`);
    }
    return compatible;
};

//
//
let isSourceXORCompatibleWithXorSource = (source: SourceData, xorSource: XorSource): boolean => {
    const compatible = !anyCountInArray(source.primaryNameSrcList, xorSource.primarySrcArray);
    return compatible;
};

//
//
let filterXorSourcesXORCompatibleWithSource = (xorSourceList: XorSource[], source: SourceData): XorSource[] => {
    let filteredXorSources: XorSource[] = [];
    for (let xorSource of xorSourceList) {
        const compatible = isSourceXORCompatibleWithXorSource(source, xorSource);
        if (compatible) {
            filteredXorSources.push(xorSource);
        }
    }
    return filteredXorSources;
};

let isCWUS_calls = 0;
let isCWUS_comps = 0;
let CWOS_calls = 0;
let CWOS_comps = 0;
let ACWOS_calls = 0;
let ACWOS_comps = 0;

//
//
let isAnySourceCompatibleWithUseSources = (sourceList: SourceList, pcd: PreComputedData): boolean => {
    //
    // NOTE: this is why --xor is required. OK for now. Fix later.
    //
    isCWUS_calls += 1;
    if (1) {
        if (listIsEmpty(pcd.useSourceLists.xor)) return true;
    } else {
        //if (listIsEmpty(pcd.useSourceList!)) return true;
    }

    let compatible = false;
    for (let sourceIndex = 0, sourceListLength = sourceList.length; sourceIndex < sourceListLength; ++sourceIndex) {
        let source = sourceList[sourceIndex];
        if (AA) console.log(`source nc: ${NameCount.listToString(source.ncList)}`);

        const xorSourceList = filterXorSourcesXORCompatibleWithSource(pcd.useSourceLists.xor, source);
        // if there were --xor sources specified, and none are compatible with the
        // current source, no further compatibility checking is necessary; continue
        // to next source.
        if (!listIsEmpty(pcd.useSourceLists.xor) && listIsEmpty(xorSourceList)) continue;

        const primarySrcArray = listToCountArray(source.primaryNameSrcList);
        compatible = isSourceArrayCompatibleWithEveryOrSource(primarySrcArray, pcd.useSourceLists.or);
        if (compatible) break;
    }
    return compatible;
};

// Here lazySourceList is a list of lazy, i.e., not yet fully initialized sources.
//
// Construct and fully populate exactly 2 new component sources for each lazy source,
// then perform a full merge on those sources.
//
// Return a list of fully merged sources.
//
let loadAndMergeSourceList = (lazySourceList: LazySourceData[], args: MergeArgs): SourceList => {
    let sourceList: SourceList = [];
    for (let lazySource of lazySourceList) {
        Assert(lazySource.ncList.length === 2);
        Assert(lazySource.validateResultList.length === 2);
        let sourcesToMerge: AnySourceData[] = []; // TODO: not ideal, would prefer SourceData here
        for (let index = 0; index < 2; ++index) {
            const sourceData = getSourceData(lazySource.ncList[index], lazySource.validateResultList[index], false);
            if (0 && ZZ) console.log(`lamSourceData[${index}]: ${Stringify2(sourceData)}`);
            sourcesToMerge.push(sourceData);
        }
        const mergedSource = mergeSources(sourcesToMerge[0], sourcesToMerge[1], false) as SourceData;
        if (propertyCountIsInBounds(mergedSource.synonym, args.min_synonyms, args.max_synonyms)) {
            if (0 && ZZ) console.log(`lamMerged[${NameCount.listToString(lazySource.ncList)}]: ${Stringify2(mergedSource)}`);
            sourceList.push(mergedSource);
        }
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
    
    const MILLY = 1000000n;
    const start = process.hrtime.bigint();

    //    const useSourceList: UseSource[] = pcd.useSourceList;
    //    if (1) console.log(`xorSourceList: ${Stringify2(pcd.xorSourceList)}`);

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
        comboCount += 1;

        //console.log(`sum(${sum}) max(${max}) countList: ${Stringify(countList)}`);
        let sourceIndexes: number[] = [];
        let result = first(countList, sourceIndexes);
        if (result.done) return; // continue; 

        let numVariations = 1;

        // this is effectively Peco.getCombinations().forEach()
        let firstIter = true;
        while (!result.done) {
            if (!firstIter) {
                // TODO:
                // why is this (apparently) considering the first two entries of the same
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

            
            if (0 && result.nameList!.includes('buffalo spring')) {
                //if (0 && result.nameList!.toString() === 'dark,note') {
                console.log(`hit: ${result.nameList}`);
                ZZ = true;
                AA = true
                WW = true
            } else {
                ZZ = false;
                AA = false;
                WW = false;
            }


            const key: string = NameCount.listToSortedString(result.ncList!);
            let cacheHit = false;
            let lazySourceList: LazySourceData[];
            if (!hash[key]) {
                lazySourceList = mergeAllCompatibleSources(result.ncList!, mergeArgs) as LazySourceData[];
                if (_.isEmpty(lazySourceList)) ++numMergeIncompatible;
                //console.log(`$$ sources: ${Stringify2(sourceList)}`);
                hash[key] = { lazySourceList };
            } else {
                // TODO: NOT lazy, potentially
                lazySourceList = hash[key].lazySourceList;
                cacheHit = true;
                numCacheHits += 1;
            }

            if (logging) console.log(`  found compatible sources: ${!_.isEmpty(lazySourceList)}`);

            // failed to find any compatible combos
            if (listIsEmpty(lazySourceList)) continue;

            if (_.isUndefined(hash[key].isCompatible)) {
                //*****************************************
                // TODO: HEY WAIT A SECOND WHY DONT I CACHE THE POPULATED SOURCE HERE
                //*****************************************
                hash[key].isCompatible = isAnySourceCompatibleWithUseSources(
                    loadAndMergeSourceList(lazySourceList, args as MergeArgs), pcd)
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
        console.error(`--combos: ${PrettyMs(d)}, oob: ${oob}` +
            `, isCWUS calls(${isCWUS_calls}), comps(${isCWUS_comps}), avg comps(${_.toInteger(isCWUS_comps / isCWUS_calls)})` +
            `, CWOS calls(${CWOS_calls}), comps(${CWOS_comps}), avg comps(${_.toInteger(CWOS_comps / CWOS_calls)})`);

        
        // ++ INSTRUMENTATION
        /*
        if (0)  {
            let cwusEntries = [...Object.entries(CwusHash)].sort((a, b) => b[1].count - a[1].count);
            for (let x = 0; x < 100; ++x) {
                if (cwusEntries[x]) console.error(cwusEntries[x]);
            }
            console.error(`total: ${cwusEntries.length}`);
        }
        */
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

function combinationNcList (indexList: number[], ncLists: NameCount.List[]): NameCount.List {
    return indexList.map((ncIndex: number, listIndex: number) => ncLists[listIndex][ncIndex]);
}

// TODO: ncDataCombinationFromNCLists
function combinationNcDataList (indexList: number[], ncLists: NameCount.List[]): NCDataList {
    return indexList.map((ncIndex: number, listIndex: number) => Object({ ncList: ncLists[listIndex][ncIndex]}));
}

// same as combinationNcDataList but takes NCDataList[] instead of NameCount.List[]
function ncDataCombinationsToNcDataList (indexList: number[], ncDataLists: NCDataList[]): NCDataList {
    return indexList.map((ncDataIndex: number, listIndex: number) => ncDataLists[listIndex][ncDataIndex]);
}

function ncListsToCombinations (ncLists: NameCount.List[]): NameCount.List[] {
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]),       // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)              // sum of lengths of nclists
    }).getCombinations()
        .map((indexList: number[]) => combinationNcList(indexList, ncLists));
}

// TODO: tuple
function minMaxNcListsTupleToNcDataCombinations (minMaxNcListsTuple: any[]): NCDataList[] {
    const minMax = minMaxNcListsTuple[0];
    const ncLists = minMaxNcListsTuple[1] as NameCount.List[];
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]),       // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)              // sum of lengths of nclists
    }).getCombinations()
        .map((indexList: number[]) => {
            let ncData: NCData = {
                ncList: combinationNcList(indexList, ncLists)
            };
            if (minMax) ncData.synonym = minMax;
            return ncData;
        });
}


function getCombinationNcLists (useArgsList: string[]): any {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    return useArgsList.map(useArg => useArg.split(','))
        .map(nameOrNcStrList => nameOrNcStrListToKnownNcLists(nameOrNcStrList))
        .map(knownNcLists => ncListsToCombinations(knownNcLists));
}

let isSingleNumericDigit = (arg: string): boolean => {
    return arg.length === 1 && arg[0] >= '0' && arg[0] <= '9';
}

let splitArgListAndMinMax = (argList: string[]): [MinMax.Type, string[]] => {
    Assert(argList.length > 2);
    const minArg = argList[argList.length - 2];
    const maxArg = argList[argList.length - 1];
    Assert(isSingleNumericDigit(minArg));
    Assert(isSingleNumericDigit(maxArg));
    return [MinMax.init(minArg, maxArg), argList.slice(0, argList.length - 2)];
};

let useArgToMinMaxNameListTuple = (useArg: string, hasMinMax: boolean): [MinMax.Type|undefined, string[]] => {
    let argList = useArg.split(',');
    let minMax: MinMax.Type|undefined = undefined;
    if (hasMinMax) {
        [minMax, argList] = splitArgListAndMinMax(argList);
    }
    return [minMax, argList];
};

function getCombinationNcDataLists (useArgsList: string[], minMax: boolean = false): any {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    return useArgsList.map(useArg => useArgToMinMaxNameListTuple(useArg, minMax))
        .map(minMaxNameListTuple => { return [minMaxNameListTuple[0], nameOrNcStrListToKnownNcLists(minMaxNameListTuple[1])]; }) // nameOrNcStrList 
        .map(minMaxNcListsTuple => minMaxNcListsTupleToNcDataCombinations(minMaxNcListsTuple)); // knownNcLists
    /*
        .map((minMaxNameListTuple: [MinMax.Type|undefined, string[]]) => {
            return [minMaxNameListTuple[0], nameOrNcStrListToKnownNcLists(minMaxNameListTuple[1])]; // nameOrNcStrList
        }) 
        .map((minMaxNcListsTuple: [MinMax.Type|undefined, NameCount.List[][]]) => ncListsToNcDataCombinations(minMaxNcListsTuple[1])); // knownNcLists
    */
}

// This is the exact same method as ncListsToCombinations? except for final map method. could pass as parameter.
function combinationsToNcLists (combinationNcLists: NameCount.List[]): NameCount.List[] {
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]), // keys of array are 0..ncList.length-1
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)        // sum of lengths of nclists
    }).getCombinations()
        .map((indexList: number[]) => combinationNcList(indexList, combinationNcLists));
}

// TODO: get rid of this and combinationsToNcLists, and add extra map step in buildAllUseNCData
function combinationsToNcDataLists (combinationNcLists: NameCount.List[]): NCDataList[] {
    Debug(`combToNcDataLists() combinationNcLists: ${Stringify(combinationNcLists)}`);
    return Peco.makeNew({
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]), // keys of array are 0..ncList.length-1
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)        // sum of lengths of nclists
    }).getCombinations()
        .map((ncListIndexes: number[]) => combinationNcDataList(ncListIndexes, combinationNcLists));
}

// a version of combinationsToNcDataLists that takes NCDataList[] instead of NameCount.List[]
function ncDataCombinationListsToNcDataLists (ncDataCombinationLists: NCDataList[]): NCDataList[] {
    Debug(`ncDataCombinationsToNcDataLists() ncDataCombinationLists: ${Stringify(ncDataCombinationLists)}`);
    return Peco.makeNew({
        listArray: ncDataCombinationLists.map(ncDataList => [...Array(ncDataList.length).keys()]), // keys of array are 0..ncDataList.length-1
        max: ncDataCombinationLists.reduce((sum, ncDataList) => sum + ncDataList.length, 0)        // sum of lengths of ncDataLists
    }).getCombinations()
        .map((indexList: number[]) => ncDataCombinationsToNcDataList(indexList, ncDataCombinationLists));
}

/*
//
//
function buildAllUseNcLists (useArgsList: string[]): NameCount.List[] {
    return combinationsToNcLists(getCombinationNcLists(useArgsList));
}
*/

//
//
function buildAllUseNcDataLists (useArgsList: string[]): NCDataList[] {
    const combinationNcLists = getCombinationNcLists(useArgsList);
    //console.log(`combinationNcLists: ${Stringify2(combinationNcLists)}`);
    const ncDataList = combinationsToNcDataLists(combinationNcLists);
    //console.log(`ncDataList: ${Stringify2(ncDataList)}`);
    return ncDataList;
}

// for combining --xor with --xormm
//
function buildCombinedUseNcDataLists (useArgsList: string[], minMaxUseArgsList: string[]): NCDataList[] {
    const standardNcDataLists = getCombinationNcDataLists(useArgsList);
    const minMaxNcDataLists = getCombinationNcDataLists(minMaxUseArgsList, true);
    //console.log(`standardNcDataLists: ${Stringify2(standardNcDataLists)}`);
    //console.log(`minMaxNcDataLists: ${Stringify2(minMaxNcDataLists)}`);
    const combinedNcDataLists = [...standardNcDataLists, ...minMaxNcDataLists];
    //console.log(`combinedNcDataLists: ${Stringify2(combinedNcDataLists)}`);
    const ncDataLists = ncDataCombinationListsToNcDataLists(combinedNcDataLists);
    //console.log(`ncDataLists: ${Stringify2(ncDataLists)}`);
    return ncDataLists;
}

//
//
let buildCompatibleOrSourceListContainer = (sourceList: SourceList): CompatibleOrSource.ListContainer => {
    const listContainer = CompatibleOrSource.listContainerInit();
    for (let source of sourceList) {
        const compatibleSource = CompatibleOrSource.init(source);
        listContainer.compatibleSourceList.push(compatibleSource);
    }
    return listContainer;
};

//
let buildOrSourceList = (sourceLists: SourceList[]): OrSource[] => {
    let orSourceList: OrSource[] = [];
    for (let sourceList of sourceLists) {
        const orSource: OrSource = {
            sourceListContainer: buildCompatibleOrSourceListContainer(sourceList)
        };
        orSourceList.push(orSource);
    }
    return orSourceList;
};

//
//
let isSourceXORCompatibleWithAnyXorSource = (source: SourceData, xorSourceList: XorSource[]): boolean => {
    let compatible = listIsEmpty(xorSourceList); // empty list == compatible
    for (let xorSource of xorSourceList) {
        compatible = !anyCountInArray(source.primaryNameSrcList, xorSource.primarySrcArray);
        if (compatible) break;
    }
    return compatible;
    /*
    if (0 && AA) {
	console.log(` -xorCompatible: ${compatible}` + 
	    ` source.pnslCounts[${NameCount.listToCountList(source.primaryNameSrcList)}]` +
	    ` xorSource.primarySrcArray[${countArrayToNumberList(xorSource.primarySrcArray)}]`);
    }
    */
};


// Given a list of XorSources, and a list of OrSources, TODO
//
let markAllXORCompatibleOrSources = (xorSourceList: XorSource[], orSourceList: OrSource[]): void => {
    for (let orSource of orSourceList) {
        let compatibleSourceList = orSource.sourceListContainer.compatibleSourceList;
        for (let compatibleSource of compatibleSourceList) {
            if (isSourceXORCompatibleWithAnyXorSource(compatibleSource.source, xorSourceList)) {
                compatibleSource.xorCompatible = true;
            }
        }
    }
}

//
//
let buildUseSourceListsFromNcData = (args: any): UseSourceLists => {
    // XOR first
    let xorSourceList = mergeCompatibleXorSourceCombinations(getUseSourceLists(args.allXorNcDataLists, args)/*, mergeArgs*/);
    console.error(`xorSourceList(${xorSourceList.length})`);

    // OR next
    let orSourceList = buildOrSourceList(getUseSourceLists(args.allOrNcDataLists, args) /*, mergeArgs*/);
    console.error(`orSourceList(${orSourceList.length})`);

    //console.log(`orSourceList: ${Stringify2(orSourceList)}`);
    markAllXORCompatibleOrSources(xorSourceList, orSourceList);

    /*
    if (1) {
        useSourceList.forEach((source, index) => {
            const orSource: OrSource = source.orSource!;
            if (orSource && !orSource.sourceLists) console.log(`orSourceLists ${orSource.sourceLists}`);
            if (orSource && orSource.sourceLists && listIsEmpty(orSource.sourceLists)) console.log(`EMPTY4`);
        });
    }
    */
    return { xor: xorSourceList, or: orSourceList };
};

/*
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
*/

//
//
let preCompute = (args: any): PreComputedData => {
   let begin = new Date();
    //args.allXorNcDataLists = args.xor ? buildAllUseNcDataLists(args.xor) : [ [] ];
    args.allXorNcDataLists = buildCombinedUseNcDataLists(args.xor, args.xormm);
    //console.error(`allXorNcDataLists(${args.allXorNcDataLists.length})`);
    args.allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or) : [ [] ];
    
    let useSourceLists = buildUseSourceListsFromNcData(args);

    /*
    // TODO: there is a faster way to generate this map, in mergeOrSources or something.
    let orSourcesNcCsvMap = new Map<string, number>(); // getOrSourcesNcCsvCountMap(useSourcesList);
    // ++DEBUG
    if (0) {
        let list: [string, number][] = [...orSourcesNcCsvMap.entries()]; //.sort((a, b) => b[1] - a[1]);
        console.error(`orSourcesNcCsvCount(${list.length})`);
    }
    // ++DEBUG
    */

    let d = new Duration(begin, new Date()).milliseconds;
    console.error(`Precompute(${PrettyMs(d)})`);

    /*
    if ((listIsEmpty(xorSourceList) && args.xor) ||
        (listIsEmpty(orSourceList) && args.or))
    */
    if (listIsEmpty(useSourceLists.xor) && args.xor)
    {
        console.error('incompatible --xor/--or params');
        process.exit(-1);
    }
    return { useSourceLists }; // , orSourcesNcCsvMap };
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
