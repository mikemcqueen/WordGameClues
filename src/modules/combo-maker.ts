//
// combo-maker.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const BootstrapComboMaker = require('../../modules/bootstrap-combo-maker');
const ResultMap   = require('../../types/result-map');
const Peco        = require('../../modules/peco');
const Log         = require('../../modules/log')('combo-maker');
const My          = require('../../modules/util');
const NativeComboMaker = require('../../native-modules/experiment/build/Release/experiment.node');

const Assert      = require('assert');
const Debug       = require('debug')('combo-maker');
const Duration    = require('duration');
const Expect      = require('should/as-function');
const Fs          = require('fs-extra');
const OS          = require('os');
const Parallel    = require('paralleljs');
const Path        = require('path');
const PrettyMs    = require('pretty-ms');
const stringify   = require('javascript-stringify').stringify;
const Stringify2  = require('stringify-object');

import * as Clue from '../types/clue';
import * as ClueManager from './clue-manager';
//import * as CountBits from '../types/count-bits-roaring';
import * as CountBits from '../types/count-bits-fastbitset';
import * as MinMax from '../types/min-max';
import * as NameCount from '../types/name-count';
import * as Synonym from './synonym';

import { ValidateResult } from './validator';

// TODO: import from somewhere. also defined in clue-manager
const DATA_DIR =  Path.normalize(`${Path.dirname(module.filename)}/../../../data/`);

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
    synonymMinMax: MinMax.Type;
    lazy?: boolean;
};

//
//
interface NCData {
    ncList: NameCount.List;
    synonymMinMax?: MinMax.Type;
}
type NCDataList = NCData[];

//
//

type CountArray = Int32Array;

interface CountArrayAndSize {
    array: CountArray;
    size: number;
}

// TODO: renaming this "Source" stuff would be welcome. isn't it close to ValidateResult?
//
interface SourceBase {
    primaryNameSrcList: NameCount.List;
    primarySrcBits: CountBits.Type;
    ncList: NameCount.List;
}

//
//
interface LazySourceData extends SourceBase {
    synonymCounts: Clue.PropertyCounts.Type;
    validateResultList: ValidateResult[];
}

//
//
interface SourceData extends SourceBase {
    synonymCounts: Clue.PropertyCounts.Type;
    sourceNcCsvList: string[];
//    ncCsv?: string;
}
type SourceList = SourceData[];
type AnySourceData = LazySourceData | SourceData;

type XorSource = SourceBase;
type XorSourceList = SourceBase[];

type SourceNcCsvMap = Record<string, number[]>;

//
//
interface MergedSources {
    primarySrcBits: CountBits.Type;
    sourceList: SourceList;
}
type MergedSourcesList = MergedSources[];


interface OrSourceData {
    source: SourceData;
    xorCompatible: boolean;
    andCompatible: boolean;
}
type OrSourceList = OrSourceData[];

let initOrSource = (source: SourceData): OrSourceData => {
    return {
        source,
        xorCompatible: false,
        andCompatible: false
    };
}

// One OrArgData contains all of the data for a single --or argument.
//
interface OrArgData {
    orSourceList: OrSourceList;
    compatible: boolean;
}
type OrArgDataList = OrArgData[];

let initOrArgData = (): OrArgData => {
    return {
        orSourceList: [],
        compatible: false
    };
}

interface UseSourceLists {
    xor: XorSourceList;
    orArgDataList: OrArgDataList;
}

interface PreComputedData {
    useSourceLists: UseSourceLists;
    sourceListMap: Map<string, AnySourceData[]>;
}

let PCD: PreComputedData;

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
let WW = false;

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
        result += `    primaryNameSrcList: ${source.primaryNameSrcList}\n`;
        result += `    ncList: ${source.ncList}\n`;
        result += `    synonymCounts: ${Stringify2(source.synonymCounts)}\n`;
        //result += `    sourcNcCsvList: ${Stringify2(source.sourceNcCsvList)}\n`;
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
let sourceListToNcList = (sourceList: AnySourceData[]): NameCount.List => {
    return sourceList.reduce((ncList: NameCount.List, source) => {
        ncList.push(...source.ncList);
        return ncList;
    }, []);
};

//
//
let allCountUnique = (ncList: NameCount.List): boolean => {
    let set = new Set<number>();
    for (let nc of ncList) {
        if (set.has(nc.count)) return false;
	set.add(nc.count);
    }
    return true;
};

//
//
let allCountUnique2 = (nameSrcList1: NameCount.List, nameSrcList2: NameCount.List): boolean => {
    let set = new Set<number>();
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
};

let listGetNumEmptySublists = (listOfLists: any[][]) => {
    let numEmpty = 0;
    for (let list of listOfLists) {
        if (listIsEmpty(list)) ++numEmpty;
    }
    return numEmpty;
};

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

//
//
let listToCountArray = (ncList: NameCount.List): CountArray => {
    return ncList.reduce((array, nc) => {
        array[nc.count] = nc.count;
        return array;
    }, new Int32Array(ClueManager.getNumPrimarySources()));
};

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
                return key;
            }
            // C: single nested key with array value type: ignore; TODO assert primary?
        }
        return [];
    });
    if (!_.isEmpty(keys)) {
        // push combined sorted keys for multi-key case
        if (keys.length > 1) {
            let sortedKeys = keys.sort().toString();
            list.push(sortedKeys);
        }
        keys.forEach(key => {
            // push individual keys
            list.push(key);
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
// TODO: ForPrimaryClue
let getPropertyCountsMapForPrimaryNameSrc = (nameSrc: NameCount.Type): Clue.PropertyCounts.Map => {
    return _.find(ClueManager.getClueList(1), {
        name: nameSrc.name,
        src: _.toString(nameSrc.count)
    }).propertyCounts!;
};

//
//
let getPropertyCountsMapForCompoundClue = (clue: Clue.Compound, count: number): Clue.PropertyCounts.Map => {
    return Clue.PropertyCounts.createMapFromClue(_.find(ClueManager.getClueList(count), clue));
};

//
//
let getPropertyCountsMapForValidateResult = (validateResult: ValidateResult): Clue.PropertyCounts.Map => {
    const count = validateResult.nameSrcList.length;
    if (count === 1) {
        // primary clue: propertyCounts map is attached to clue itself
        // TODO: Clue.fromPrimaryNameSrc (): Clue.Primary
        return getPropertyCountsMapForPrimaryNameSrc(validateResult.nameSrcList[0]); // because length is 1
    } else {
        // compound clue: propertyCounts of sources are attached to each ValidateResult
        return validateResult.propertyCounts!;
    }
};

// Return keys of form ['name1:M', 'name2:N'] as array of form ['name1', 'name2'].
// 
let getResultMapTopLevelClueNameList = (resultMap: any): string[] => {
    return Object.keys(resultMap.internal_map)
        .map(nameSrcStr => nameSrcStr.split(':')[0])
};

//
//
let getPropertyCountsMapForNcAndValidateResult = (nc: NameCount.Type,
                                                  validateResult: ValidateResult): Clue.PropertyCounts.Map => {
    Assert(nc.count === validateResult.nameSrcList.length); // a hypothesis
    const count = validateResult.nameSrcList.length;
    const propertyCounts = getPropertyCountsMapForValidateResult(validateResult);

    // For primary clue, there's only one source variation - the source is the source.
    // PropertyCounts are attached to the clue itself, which we get via the validateResult.
    if (count === 1) return propertyCounts;

    // Compound clues may have many source combination variations, and as a result the
    // the propertyCounts "totals" (for the clue itself plus all its sources) are not
    // stored in the clue itself. Instead, we must merge (add) the propertyCounts of a
    // *particular* source (ValidateResult) with those of the clue itself.
    // TODO: Clue.fromNameAndNameSrcList (): Clue.Compound
    // TODO: 
    const clue = {
        name: nc.name,
        src: getResultMapTopLevelClueNameList(validateResult.resultMap).sort().toString()
    };
    //console.error(`clue: ${Clue.toJSON(clue)}, propertyCounts: ${propertyCounts}`);
    return Clue.PropertyCounts.mergeMaps(propertyCounts,
                                         getPropertyCountsMapForCompoundClue(clue, nc.count));
}

//
//
let getSynonymCountsForValidateResult = (validateResult: ValidateResult): Clue.PropertyCounts.Type => {
    return getPropertyCountsMapForValidateResult(validateResult)[Clue.PropertyName.Synonym];
};

// For primary clues, this is just the synonymCounts attached to the clue.
// For compound clues, this is a combination of the synonymCounts attached to the
// clue, and the synonymCounts attached to the sources represented by validateResult.
let getSynonymCountsForNcAndValidateResult = (nc: NameCount.Type,
                                              validateResult: ValidateResult): Clue.PropertyCounts.Type => {
    return getPropertyCountsMapForNcAndValidateResult(nc, validateResult)[Clue.PropertyName.Synonym];
};

// this really could be reduced to "makeSourceNcCsvList"
//
let makeSourceData = (nc: NameCount.Type, validateResult: ValidateResult,
                      orSourcesNcCsvMap?: Map<string, number>): SourceData => {
    let sourceNcCsvList;
    if (validateResult.resultMap) {
        sourceNcCsvList = buildSrcNcList(validateResult.resultMap.map());
    } else {
        Assert(validateResult.ncList.length === 1 && validateResult.ncList[0].count === 1, 'wrong assumption');
        sourceNcCsvList = [NameCount.listToString(validateResult.ncList)];
    }
    if (nc.count > 1) {
        sourceNcCsvList.push(NameCount.toString(nc));
    }
    if (orSourcesNcCsvMap) {
        sourceNcCsvList = sourceNcCsvList.filter(ncCsv => orSourcesNcCsvMap.has(ncCsv));
    }
    return {
	primaryNameSrcList: validateResult.nameSrcList,
	primarySrcBits: validateResult.srcBits,
	ncList: [nc], // TODO i could try getting rid of "LazySource.nc" and just make this part of LazySouceData
	synonymCounts: getSynonymCountsForNcAndValidateResult(nc, validateResult),
	sourceNcCsvList
    };
};

//
//
let getSourceData = (nc: NameCount.Type, validateResult: ValidateResult, lazy: boolean | undefined,
    orSourcesNcCsvMap?: Map<string, number>): AnySourceData =>
{
    Assert(validateResult.srcBits, `${nc}`);
    return lazy ? {
	primaryNameSrcList: validateResult.nameSrcList,
	primarySrcBits: validateResult.srcBits,
        ncList: [nc],
        synonymCounts: getSynonymCountsForNcAndValidateResult(nc, validateResult),
        validateResultList: [validateResult]
    } : makeSourceData(nc, validateResult, orSourcesNcCsvMap);
};

// out of bounds
let oob = 0;

//
//
let propertyCountsIsInBounds = (propertyCount: Clue.PropertyCounts.Type, minMax: MinMax.Type|undefined): boolean => {
    const total = propertyCount.total;
    return minMax ? (minMax.min <= total) && (total <= minMax.max) : true;
};

//
//
let filterPropertyCountsOutOfBounds = (nc: NameCount.Type, result: ValidateResult, args: MergeArgs): boolean => {
    const synonymCounts = getSynonymCountsForNcAndValidateResult(nc, result);
    const inBounds = propertyCountsIsInBounds(synonymCounts, args.synonymMinMax);
    if (!inBounds) oob++;
    return inBounds;
};

//
//
let getSourceList = (nc: NameCount.Type, args: MergeArgs): AnySourceData[] => {
    const sourceList: AnySourceData[] = [];

    ClueManager.getKnownSourceMapEntries(nc)
        .forEach((sourceData: ClueManager.SourceData) => {
            sourceList.push(...sourceData.results
                //.filter((result: ValidateResult) => filterPropertyCountsOutOfBounds(nc, result, args))
                .map((result: ValidateResult) =>  getSourceData(nc, result, args.lazy)));
        });
    return sourceList;
};

let merges = 0;
let list_merges = 0;

//
//
let mergeSources = (source1: AnySourceData, source2: AnySourceData, lazy: boolean | undefined): AnySourceData => {
    merges += 1;
    const primaryNameSrcList = [...source1.primaryNameSrcList, ...source2.primaryNameSrcList];
    const primarySrcBits = CountBits.or(source1.primarySrcBits, source2.primarySrcBits);
    const ncList = [...source1.ncList, ...source2.ncList];
    if (lazy) {
        Assert(ncList.length === 2, `ncList.length(${ncList.length})`);
        source1 = source1 as LazySourceData;
        source2 = source2 as LazySourceData;
        const result: LazySourceData = {
            primaryNameSrcList,
	    primarySrcBits,
            ncList,
            synonymCounts: Clue.PropertyCounts.merge(
                getSynonymCountsForValidateResult(source1.validateResultList[0]),
                getSynonymCountsForValidateResult(source2.validateResultList[0])),
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
	primarySrcBits,
        ncList,
        synonymCounts: Clue.PropertyCounts.merge(source1.synonymCounts, source2.synonymCounts),
        sourceNcCsvList: [...source1.sourceNcCsvList, ...source2.sourceNcCsvList]
    };
    // TODO: still used?
    //mergedSource.ncCsv = NameCount.listToSortedString(mergedSource.ncList);
    return mergedSource;
};

//
//
let mergeCompatibleSources = (source1: AnySourceData, source2: AnySourceData, args: MergeArgs): AnySourceData[] => {
    // TODO: this logic could be part of mergeSources
    // also, uh, isn't there a primarySrcArray I can be using here?
//  return allCountUnique2(source1.primaryNameSrcList, source2.primaryNameSrcList)
    return !CountBits.intersects(source1.primarySrcBits, source2.primarySrcBits)
        ? [mergeSources(source1, source2, args.lazy)]
        : [];
};

//
//
let mergeCompatibleSourceLists = (sourceList1: AnySourceData[], sourceList2: AnySourceData[], args: MergeArgs): AnySourceData[] => {
    list_merges += 1;
    let mergedSourcesList: AnySourceData[] = [];
    for (const source1 of sourceList1) {
        for (const source2 of sourceList2) {
            mergedSourcesList.push(...mergeCompatibleSources(source1, source2, args))
        }
    }
    return mergedSourcesList;
};

let validateSourceBits = (source: SourceData): CountBits.Type => {
    let sourceBits = CountBits.makeNew();
    CountBits.setMany(sourceBits, NameCount.listToCountList(source.primaryNameSrcList));
    if (!sourceBits.equals(source.primarySrcBits)) {
	throw new Error("**source bits mismatch**");
    }
    return sourceBits;
};

let validateMergedSourcesBits = (mergedSources: MergedSources,
    lastMergedSource: SourceData | undefined = undefined): void =>
{
    let mergedBits = CountBits.makeNew();
    let ncCsvList: string[] = [];
    for (const source of mergedSources.sourceList) {
	const sourceBits = validateSourceBits(source);
	CountBits.orInPlace(mergedBits, sourceBits);
	ncCsvList.push(NameCount.listToString(source.ncList));
    }
    if (!mergedBits.equals(mergedSources.primarySrcBits)) {
	console.error(ncCsvList.toString());
	if (lastMergedSource) {
	    console.error(`lastMergedSource, nc(${NameCount.listToString(lastMergedSource.ncList)})` +
		`, primary(${NameCount.listToString(lastMergedSource.primaryNameSrcList)})`);
	}
	console.error(mergedSources.primarySrcBits.toString());
	console.error(mergedBits.toString());
	throw new Error("**merged bits mismatch**");
    }
};

let validatePrimarySrcBits = (mergedSourcesList: MergedSourcesList) : void => {
    for (const mergedSources of mergedSourcesList) {
	validateMergedSourcesBits(mergedSources);
    }
};

//
//
let mergeCompatibleSourceLists2 = (mergedSourcesList: MergedSourcesList,
    sourceList: SourceList): MergedSourcesList =>
{
    list_merges += 1;
    let result: MergedSourcesList = [];
    for (const mergedSources of mergedSourcesList) {
	//validateMergedSourcesBits(mergedSources);
        for (const source of sourceList) {
	    //validateSourceBits(source);
	    if (!CountBits.intersects(mergedSources.primarySrcBits, source.primarySrcBits)) {
		if (0) {
		    console.log(`merging nc(${NameCount.listToString(source.ncList)}), ` +
			`primary(${NameCount.listToString(source.primaryNameSrcList)}), ` +
			`bits(${source.primarySrcBits.toString()})`);
		}
		let ms: MergedSources = {
		    primarySrcBits: CountBits.or(mergedSources.primarySrcBits, source.primarySrcBits),
		    sourceList: [...mergedSources.sourceList]
		};
		ms.sourceList.push(source);
		//validateMergedSourcesBits(ms, source);
		result.push(ms);
	    }
        }
    }
    return result;
};

let makeMergedSourcesList = (sourceList: SourceList) : MergedSourcesList => {
    let mergedSourcesList: MergedSourcesList = [];
    for (const source of sourceList) {
	mergedSourcesList.push({
	    primarySrcBits: CountBits.makeFrom(source.primarySrcBits),
	    sourceList: [source]
	});
    }
    return mergedSourcesList;
};

//
//
let mergeAllCompatibleSources2 = (ncList: NameCount.List,
    sourceListMap: Map<string, AnySourceData[]>): MergedSourcesList =>
{
    // because **maybe** broken for > 2 below
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    // TODO: reduce (or some) here
    let mergedSourcesList: MergedSourcesList =
	makeMergedSourcesList(sourceListMap.get(NameCount.toString(ncList[0])) as SourceList);
    for (let ncIndex = 1; ncIndex < ncList.length; ++ncIndex) {
        const nextSourceList: SourceList = sourceListMap.get(NameCount.toString(ncList[ncIndex])) as SourceList;
        mergedSourcesList = mergeCompatibleSourceLists2(mergedSourcesList, nextSourceList);
        //if (!sourceListHasPropertyCountInBounds(sourceList, args.synonymMinMax)) sourceList = [];
        // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
        if (listIsEmpty(mergedSourcesList)) break;
    }
    return mergedSourcesList;
};

//
//
let getSynonymCounts = (sourceList: AnySourceData[]): Clue.PropertyCounts.Type => {
    return sourceList.reduce(
        (counts, source) => Clue.PropertyCounts.add(counts, source.synonymCounts),
        Clue.PropertyCounts.empty());
};
                      
//
//
let sourceListHasPropertyCountInBounds = (sourceList: AnySourceData[], minMax: MinMax.Type|undefined): boolean => {
    const synonymCounts = getSynonymCounts(sourceList);
    const inBounds = propertyCountsIsInBounds(synonymCounts, minMax);
    if (!inBounds) {
        if (0) {
            console.error(`oob: [${NameCount.listToNameList(sourceListToNcList(sourceList))}]` +
                `, syn-total(${synonymCounts.total})`);
        }
    }
    return inBounds;
};

//
//
let mergeAllCompatibleSources = (ncList: NameCount.List, sourceListMap: Map<string, AnySourceData[]>,
				 args: MergeArgs): AnySourceData[] => {
    // because **maybe** broken for > 2 below
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    // TODO: reduce (or some) here
    let sourceList = sourceListMap.get(NameCount.toString(ncList[0])) as AnySourceData[];
    for (let ncIndex = 1; ncIndex < ncList.length; ++ncIndex) {
        const nextSourceList: AnySourceData[] = sourceListMap.get(NameCount.toString(ncList[ncIndex])) as AnySourceData[];
        sourceList = mergeCompatibleSourceLists(sourceList, nextSourceList, args);
        //if (!sourceListHasPropertyCountInBounds(sourceList, args.synonymMinMax)) sourceList = [];
        // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
        if (listIsEmpty(sourceList)) break;
    }
    return sourceList;
};

//
//
let buildSourceListsForUseNcData = (useNcDataLists: NCDataList[], sourceListMap: Map<string, AnySourceData[]>,
				    args: MergeArgs): SourceList[] => {
    let sourceLists: SourceList[] = [];
    // TODO: This is to prevent duplicate sourceLists. I suppose I could use a Set or Map, above?
    let hashList: StringBoolMap[] = [];
    for (let ncDataList of useNcDataLists) {
        for (let [sourceListIndex, useNcData] of ncDataList.entries()) {
            if (!sourceLists[sourceListIndex]) sourceLists.push([]);
            if (!hashList[sourceListIndex]) hashList.push({});
            // give priority to any min/max args specific to an NcData, for example, through --xormm,
            // but fallback to the values we were called with
            const mergeArgs = useNcData.synonymMinMax ? { synonymMinMax: useNcData.synonymMinMax } : args;
            const sourceList = mergeAllCompatibleSources(useNcData.ncList, sourceListMap, mergeArgs) as SourceList;
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
//
let addNcListToSourceListMap = (ncList: NameCount.List,
    map: Map<string, AnySourceData[]>, mergeArgs: MergeArgs) : void =>
{
    for (let nc of ncList) {
	const key = NameCount.toString(nc)
	if (!map.has(key)) {
	    const sourceList = getSourceList(nc, mergeArgs);
	    if (listIsEmpty(sourceList)) {
		throw new Error(`empty sourceList: ${NameCount.toString(nc)}`);
	    }
	    map.set(key, sourceList);
	}
    }
};

//
//
let getUseNcSourceListMap = (useNcDataLists: NCDataList[],
    mergeArgs: MergeArgs): Map<string, AnySourceData[]> =>
{
    let map = new Map<string, AnySourceData[]>();
    let numNc = 0;
    let begin = new Date();
    for (let useNcDataList of useNcDataLists) {
        for (let useNcData of useNcDataList) {
	    numNc += useNcData.ncList.length;
	    addNcListToSourceListMap(useNcData.ncList, map, mergeArgs);
	}
    }
    let end = new Duration(begin, new Date()).milliseconds;
    console.error(` getUseNcSourceListMap(${PrettyMs(end)}), nc(${numNc}), unique(${map.size}`);
    return map;
};

//
//
let fillKnownNcSourceListMapForSum = (map: Map<string, AnySourceData[]>,
    sum: number, max: number, mergeArgs: MergeArgs) : void =>
{
    // Given a sum, such as 4, and a max # of numbers to combine, such as 2, generate
    // an array of addend arrays ("count lists"), for each 2 <= N <= max, that add up
    // to that sum, such as [ [1, 3], [2, 2] ]
    let countListArray: number[][] = Peco.makeNew({ sum, max }).getCombinations(); 
    // for each countList
    countListArray.forEach((countList: number[]) => {
        let sourceIndexes: number[] = [];
        let result = first(countList, sourceIndexes);
        if (result.done) return; // continue; 

        let firstIter = true;
        while (!result.done) {
            if (!firstIter) {
                result = next(countList, sourceIndexes);
                if (result.done) break;
            } else {
                firstIter = false;
            }
	    addNcListToSourceListMap(result.ncList!, map, mergeArgs);
	}
    });
};

//
//
let getKnownNcSourceListMap = (first: number, last: number,
    args: any): Map<string, AnySourceData[]> =>
{
    // NOTE: correct, but hacky
    last = ClueManager.getNumPrimarySources();
    let map = new Map<string, AnySourceData[]>();
    const mergeArgs = { synonymMinMax: args.synonymMinMax };
    let begin = new Date();
    for (let sum = first; sum <= last; ++sum) {
        // TODO: Fix this abomination
        args.sum = sum;
	let max = args.max;
        args.max = Math.min(args.max, args.sum);
        // TODO: return # of combos filtered due to note name match
        fillKnownNcSourceListMapForSum(map, sum, args.max, mergeArgs);
        args.max = max;
    }
    let d = new Duration(begin, new Date()).milliseconds;
    console.error(`getKnownNcSourceListMap: ${PrettyMs(d)}, size: ${map.size}`);
    return map;
};

//
//
let getUseSourceLists = (ncDataLists: NCDataList[], args: any): SourceList[] => {
    if (listIsEmpty(ncDataLists) || listIsEmpty(ncDataLists[0])) return [];

    const mergeArgs = { synonymMinMax: args.synonymMinMax };
    const sourceListMap = getUseNcSourceListMap(ncDataLists, mergeArgs);

    let begin = new Date();
    const sourceLists = buildSourceListsForUseNcData(ncDataLists, sourceListMap, mergeArgs);
    let end = new Duration(begin, new Date()).milliseconds;

    let sum = sourceLists.reduce((total, sl) => { return total + sl.length; }, 0);
    console.error(` buildSLforUseNCD(${PrettyMs(end)}), sourceLists(${sum}), ` +
	`list_merges(${list_merges}), merges(${merges})`);
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
//
let buildOrSourceNcCsvMap = (orSourceLists: SourceList[]): SourceNcCsvMap => {
    return orSourceLists.reduce((map: SourceNcCsvMap, sourceList: SourceList, index: number) => {
        const key = NameCount.listToSortedString(_.flatMap(sourceList, source => source.ncList));
        if (!map[key]) map[key] = [];
        map[key].push(index);
        return map;
    }, {});
};

// TODO: function name
//
const mergeCompatibleXorSources = (indexList: number[], sourceLists: SourceList[]): XorSource[] => {
    let compatible = true;

    let sources: SourceList = [];
    let srcSet = new Set<number>();
    
    // TODO: indexList.some()
    for (let [sourceListIndex, sourceIndex] of indexList.entries()) {
        const source = sourceLists[sourceListIndex][sourceIndex];
	if (!NameCount.listAddCountsToSet(source.primaryNameSrcList, srcSet)) {
	    compatible = false;
	    break;
	}
	sources.push(source);
    }
    if (compatible) {
	let primaryNameSrcList: NameCount.List = [];
	let ncList: NameCount.List = [];
	let primarySrcBits = CountBits.makeNew();
	for (let source of sources) {
	    primaryNameSrcList.push(...source.primaryNameSrcList);
	    primarySrcBits.orInPlace(source.primarySrcBits);
	    ncList.push(...source.ncList);
	}

        // I feel like this is still valid and worth removing or commenting
        //Assert(!_.isEmpty(primaryNameSrcList), 'empty primaryNameSrcList');
        let result: XorSource = {
            primaryNameSrcList,
            primarySrcBits,
            ncList,
        };
        return [result];
    }
    return [];
};

//
// TODO function name,
let mergeCompatibleXorSourceCombinations = (sourceLists: SourceList[]/*, args: MergeArgs*/): XorSource[] => {
    if (listIsEmpty(sourceLists)) return [];
    let begin = new Date();
    const numEmptyLists = listGetNumEmptySublists(sourceLists);
    if (numEmptyLists > 0) {
        // TODO: sometimes a sourceList is empty, like if doing $(cat required) with a
        // low clue count range (e.g. -c2,4). should that even be allowed?
        Assert(false, `numEmpty(${numEmptyLists}), numLists(${sourceLists.length})`);
    }
    let listArray = sourceLists.map(sourceList => [...Array(sourceList.length).keys()]);
    const peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    let combos = 0;
    let sourceList: XorSource[] = [];
    for (let indexList = peco.firstCombination(); indexList; ) {
        const mergedSources: XorSource[] = mergeCompatibleXorSources(indexList, sourceLists);
        sourceList.push(...mergedSources);
	indexList = peco.nextCombination();
	combos += 1;
    }
    let end = new Duration(begin, new Date()).milliseconds;
    console.error(` merge(${PrettyMs(end)}), combos(${combos})`);
    return sourceList;
};

//
//
let nextIndex = function(countList: number[], clueIndexes: number[]): boolean {
    // increment last index
    let index = clueIndexes.length - 1;
    clueIndexes[index] += 1;

    // while last index is maxed: reset to zero, increment next-to-last index, etc.
    while (clueIndexes[index] === ClueManager.getClueList(countList[index]).length) { // clueSourceList[index].list.length) {
        clueIndexes[index] = 0;
        if (--index < 0) {
            return false;
        }
	clueIndexes[index] += 1;
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
let next = (countList: number[], clueIndexes: number[]): FirstNextResult => {
    for (;;) {
        if (!nextIndex(countList, clueIndexes)) {
            return { done: true };
        }
        let ncList: NameCount.List = [];    // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
        let nameList: string[] = [];        // e.g. [ "pollock", "jackson" ]
        let srcCountStrList: string[] = []; // e.g. [ "white,fish:2", "moon,walker:4" ]
        if (!countList.every((count, index) => {
            let clue = ClueManager.getClueList(count)[clueIndexes[index]];
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
let first = (countList: number[], clueIndexes: number[]): FirstNextResult => {
    // TODO: _.fill?
    for (let index = 0; index < countList.length; ++index) {
	if (listIsEmpty(ClueManager.getClueList(countList[index]))) {
	    return { done: true };
	}
        clueIndexes[index] = 0;
    }
    clueIndexes[clueIndexes.length - 1] = -1;
    return next(countList, clueIndexes);
};

/*
//
//
let isAnyCompatibleOrSourceANDCompatibleWithSource = (
    compatibleSourceList: CompatibleOrSource.Type[],
    source: SourceData): boolean =>
{
    let compatible = false;
    for (let compatibleSource of compatibleSourceList) {
        // this should never happen because AND compatibility should have propagated up to the
        // container level, and we never should have been called if container is compatible.
        // (not so sure about this anymore, see note somewhere else on AND compatibility at
        // container level)
        Assert(!compatibleSource.andCompatible);
	// TODO:
        //compatible = CountBits.every 1 in 2 (compatibleSource.source.primarySrcBits, primarySrcArray);
        if (compatible) break;
    }
    return compatible;
};

//
//
let isAnyCompatibleOrSourceXORCompatibleWithSource = (
    compatibleSourceList: CompatibleOrSource.Type[],
    source: SourceData): boolean =>
{
    let compatible = false;
    for (let compatibleSource of compatibleSourceList) {
        // skip any sources that were already determined to be XOR incompatible or AND compatible
        // with command-line supplied --xor sources.
        if (!compatibleSource.xorCompatible || compatibleSource.andCompatible) continue;
        compatible = !CountBits.intersects(compatibleSource.source.primarySrcBits, source.primarySrcBits);
        if (compatible) break;
    }
    return compatible;
};

// OR == XOR || AND
//
let isSourceCompatibleWithEveryOrSource = (source: SourceData, orSourceList: OrSource[]) : boolean => {
    let compatible = true; // if no --or sources specified, compatible == true
    for (let orSource of orSourceList) {
        // TODO: skip calls to here if container.compatible = true  which may have been
        // determined in Precompute phase @ markAllANDCompatibleOrSources()
        // and skip the XOR check as well in this case.

        // First check for XOR compatibility
        compatible = isAnyCompatibleOrSourceXORCompatibleWithSource(
            orSource.sourceListContainer.compatibleSourceList, source);
        // Any XOR compatible sources, means "OR compatibility" was achieved with this OrSource
        if (compatible) continue;

        // Next check for AND compatibility, our last hope at achieving "OR compatibility"
        compatible = isAnyCompatibleOrSourceANDCompatibleWithSource(
            orSource.sourceListContainer.compatibleSourceList, source);
        if (!compatible) break;
    }
    return compatible;
};
*/

/*
//
//
let filterXorSourcesXORCompatibleWithSource = (xorSourceList: XorSource[], source: SourceData): XorSource[] => {
    let filteredXorSources: XorSource[] = [];
    for (let xorSource of xorSourceList) {
        if (isSourceXORCompatibleWithXorSource(source, xorSource)) {
            filteredXorSources.push(xorSource);
        }
    }
    return filteredXorSources;
};
*/

/*
//
//
let isSourceXORCompatibleWithXorSource = (source: SourceData, xorSource: XorSource): boolean => {
    const compatible = !CountBits.intersects(source.primarySrcBits, xorSource.primarySrcBits);
    return compatible;
};
*/

/*
//
//
let isSourceXORCompatibleWithAnyOrSource = (source: SourceData, xorSourceList: XorSource[]): boolean => {
    for (let xorSource of xorSourceList) {
        if (isSourceXORCompatibleWithXorSource(source, xorSource)) {
            return true;
        }
    }
    return false;
};
*/

//
//
let isSourceXORCompatibleWithAnyXorSource = (source: SourceData, xorSourceList: XorSource[]): boolean => {
    let compatible = listIsEmpty(xorSourceList); // empty list == compatible
    for (let xorSource of xorSourceList) {
        compatible = !CountBits.intersects(source.primarySrcBits, xorSource.primarySrcBits);
        if (compatible) break;
    }
    return compatible;
};

//
//
let isAnySourceCompatibleWithUseSources = (sourceList: SourceList, pcd: PreComputedData): boolean => {
    // TODO: this is why --xor is required with --or. OK for now. Fix later.
    if (listIsEmpty(pcd.useSourceLists.xor)) return true;

    let compatible = false;
    for (let source of sourceList) {
        //const xorSourceList 
	compatible = isSourceXORCompatibleWithAnyXorSource(source, pcd.useSourceLists.xor);
        // if there were --xor sources specified, and none are compatible with the
        // current source, no further compatibility checking is necessary; continue
        // to next source.
        if (!compatible) continue;

	// TODO:
	//compatible = isSourceCompatibleWithEveryOrSource(source, pcd.useSourceLists.orArgDataList);
        if (compatible) break;
    }
    return compatible;
};

//
//
let showBits = (bits: CountBits.Type, xorSourceList: XorSourceList): void => {
    for (let xorSource of xorSourceList) {
        let compatible = !CountBits.intersects(bits, xorSource.primarySrcBits);
	let bb: CountBits.Type = CountBits.and(bits, xorSource.primarySrcBits);
	let bbEmpty = bb.isEmpty();
        console.log(`${bits}\n` +
	    `${xorSource.primarySrcBits}\n` +
	    `${bb}\n` +
	    `------- compatible(${compatible}), empty(${bbEmpty}`);
    }
};

//
//
let showMergedSourcesBits = (mergedSourcesList: MergedSourcesList, xorSourceList: XorSourceList): void => {
    // TODO: this is why --xor is required with --or. OK for now. Fix later.
    if (listIsEmpty(xorSourceList)) return;

    let compatible = false;
    for (let mergedSources of mergedSourcesList) {
	console.log(`###############`);
        //const xorSourceList 
	showBits(mergedSources.primarySrcBits, xorSourceList);
    }
};

// Here lazySourceList is a list of lazy, i.e., not yet fully initialized source data for
// generated word-pairs (i.e. a "combo").
//
// Fully load the source data for each word, then perform a full merge on those sources.
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
            sourcesToMerge.push(sourceData);
        }
        const mergedSource = mergeSources(sourcesToMerge[0], sourcesToMerge[1], false) as SourceData;
        if (propertyCountsIsInBounds(mergedSource.synonymCounts, args.synonymMinMax)) {
            sourceList.push(mergedSource);
        }
    }
    return sourceList;
};

//
//
let makeSynonymFilePath = (name: string): string => {
    return Path.format({
        dir: `${DATA_DIR}syns`,
        base: `${name}.json`
    });
};

//
//
let synonymGetNameList = (name: string): string[] => {
    const path = makeSynonymFilePath(name);
    let json;
    let synListData: Synonym.ListData ;
    try {
        json = Fs.readFileSync(path, 'utf8');
        synListData = JSON.parse(json);
    } catch (err: any) {
        if (err.code !== 'ENOENT') {
            console.error(path);
            throw err;
        }
        return [];
    }
    return synListData.ignore ? []
        : synListData.list.filter(synData => !synData.ignore)
            .map(synData => synData.name);
};

// This is designed for purpose at the moment in that it assumes syn-max is at most 1.
// Therefore if synonymCounts.total is already one for the supplied sourceList, allow 
// no synonyms. If synonymCounts.total is zero, allow one.
//
// There are reasons for doing it this way. For one, it's more complicated and requires
// more recordkeeping to consider which of the two combo-word sources that were merged
// together into sourceList, were the source of a synonym total > 1. It could have been
// 1 from each, or 2 from one.
//
// If I'm ever *serious* about playing with syn-max > 1 though, I'll have to fix this.
//
let getSynonymCombos = (nameList: string[], sourceList: SourceList, args: any): string[] => {
    // NOTE: assumes -x2
    Assert(nameList.length === 2);

    // TODO: also check for --synmin/max here. if max = 0, exit.
    if (!args.use_syns) return [];
    const synonymCounts = getSynonymCounts(sourceList);
    // NOTE: assumes --syn-max = 1
    if (synonymCounts.total > 0) return [];

    let combos: string[] = [];
    const minMax = args.synonymMinMax;
    for (let index = 0; index < 2; ++index) {
        const synList = synonymGetNameList(nameList[index]);
        combos.push(...synList
            .map(synonym => [synonym, nameList[1 - index]])   // map to nameList
            .sort()                                           
            .map(nameList => nameList.toString()));           // map to nameCsv
    }
    return combos;
};

let getSynonymCombosForMergedSourcesList = (nameList: string[],
    mergedSourcesList: MergedSourcesList, args: any): string[] => 
{
    // NOTE: assumes -x2
    Assert(nameList.length === 2);

    // TODO: also check for --synmin/max here. if max = 0, exit.
    if (!args.use_syns) return [];
    //const synonymCounts = getSynonymCounts(sourceList);
    // NOTE: assumes --syn-max = 1
    //if (synonymCounts.total > 0) return [];

    let combos: string[] = [];
    const minMax = args.synonymMinMax;
    for (let index = 0; index < 2; ++index) {
        const synList = synonymGetNameList(nameList[index]);
        combos.push(...synList
            .map(synonym => [synonym, nameList[1 - index]])   // map to nameList
            .sort()                                           
            .map(nameList => nameList.toString()));           // map to nameCsv
    }
    return combos;
};

let setPrimarySrcBits = (sourceList: SourceBase[]): void => {
    for (let source of sourceList) {
	source.primarySrcBits = CountBits.makeFrom(NameCount.listToCountList(source.primaryNameSrcList));
    }
};

let anyMergedSourcesHasBits = (mergedSourcesList: MergedSourcesList, bits: CountBits.Type) : boolean => {
    for (const mergedSources of mergedSourcesList) {
	//if (CountBits.and(bits, mergedSources.primarySrcBits).equals(bits)) return true;
	if (mergedSources.primarySrcBits.equals(bits)) return true;
    }
    return false;
};

//
// args:
//   synonymMinMax
//
let getCombosForUseNcLists = (sum: number, max: number, pcd: PreComputedData, args: any): string[] => {
    let hash: StringAnyMap = {};
    let combos: string[] = [];

    let comboCount = 0;
    let totalVariations = 0;
    let numCacheHits = 0;
    let numMergeIncompatible = 0;
    let numUseIncompatible = 0;
    let isany = 0;
    
    const MILLY = 1000000n;
    const start = process.hrtime.bigint();

    const lazyMergeArgs = {
        synonymMinMax: args.synonymMinMax
//	,lazy: true
    };

    // Given a sum, such as 4, and a max # of numbers to combine, such as 2, generate
    // an array of addend arrays ("count lists"), for each 2 <= N <= max, that add up
    // to that sum, such as [ [1, 3], [2, 2] ]
    let countListArray: number[][] = Peco.makeNew({ sum, max }).getCombinations(); 

    // for each countList
    countListArray.forEach((countList: number[]) => {
        comboCount += 1;

        //console.log(`sum(${sum}) max(${max}) countList: ${Stringify(countList)}`);
        let clueIndexes: number[] = [];
        let result = first(countList, clueIndexes);
        if (result.done) return; // continue; 

        let numVariations = 1;

        // this is effectively Peco.getCombinations().forEach()
        let firstIter = true;
        while (!result.done) {
            if (!firstIter) {
                result = next(countList, clueIndexes);
                if (result.done) break;
                numVariations += 1;
            } else {
                firstIter = false;
            }


            if (0 && (result.nameList!.toString() == "nineteenth century,polar")) {
                console.error(`hit: ${result.nameList}`);
                WW = true
            } else {
                WW = false;
            }


            const key: string = NameCount.listToString(result.ncList!);
            let cacheHit = false;
            //let sourceList: AnySourceData[];
	    let mergedSourcesList: MergedSourcesList = [];
            if (!hash[key]) {
		// TODO: use native here
                if (1) { // No. JS is faster
		    mergedSourcesList = mergeAllCompatibleSources2(result.ncList!, pcd.sourceListMap);
		    //validatePrimarySrcBits(mergedSourcesList);
		} else {
		    //mergedSourcesList = NativeComboMaker.mergeAllCompatibleSources(result.ncList!);
		    //setPrimarySrcBits(mergedSourcesList); // TODO
		}
                if (listIsEmpty(mergedSourcesList)) {
		    ++numMergeIncompatible;
		}
                hash[key] = { mergedSourcesList };
            } else {
                mergedSourcesList = hash[key].mergedSourcesList;
                cacheHit = true;
                numCacheHits += 1;
            }

            if (logging) console.log(`  found compatible sources: ${!listIsEmpty(mergedSourcesList)}`);

            // failed to find any compatible combos
            if (listIsEmpty(mergedSourcesList)) continue;

            if (hash[key].isCompatible === undefined) {
                //sourceList = loadAndMergeSourceList(sourceList as LazySourceData[], { synonymMinMax: args.synonymMinMax });
		// can't i precompute this state also?
		isany += 1;
		if (0) {
                    //hash[key].isCompatible = isAnySourceCompatibleWithUseSources(mergedSourcesList as SourceList, pcd)
		} else {
		    let flag = false;
		    if (WW) {
			let bits = CountBits.makeFrom([35,36,37,38,73,79,80]);
			if (anyMergedSourcesHasBits(mergedSourcesList, bits)) {
			    flag = true;
			}
		    }
                    hash[key].isCompatible = NativeComboMaker.isAnySourceCompatibleWithUseSources(mergedSourcesList, flag);
		    if (WW) {
			console.error(`compat: ${hash[key].isCompatible}`);
			showMergedSourcesBits(mergedSourcesList, pcd.useSourceLists.xor);
		    }
		}
            }
            if (hash[key].isCompatible) {
                combos.push(result.nameList!.toString());
                //combos.push(...getSynonymCombosForMergedSourcesList(result.nameList!, mergedSourcesList, args));
            } else if (!cacheHit) {
                numUseIncompatible += 1;
            }
        }
        totalVariations += numVariations;
    });

    let duration = (process.hrtime.bigint() - start) / MILLY;
    Debug(`sum(${sum}) combos(${comboCount}) variations(${totalVariations}) cacheHits(${numCacheHits}) ` +
        `merge-incompatible(${numMergeIncompatible}) use-incompatible(${numUseIncompatible}) ` +
        `actual(${totalVariations - numCacheHits - numUseIncompatible}) ${duration}ms`);

    if (1) {
        console.error(`sum(${sum}) combos(${comboCount}) ` +
	    `variations(${totalVariations}) cacheHits(${numCacheHits}) ` +
            `merge-incompat(${numMergeIncompatible}) ` +
	    `use-incompat(${numUseIncompatible}) ` +
            `actual(${totalVariations - numCacheHits - numUseIncompatible}) ` +
	    `isany(${isany}) ${duration}ms `);
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
export let makeCombosForSum = (sum: number, max: number, args: any): string[] => {
    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
        // TODO: whereever this is actually enforced:
        // console.error(`Enforcing max results: ${args.maxResults}`);
    }
    // TODO move this a layer or two out; use "validateArgs" 
    Assert(_.isEmpty(args.require), 'require not yet supported');
    Assert(!args.sources, 'sources not yet supported');
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
            xormm: args.xormm,
            or: args.or,
            fast: args.fast,
            load_max: ClueManager.getNumPrimarySources(),
            parallel: true,
            use_syns: args.use_syns,
            synonymMinMax: args.synonymMinMax
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
export let makeCombos = (args: any): any => {
    Assert(args.sum, 'missing sum');
    let sumRange: number[] = args.sum.split(',').map(_.toNumber);
    Assert(sumRange.length <= 2, `funny sum (${sumRange.length})`);

    Debug(`++combos, sum: ${sumRange} max: ${args.max} use-syns: ${args.use_syns}`);

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
        PCD = preCompute(first, last, args);
	//console.log(JSON.stringify(PCD.useSourceLists.xor));
        let comboMap = {};
        for (let sum = first; sum <= last; ++sum) {
            // TODO: Fix this abomination
            args.sum = sum;
            let max = args.max;
            args.max = Math.min(args.max, args.sum);
            // TODO: return # of combos filtered due to note name match
            const comboList = makeCombosForSum(sum, args.max, args);
            args.max = max;
            total += comboList.length;
            const filterResult = ClueManager.filter(comboList, sum, comboMap);
        }
        let d = new Duration(begin, new Date()).milliseconds;
        console.error(`--combos: ${PrettyMs(d)}, oob: ${oob}`);
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

// TODO: combinationNcDataListFromNcLists
function combinationNcDataList (indexList: number[], ncLists: NameCount.List[]): NCDataList {
    return indexList.map((ncIndex: number, listIndex: number) => Object({ ncList: ncLists[listIndex][ncIndex]}));
}

function ncListsToCombinations (ncLists: NameCount.List[]): NameCount.List[] {
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]),       // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)              // sum of lengths of nclists
    }).getCombinations()
        .map((indexList: number[]) => combinationNcList(indexList, ncLists));
}

// TODO: tuple type interface
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
            if (minMax) ncData.synonymMinMax = minMax;
            return ncData;
        });
}

function getCombinationNcLists (useArgsList: string[]): any {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    return useArgsList.map(useArg => useArg.split(','))
        .map((nameOrNcStrList: string[]) => nameOrNcStrListToKnownNcLists(nameOrNcStrList))
        .map((knownNcLists: NameCount.List[]) => ncListsToCombinations(knownNcLists));
}

let isSingleNumericDigit = (arg: string): boolean => {
    return arg.length === 1 && arg[0] >= '0' && arg[0] <= '9';
}

let splitArgListAndMinMax = (argList: string[]): [MinMax.Type, string[]] => {
    Assert(argList.length > 2);
    const minArg = argList[argList.length - 2];
    const maxArg = argList[argList.length - 1];
    Assert(isSingleNumericDigit(minArg) && isSingleNumericDigit(maxArg));
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

function getCombinationNcDataLists (useArgsList: string[], hasMinMax: boolean = false): any {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    if (!useArgsList) return [];
    return useArgsList.map(useArg => useArgToMinMaxNameListTuple(useArg, hasMinMax))
        .map(minMaxNameListTuple => {
	    return [minMaxNameListTuple[0], nameOrNcStrListToKnownNcLists(minMaxNameListTuple[1])];  // nameOrNcStrList 
	})
        .map(minMaxNcListsTuple => minMaxNcListsTupleToNcDataCombinations(minMaxNcListsTuple)); // knownNcLists
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
        // TODO: List.toIndexList()
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]), // keys of array are 0..ncList.length-1
        // TODO: List.sumOfSublistLengths()
        max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)        // sum of lengths of nclists
    }).getCombinations()
        .map((ncListIndexes: number[]) => combinationNcDataList(ncListIndexes, combinationNcLists));
}

// TODO: combinationNcDataListFromNcDataLists
// same as combinationNcDataList but takes NCDataList[] instead of NameCount.List[]
function ncDataCombinationsToNcDataList (indexList: number[], ncDataLists: NCDataList[]): NCDataList {
    return indexList.map((ncDataIndex: number, listIndex: number) => ncDataLists[listIndex][ncDataIndex]);
}

// TODO: ncDataListsToNcDataCombinations
// a version of combinationsToNcDataLists that takes NCDataList[] instead of NameCount.List[]
function ncDataCombinationsToNcDataLists (combinationNcDataLists: NCDataList[]): NCDataList[] {
    Debug(`ncDataCombinationsToNcDataLists() ncDataCombinationLists: ${Stringify(combinationNcDataLists)}`);
    if (listIsEmpty(combinationNcDataLists)) return [ [] ];
    return Peco.makeNew({
        // TODO: List.toIndexList()
        listArray: combinationNcDataLists.map(ncDataList => [...Array(ncDataList.length).keys()]), // 0..ncDataList.length-1
        // TODO: List.sumOfSublistLengths()
        max: combinationNcDataLists.reduce((sum, ncDataList) => sum + ncDataList.length, 0)        // sum of lengths of ncDataLists
    }).getCombinations()
        .map((indexList: number[]) => ncDataCombinationsToNcDataList(indexList, combinationNcDataLists));
}

/*
//
//
function buildAllUseNcLists (useArgsList: string[]): NameCount.List[] {
    return combinationsToNcLists(getCombinationNcLists(useArgsList));
}
*/

let sumOfNcDataListCounts = (ncDataList: NCDataList): number => {
    let sum = 0;
    for (let ncData of ncDataList) {
	sum += NameCount.listCountSum(ncData.ncList);
    }
    return sum;
}

//
//
function buildAllUseNcDataLists (useArgsList: string[]): NCDataList[] {
    const combinationNcLists = getCombinationNcLists(useArgsList);
    //console.log(`combinationNcLists: ${Stringify2(combinationNcLists)}`);
    const ncDataLists = combinationsToNcDataLists(combinationNcLists)
    //console.log(`ncDataList: ${Stringify2(ncDataList)}`);
    return ncDataLists;
    //.filter((ncDataList: NCDataList) => sumOfNcDataListCounts(ncDataList) <= maxSum);
}

// for combining --xor with --xormm
//
let buildCombinedUseNcDataLists = (useArgsList: string[],
    minMaxUseArgsList: string[]): NCDataList[] =>
{
    const standardNcDataLists = getCombinationNcDataLists(useArgsList);
    const minMaxNcDataLists = getCombinationNcDataLists(minMaxUseArgsList, true);
    const combinedNcDataLists = [...standardNcDataLists, ...minMaxNcDataLists];
    // TODO: if (listIsEmpty(combinedNcDataLists)) combinedNcDataLists.push([]);
    const ncDataLists = ncDataCombinationsToNcDataLists(combinedNcDataLists);
    return ncDataLists;
    //.filter((ncDataList: NCDataList) => sumOfNcDataListCounts(ncDataList) <= maxSum);
}

//
//
let buildOrArgData = (sourceList: SourceList): OrArgData => {
    const orArgData = initOrArgData();
    for (let source of sourceList) {
        const orSource = initOrSource(source);
        orArgData.orSourceList.push(orSource);
    }
    return orArgData;
};

//
let buildOrArgDataList = (sourceLists: SourceList[]): OrArgDataList => {
    let orArgDataList: OrArgDataList = [];
    for (let sourceList of sourceLists) {
        const orArgData: OrArgData = buildOrArgData(sourceList);
        orArgDataList.push(orArgData);
    }
    return orArgDataList;
};

// Given a list of XorSources, and a list of OrSources, TODO
//
let markAllXORCompatibleOrSources = (xorSourceList: XorSource[], orArgDataList: OrArgDataList): void => {
    for (let orArgData of orArgDataList) {
        const orSourceList = orArgData.orSourceList;
        for (let orSource of orSourceList) {
            if (isSourceXORCompatibleWithAnyXorSource(orSource.source, xorSourceList)) {
                orSource.xorCompatible = true;
            }
        }
    }
}

//
//
let buildUseSourceListsFromNcData = (sourceListMap: Map<string, AnySourceData[]>, args: any): UseSourceLists => {
    // XOR first
    let xor0 = new Date();
    let xorSourceList: XorSourceList = NativeComboMaker.mergeCompatibleXorSourceCombinations(
	args.allXorNcDataLists, Array.from(sourceListMap.entries()));
    setPrimarySrcBits(xorSourceList);
    let xdur = new Duration(xor0, new Date()).milliseconds;
    console.error(` Native.mergeCompatibleXorSourceCombinations(${PrettyMs(xdur)})`);

    // OR next
    let or0 = new Date();
    const mergeArgs = { synonymMinMax: args.synonymMinMax };
    let orArgDataList = buildOrArgDataList(buildSourceListsForUseNcData(
	args.allOrNcDataLists, sourceListMap, mergeArgs));
    let odur = new Duration(or0, new Date()).milliseconds;
    console.error(`orArgDataList(${orArgDataList.length}), build(${PrettyMs(odur)})`);

    // Thoughts on AND compatibility of OrSources:
    // Just because (one sourceList of) an OrSource is AND compatible with an
    // XorSource doesn't mean the OrSource is redundant and can be ignored
    // (i.e., the container cannot be marked as "compatible.") We still need
    // to check the possibility that any of the other XOR-but-not-AND-compatible
    // sourceLists could be AND-compatible with the generated-combo sourceList.
    // So, a container can be marked compatible if and only if there are no
    // no remaining XOR-compatible sourceLists.
    //TODO: markAllANDCompatibleOrSources(xorSourceList, orSourceList);
    let mark0 = new Date();
    markAllXORCompatibleOrSources(xorSourceList, orArgDataList);
    let mdur = new Duration(mark0, new Date()).milliseconds;
    console.error(`mark(${PrettyMs(mdur)})`);

    NativeComboMaker.setOrArgDataList(orArgDataList);

    /*
    if (1) {
        useSourceList.forEach((source, index) => {
            const orSource: OrSource = source.orSource!;
            if (orSource && !orSource.sourceLists) console.log(`orSourceLists ${orSource.sourceLists}`);
            if (orSource && orSource.sourceLists && listIsEmpty(orSource.sourceLists)) console.log(`EMPTY4`);
        });
    }
    */
    return { xor: xorSourceList, orArgDataList: orArgDataList };
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
export let preCompute = (first: number, last: number, args: any): PreComputedData => {
    const begin = new Date();
    const build1 = new Date();
    args.allXorNcDataLists = buildCombinedUseNcDataLists(args.xor, args.xormm);
    const d1 = new Duration(build1, new Date()).milliseconds;
    console.error(` build1(${PrettyMs(d1)})`);
    //console.error(`allXorNcDataLists(${args.allXorNcDataLists.length})`);

    const build2 = new Date();
    args.allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or) : [ [] ];
    const d2 = new Duration(build2, new Date()).milliseconds;
    console.error(` build2(${PrettyMs(d2)})`);
    
    const sourceListMap = getKnownNcSourceListMap(first, last, args);
    //const sourceListMap = getUseNcSourceListMap(args.allXorNcDataLists, args);

    const build3 = new Date();
    const useSourceLists = buildUseSourceListsFromNcData(sourceListMap, args);
    const d3 = new Duration(build3, new Date()).milliseconds;
    console.error(` build3(${PrettyMs(d3)})`);

    const d = new Duration(begin, new Date()).milliseconds;
    console.error(`Precompute(${PrettyMs(d)})`);

    if (listIsEmpty(useSourceLists.xor) && args.xor)
        // || (listIsEmpty(orSourceList) && args.or))
    {
        console.error('incompatible --xor/--or params');
        process.exit(-1);
    }
    return { useSourceLists, sourceListMap };
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
