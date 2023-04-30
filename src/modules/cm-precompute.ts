//
// cm-precompute.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

//const ResultMap   = require('../../types/result-map');
const Peco        = require('../../modules/peco');
//const Log         = require('../../modules/log')('combo-maker');
//const My          = require('../../modules/util');
const NativeComboMaker = require('../../native-modules/experiment/build/Release/experiment.node');

const Assert      = require('assert');
const Debug       = require('debug')('cm-precompute');
const Duration    = require('duration');
//const Fs          = require('fs-extra');
//const Path        = require('path');
const PrettyMs    = require('pretty-ms');
//const stringify   = require('javascript-stringify').stringify;
const Stringify  = require('stringify-object');

import * as Clue from '../types/clue';
import * as ClueManager from './clue-manager';
import * as ComboMaker from './combo-maker';
import * as CountBits from '../types/count-bits-fastbitset';
import * as MinMax from '../types/min-max';
import * as NameCount from '../types/name-count';
import * as Sentence from '../types/sentence';
import * as Source from './source';

import { ValidateResult } from './validator';

/////////

interface StringBoolMap {
    [key: string]: boolean; // for now; eventually maybe array of string (sorted primary nameSrcCsv)
}

type XorSource = Source.Data;
type XorSourceList = XorSource[];

interface OrSourceData {
    source: Source.Data;
    xorCompatible: boolean;
    andCompatible: boolean;
}
type OrSourceList = OrSourceData[];

// One OrArgData contains all of the data for a single --or argument.
//
interface OrArgData {
    orSourceList: OrSourceList;
    compatible: boolean;
}
type OrArgDataList = OrArgData[];

interface UseSourceLists {
    xor: XorSourceList;
    orArgDataList: OrArgDataList;
}

export interface Data {
    useSourceLists: UseSourceLists;
    sourceListMap: Map<string, Source.AnyData[]>;
}

interface NCData {
    ncList: NameCount.List;
    synonymMinMax?: MinMax.Type;
}
type NCDataList = NCData[];

//////////

let listIsEmpty = (list: any[]): boolean => {
    return list.length === 0;
};

const isSingleNumericDigit = (arg: string): boolean => {
    return (arg.length === 1) && (arg[0] >= '0') && (arg[0] <= '9');
}

const splitArgListAndMinMax = (argList: string[]): [MinMax.Type, string[]] => {
    Assert(argList.length > 2);
    const minArg = argList[argList.length - 2];
    const maxArg = argList[argList.length - 1];
    Assert(isSingleNumericDigit(minArg) && isSingleNumericDigit(maxArg));
    return [MinMax.init(minArg, maxArg), argList.slice(0, argList.length - 2)];
};

const useArgToMinMaxNameListTuple = (useArg: string, hasMinMax: boolean):
    [MinMax.Type|undefined, string[]] =>
{
    let argList = useArg.split(',');
    let minMax: MinMax.Type|undefined = undefined;
    if (hasMinMax) {
        [minMax, argList] = splitArgListAndMinMax(argList);
    }
    return [minMax, argList];
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

// TODO: tuple type interface
function minMaxNcListsTupleToNcDataCombinations (minMaxNcListsTuple: any[]): NCDataList[] {
    const minMax = minMaxNcListsTuple[0];
    const ncLists = minMaxNcListsTuple[1] as NameCount.List[];
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]), // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)        // sum of lengths of nclists
    }).getCombinations().map((indexList: number[]) => {
        let ncData: NCData = {
	    ncList: combinationNcList(indexList, ncLists)
	};
        if (minMax) ncData.synonymMinMax = minMax;
        return ncData;
    });
}

function getCombinationNcDataLists (useArgsList: string[], hasMinMax: boolean = false): any {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    if (!useArgsList) return [];
    return useArgsList
	.map(useArg => useArgToMinMaxNameListTuple(useArg, hasMinMax))
        .map(minMaxNameListTuple => {
	    return [minMaxNameListTuple[0], nameOrNcStrListToKnownNcLists(minMaxNameListTuple[1])];  // nameOrNcStrList 
	})
        .map(minMaxNcListsTuple => minMaxNcListsTupleToNcDataCombinations(minMaxNcListsTuple)); // knownNcLists
}

// TODO: combinationNcDataListFromNcDataLists
// same as combinationNcDataList but takes NCDataList[]
// instead of NameCount.List[]
function ncDataCombinationsToNcDataList (indexList: number[],
    ncDataLists: NCDataList[]): NCDataList
{
    return indexList.map((ncDataIndex: number, listIndex: number) =>
	ncDataLists[listIndex][ncDataIndex]);
}

// TODO: ncDataListsToNcDataCombinations
// same as combinationsToNcDataLists that takes NCDataList[]
// instead of NameCount.List[]
function ncDataCombinationsToNcDataLists (combinationNcDataLists: NCDataList[]): NCDataList[] {
    Debug(`ncDataCombinationsToNcDataLists() ` +
	`ncDataCombinationLists: ${Stringify(combinationNcDataLists)}`);
    if (listIsEmpty(combinationNcDataLists)) return [ [] ];
    return Peco.makeNew({
        // TODO: List.toIndexList()
        listArray: combinationNcDataLists.map(ncDataList =>
	    [...Array(ncDataList.length).keys()]), // 0..ncDataList.length-1
        // TODO: List.sumOfSublistLengths()
        max: combinationNcDataLists.reduce((sum, ncDataList) =>
	    sum + ncDataList.length, 0) // sum of lengths of ncDataLists
    }).getCombinations()
        .map((indexList: number[]) => ncDataCombinationsToNcDataList(indexList, combinationNcDataLists));
}

// for combining --xor with --xormm
//
const buildCombinedUseNcDataLists = (useArgsList: string[],
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

function ncListsToCombinations (ncLists: NameCount.List[]): NameCount.List[] {
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()]),       // keys of array are 0..ncList.length
        max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)              // sum of lengths of nclists
    }).getCombinations()
        .map((indexList: number[]) => combinationNcList(indexList, ncLists));
}

function getCombinationNcLists (useArgsList: string[]): any {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    return useArgsList
	.map(useArg => useArg.split(','))
        .map((nameOrNcStrList: string[]) => nameOrNcStrListToKnownNcLists(nameOrNcStrList))
        .map((knownNcLists: NameCount.List[]) => ncListsToCombinations(knownNcLists));
}

// TODO: combinationNcDataListFromNcLists
const combinationNcDataList = (indexList: number[], ncLists: NameCount.List[]):
    NCDataList =>
{
    return indexList.map((ncIndex: number, listIndex: number) =>
	Object({ ncList: ncLists[listIndex][ncIndex]}));
}

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

function buildAllUseNcDataLists (useArgsList: string[]): NCDataList[] {
    const combinationNcLists = getCombinationNcLists(useArgsList);
    const ncDataLists = combinationsToNcDataLists(combinationNcLists)
    return ncDataLists;
    //.filter((ncDataList: NCDataList) => sumOfNcDataListCounts(ncDataList) <= maxSum);
}

//////////

let getSourceList = (nc: NameCount.Type/*, args: MergeArgs*/): Source.List => {
    const sourceList: Source.List = [];
    ClueManager.getKnownSourceMapEntries(nc)
        .forEach((sourceData: ClueManager.SourceData) => {
            sourceList.push(...sourceData.results
                .map((result: ValidateResult) => Source.makeData(nc, result)));
        });
    return sourceList;
};

const addNcListToSourceListMap = (ncList: NameCount.List,
    map: Map<string, Source.AnyData[]>/*, mergeArgs: MergeArgs*/) : void =>
{
    for (let nc of ncList) {
	const key = NameCount.toString(nc)
	if (!map.has(key)) {
	    const sourceList = getSourceList(nc, /*mergeArgs*/);
	    if (listIsEmpty(sourceList)) {
		throw new Error(`empty sourceList: ${NameCount.toString(nc)}`);
	    }
	    map.set(key, sourceList);
	}
    }
};

const fillKnownNcSourceListMapForSum = (map: Map<string, Source.AnyData[]>,
    sum: number, max: number/*, mergeArgs: MergeArgs*/) : void =>
{
    // Given a sum, such as 4, and a max # of numbers to combine, such as 2, generate
    // an array of addend arrays ("count lists"), for each 2 <= N <= max, that add up
    // to that sum, such as [ [1, 3], [2, 2] ]
    let countListArray: number[][] = Peco.makeNew({ sum, max }).getCombinations(); 
    // for each countList
    countListArray.forEach((countList: number[]) => {
        let sourceIndexes: number[] = [];
        let result = ComboMaker.first(countList, sourceIndexes);
        if (result.done) return; // continue; 

        let firstIter = true;
        while (!result.done) {
            if (firstIter) {
                firstIter = false;
            } else {
                result = ComboMaker.next(countList, sourceIndexes);
                if (result.done) break;
            }
	    addNcListToSourceListMap(result.ncList!, map/*, mergeArgs*/);
	}
    });
};

const getKnownNcSourceListMap = (first: number, last: number,
    args: any): Map<string, Source.AnyData[]> =>
{
    // NOTE: correct, but hacky
    last = ClueManager.getNumPrimarySources();
    let map = new Map<string, Source.AnyData[]>();
    //const mergeArgs = { synonymMinMax: args.synonymMinMax };
    let begin = new Date();
    for (let sum = first; sum <= last; ++sum) {
        // TODO: Fix this abomination
        args.sum = sum;
	let max = args.max;
        args.max = Math.min(args.max, args.sum);
        // TODO: return # of combos filtered due to note name match
        fillKnownNcSourceListMapForSum(map, sum, args.max/*, mergeArgs*/);
        args.max = max;
    }
    let d = new Duration(begin, new Date()).milliseconds;
    console.error(`getKnownNcSourceListMap: ${PrettyMs(d)}, size: ${map.size}`);
    return map;
};

// TODO: move to Source, setAllSourceBits
const setPrimarySrcBits = (sourceList: Source.List): void => {
    for (let source of sourceList) {
	source.sourceBits = CountBits.makeFrom(
	    Sentence.legacySrcList(source.primaryNameSrcList));
    }
};

// TODO: move to Source?
const mergeSources = (source1: Source.Data, source2: Source.Data):
    Source.Data =>
{
    const primaryNameSrcList = [...source1.primaryNameSrcList, ...source2.primaryNameSrcList];
    const sourceBits = CountBits.or(source1.sourceBits, source2.sourceBits);
    const usedSources = ComboMaker.mergeUsedSources(source1.usedSources, source2.usedSources);
    const ncList = [...source1.ncList, ...source2.ncList];
    /*
    if (lazy) {
        Assert(ncList.length === 2, `ncList.length(${ncList.length})`);
        source1 = source1 as Source.LazyData;
        source2 = source2 as Source.LazyData;
        const result: Source.LazyData = {
            primaryNameSrcList,
	    sourceBits,
	    usedSources,
            ncList,
            //synonymCounts: Clue.PropertyCounts.merge(
            //  getSynonymCountsForValidateResult(source1.validateResults[0]),
            //  getSynonymCountsForValidateResult(source2.validateResults[0])),
            validateResults: [
                (source1 as Source.LazyData).validateResults[0],
                (source2 as Source.LazyData).validateResults[0]
            ]
        };
        return result;
    }
    source1 = source1 as Source.Data;
    source2 = source2 as Source.Data;
    */
    return {
        primaryNameSrcList,
	sourceBits,
	usedSources,
        ncList
    };
};

const mergeCompatibleSources = (source1: Source.Data, source2: Source.Data):
    Source.Data[] =>
{
    return Source.isXorCompatible(source1, source2)
        ? [mergeSources(source1, source2)] : [];
};

const mergeCompatibleSourceLists = (sourceList1: Source.List,
    sourceList2: Source.List): Source.List =>
{
    let result: Source.List = [];
    for (const source1 of sourceList1) {
        for (const source2 of sourceList2) {
            result.push(...mergeCompatibleSources(source1, source2));
        }
    }
    return result;
};

const mergeAllCompatibleSources = (ncList: NameCount.List,
    sourceListMap: Map<string, Source.AnyData[]>/*, args: MergeArgs*/): Source.AnyData[] =>
{
    // because **maybe** broken for > 2 below
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    // TODO: reduce (or some) here
    let sourceList = sourceListMap.get(NameCount.toString(ncList[0])) as Source.AnyData[];
    for (let ncIndex = 1; ncIndex < ncList.length; ++ncIndex) {
        const nextSourceList: Source.AnyData[] =
	    sourceListMap.get(NameCount.toString(ncList[ncIndex])) as Source.AnyData[];
        sourceList = mergeCompatibleSourceLists(sourceList, nextSourceList);
        // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
        if (listIsEmpty(sourceList)) break;
    }
    return sourceList;
};

const buildSourceListsForUseNcData = (useNcDataLists: NCDataList[],
    sourceListMap: Map<string, Source.AnyData[]>/*, args: MergeArgs*/): Source.List[] =>
{
    let sourceLists: Source.List[] = [];
    // TODO: This is to prevent duplicate sourceLists. I suppose I could use a Set or Map, above?
    let hashList: StringBoolMap[] = [];
    for (let ncDataList of useNcDataLists) {
        for (let [sourceListIndex, useNcData] of ncDataList.entries()) {
            if (!sourceLists[sourceListIndex]) sourceLists.push([]);
            if (!hashList[sourceListIndex]) hashList.push({});
            // give priority to any min/max args specific to an NcData, for example, through --xormm,
            // but fallback to the values we were called with
            //const mergeArgs = useNcData.synonymMinMax ? { synonymMinMax: useNcData.synonymMinMax } : args;
            const sourceList = mergeAllCompatibleSources(useNcData.ncList, sourceListMap/*, mergeArgs*/) as Source.List;
            for (let source of sourceList) {
                let key = NameCount.listToString(_.sortBy(source.primaryNameSrcList, NameCount.count));
                if (!hashList[sourceListIndex][key]) {
                    sourceLists[sourceListIndex].push(source as Source.Data);
                    hashList[sourceListIndex][key] = true;
                }
            }
        }
    }
    return sourceLists;
};

const initOrSource = (source: Source.Data): OrSourceData => {
    return {
        source,
        xorCompatible: false,
        andCompatible: false
    };
};

const initOrArgData = (): OrArgData => {
    return {
        orSourceList: [],
        compatible: false
    };
}

const buildOrArgData = (sourceList: Source.List): OrArgData => {
    const orArgData = initOrArgData();
    for (let source of sourceList) {
        const orSource = initOrSource(source);
        orArgData.orSourceList.push(orSource);
    }
    return orArgData;
};

const buildOrArgDataList = (sourceLists: Source.List[]): OrArgDataList => {
    let orArgDataList: OrArgDataList = [];
    for (let sourceList of sourceLists) {
        const orArgData: OrArgData = buildOrArgData(sourceList);
        orArgDataList.push(orArgData);
    }
    return orArgDataList;
};

// Given a list of XorSources, and a list of OrSources, TODO
//
const markAllXORCompatibleOrSources = (xorSourceList: XorSource[],
    orArgDataList: OrArgDataList): void =>
{
    for (let orArgData of orArgDataList) {
        const orSourceList = orArgData.orSourceList;
        for (let orSource of orSourceList) {
            if (Source.isXorCompatibleWithAnySource(orSource.source, xorSourceList)) {
                orSource.xorCompatible = true;
            }
        }
    }
}

let buildUseSourceListsFromNcData = (sourceListMap: Map<string, Source.AnyData[]>, args: any): UseSourceLists => {
    // XOR first
    let xor0 = new Date();
    let xorSourceList: XorSourceList = NativeComboMaker.mergeCompatibleXorSourceCombinations(
	args.allXorNcDataLists, Array.from(sourceListMap.entries())); // TODO? [...sourceListMap.entries()]
    setPrimarySrcBits(xorSourceList);
    let xdur = new Duration(xor0, new Date()).milliseconds;
    console.error(` Native.mergeCompatibleXorSourceCombinations(${PrettyMs(xdur)})`);

    // OR next
    let or0 = new Date();
    //const mergeArgs = { synonymMinMax: args.synonymMinMax };
    let orArgDataList = buildOrArgDataList(buildSourceListsForUseNcData(
	args.allOrNcDataLists, sourceListMap/*, mergeArgs*/));
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

export const preCompute = (first: number, last: number, args: any): Data => {
    const begin = new Date();
    const build1 = new Date();
    args.allXorNcDataLists = buildCombinedUseNcDataLists(args.xor, args.xormm);
    const d1 = new Duration(build1, new Date()).milliseconds;
    console.error(` buildAllXorNcDataLists(${PrettyMs(d1)})`);
    //console.error(`allXorNcDataLists(${args.allXorNcDataLists.length})`);

    const build2 = new Date();
    args.allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or) : [ [] ];
    const d2 = new Duration(build2, new Date()).milliseconds;
    console.error(` buildAllOrNcDataLists(${PrettyMs(d2)})`);
    
    const sourceListMap = getKnownNcSourceListMap(first, last, args);
    //const sourceListMap = getUseNcSourceListMap(args.allXorNcDataLists, args);

    const build3 = new Date();
    const useSourceLists = buildUseSourceListsFromNcData(sourceListMap, args);
    const d3 = new Duration(build3, new Date()).milliseconds;
    console.error(` buildUseSourceLists(${PrettyMs(d3)})`);

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
