//
// cm-precompute.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Peco        = require('../../modules/peco');

const NativeComboMaker = require('../../../build/Release/experiment.node');
//const NativeComboMaker = require('../../../build/Debug/experiment.node');

const Assert      = require('assert');
const Debug       = require('debug')('cm-precompute');
const Duration    = require('duration');
const PrettyMs    = require('pretty-ms');
const Stringify   = require('stringify-object');

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

interface UseSourceListSizes {
    xor: number;
//    orArgDataList: OrArgDataList;
}

export interface Data {
    xor: Source.List;
    sourceListMap: Map<string, Source.AnyData[]>;
}

export interface Result {
    success: boolean;
    data?: Data;
}

// TODO: NameCount.ListContainer
interface NCData {
    ncList: NameCount.List;
}
type NCDataList = NCData[];

//////////

const listIsEmpty = (list: any[]): boolean => {
    return list.length === 0;
};

const isSingleNumericDigit = (arg: string): boolean => {
    return (arg.length === 1) && (arg[0] >= '0') && (arg[0] <= '9');
};

//////////

const ncAsKnownNcList = (nc: NameCount.Type): NameCount.List => {
    return ClueManager.isKnownNc(nc) ? [nc] : [];
};

const getKnownNcListForName = (name: string): NameCount.List => {
    const countList = ClueManager.getCountListForName(name);
    Assert(!_.isEmpty(countList), `not a valid clue name: '${name}'`);
    return countList.map(count => ({ name, count }));
};

//
// Given a list of names and/or ncStrs, convert ncStrs to an array of (1) NC
// and convert names to an array of all known NCs for that name.
// Return a list of ncLists.
//
// ex:
//  convert: [ 'billy', 'bob:1' ]
//  to: [ [ billy:1, billy:2 ], [ bob:1 ] ]
//
const nameOrNcStrListToKnownNcLists = (nameOrNcStrList: string[]):
    NameCount.List[] =>
{
    return nameOrNcStrList.map(nameOrNcStr => NameCount.makeNew(nameOrNcStr))
        .map(nc => nc.count ? ncAsKnownNcList(nc) : getKnownNcListForName(nc.name));
};

const combinationNcList =  (indexList: number[], ncLists: NameCount.List[]):
    NameCount.List =>
{
    return indexList.map((ncIndex: number, listIndex: number) =>
        ncLists[listIndex][ncIndex]);
};

const ncListsToCombinations = (ncLists: NameCount.List[]): NameCount.List[] => {
    return Peco.makeNew({
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()])       // keys of array are 0..ncList.length
        //,max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)              // sum of lengths of nclists
    }).getCombinations()
        .map((indexList: number[]) => combinationNcList(indexList, ncLists));
};

const getCombinationNcLists = (useArgsList: string[]): any => {
    Debug(`useArgsList: ${Stringify(useArgsList)}`);
    return useArgsList
        .map(useArg => useArg.split(','))
        .map((nameOrNcStrList: string[]) => nameOrNcStrListToKnownNcLists(nameOrNcStrList))
        .map((knownNcLists: NameCount.List[]) => ncListsToCombinations(knownNcLists));
};

// TODO: combinationNcDataListFromNcLists
const combinationNcDataList = (indexList: number[], ncLists: NameCount.List[]):
    NCDataList =>
{
    return indexList.map((ncIndex: number, listIndex: number) =>
        Object({ ncList: ncLists[listIndex][ncIndex]} ));
};

const combinationsToNcDataLists = (combinationNcLists: NameCount.List[]):
    NCDataList[] =>
{
    Debug(`combToNcDataLists() combinationNcLists: ${Stringify(combinationNcLists)}`);
    return Peco.makeNew({
        // TODO: List.toIndexList()
        listArray: combinationNcLists.map(ncList => [...Array(ncList.length).keys()]), // keys of array are 0..ncList.length-1
        //max: combinationNcLists.reduce((sum, ncList) => sum + ncList.length, 0)        // sum of lengths of nclists
    }).getCombinations().map((ncListIndexes: number[]) =>
        combinationNcDataList(ncListIndexes, combinationNcLists));
};

// it's really questionable whether we want to be summing the NC counts or taking
// the largest individual NC "count" value. haven't fully thought through all the
// use cases
const largestNcDataCountSum = (ncDataList: NCDataList): [number, NameCount.List] => {
    let largest = 0;
    let ncList: NameCount.List = [];
    for (let ncData of ncDataList) {
        const biggest = ncData.ncList.reduce((sum, nc) => sum + nc.count, 0); // biggest sum
        /*
        const biggest = ncData.ncList.reduce((big, nc) => { // biggest individual NC
            if (nc.count > big) big = nc.count;
            return big;
        }, 0);
        */
        if (biggest > largest) {
            largest = biggest;
            ncList = ncData.ncList;
        }
    }
    return [largest, ncList];
}

const buildAllUseNcDataLists = (useArgsList: string[], maxSum: number): NCDataList[] => {
    const combinationNcLists = getCombinationNcLists(useArgsList);
    const ncDataLists = combinationsToNcDataLists(combinationNcLists)
    let begin = new Date();
    let r = ncDataLists.filter((ncDataList: NCDataList) => {
        const [largest, ncList] = largestNcDataCountSum(ncDataList);
        if (largest > maxSum) {
            // NOTE this doesn't display the largest ncList
            //console.error(`skipping ${largest}, ${NameCount.listToString(ncList)}`);
        }
        return largest <= maxSum;
    });
    let d = new Duration(begin, new Date()).milliseconds;
    console.error(` buildAllUseNc max(${maxSum}) - ${PrettyMs(d)}`);

    return r;
};

//////////

const getSourceList = (nc: NameCount.Type, args: any): Source.List => {
    const sourceList: Source.List = [];
    ClueManager.getKnownSourceMapEntries(nc, false, args)
        .forEach((sourceData: ClueManager.SourceData) => {
            sourceList.push(...sourceData.results
                .map((result: ValidateResult) => Source.makeData(nc, result)));
        });
    return sourceList;
};

const addNcListToSourceListMap = (ncList: NameCount.List,
    map: Map<string, Source.AnyData[]>, args: any) : void =>
{
    for (let nc of ncList) {
        const key = NameCount.toString(nc)
        if (!map.has(key)) {
            const sourceList = getSourceList(nc, args);
            if (listIsEmpty(sourceList)) {
                if (!args.ignoreErrors) {
                    throw new Error(`empty sourceList: ${key}`);
                }
                if (!args.quiet) {
                    console.error(`empty sourceList: ${key}`);
                }
            } else {
                map.set(key, sourceList);
            }
        }
    }
};

const fillKnownNcSourceListMapForSum = (map: Map<string, Source.AnyData[]>,
    sum: number, max: number, args: any) : void =>
{
    // Given a sum, such as 4, and a max # of numbers to combine, such as 2, generate
    // an array of addend arrays ("count lists"), for each 2 <= N <= max, that add up
    // to that sum, such as [ [1, 3], [2, 2] ]
    let countListArray: number[][] = Peco.makeNew({ sum, max }).getCombinations();
    // for each countList
    countListArray.forEach((countList: number[]) => {
        let sourceIndices: number[] = [];

        /* compare if output is same for this:
        for (let result = ComboMaker.first(countList, sourceIndices);
             !result.done;
             result = ComboMaker.next(countList, sourceIndices))
        {
            addNcListToSourceListMap(result.ncList!, map, args);
        }
        */
        let result = ComboMaker.first(countList, sourceIndices);
        let firstIter = true;
        while (!result.done) {
            if (firstIter) {
                firstIter = false;
            } else {
                result = ComboMaker.next(countList, sourceIndices);
                if (result.done) break;
            }
            addNcListToSourceListMap(result.ncList!, map, args);
        }
    });
};

const buildKnownNcSourceListMap = (first: number, last: number,
    args: any): Map<string, Source.AnyData[]> =>
{
    let map = new Map<string, Source.AnyData[]>();
    let begin = new Date();
    for (let sum = first; sum <= last; ++sum) {
        let max = Math.min(args.max, sum);
        fillKnownNcSourceListMapForSum(map, sum, max, args);
    }
    let d = new Duration(begin, new Date()).milliseconds;
    console.error(` buildKnownNcSourceListMap(${map.size}) - ${PrettyMs(d)}`);
    return map;
};

// TODO: move to Source.initAllCompatibilityData(sourceList), initCompatibilityData(source)
const setPrimarySrcBits = (sourceList: Source.List): void => {
    for (let source of sourceList) {
        source.usedSources = Source.getUsedSources(source.primaryNameSrcList);
    }
};

const hasCandidate = (nc: NameCount.Type): boolean => { return nc.count >= 1_000_000; }
const listCandidateCount = (ncList: NameCount.List): number => {
    let count = 0;
    for (let nc of ncList) {
        if (hasCandidate(nc)) ++count;
    }
    return count;
}

// TODO: move to Source?
const mergeSources = (source1: Source.Data, source2: Source.Data):
    Source.Data =>
{
    const primaryNameSrcList = [...source1.primaryNameSrcList, ...source2.primaryNameSrcList];
    const usedSources = Source.mergeUsedSources(source1.usedSources, source2.usedSources);
    const ncList = [...source1.ncList, ...source2.ncList];
    return {
        primaryNameSrcList,
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
    sourceListMap: Map<string, Source.AnyData[]>/*, args: MergeArgs*/):
    Source.AnyData[] =>
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
    sourceListMap: Map<string, Source.AnyData[]>/*, args: MergeArgs*/):
    Source.List[] =>
{
    let sourceLists: Source.List[] = [];
    // TODO: This is to prevent duplicate sourceLists. I suppose I could
    //       use a Set or Map, above?
    let hashList: StringBoolMap[] = [];
    for (let ncDataList of useNcDataLists) {
        for (let [sourceListIndex, useNcData] of ncDataList.entries()) {
            if (!sourceLists[sourceListIndex]) sourceLists.push([]);
            if (!hashList[sourceListIndex]) hashList.push({});
            const sourceList = mergeAllCompatibleSources(useNcData.ncList, sourceListMap) as Source.List;
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
};

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
const markAllXorCompatibleOrSources = (xorSourceList: Source.List,
    orArgDataList: OrArgDataList): void =>
{
    for (let orArgData of orArgDataList) {
        for (let orSource of orArgData.orSourceList) {
            if (Source.isXorCompatibleWithAnySource(orSource.source, xorSourceList)) {
                orSource.xorCompatible = true;
            }
        }
    }
};

const dumpNcDataLists = (ncDataLists: NCDataList[],
    sourceListMap: Map<string, Source.AnyData[]>): void =>
{
    console.error(` ncDataLists:`);
    for (let ncDataList of ncDataLists) {
        for (let ncData of ncDataList) {
            let str = NameCount.listToString(ncData.ncList);
            console.error(`  ${str}: ${sourceListMap.has(str)}`);
        }
    }
}

const buildUseSourceListsFromNcData = (sourceListMap: Map<string,
    Source.AnyData[]>, args: any): Source.List =>
{
    // XOR first
    const xorSources: Source.List = NativeComboMaker.mergeCompatibleXorSourceCombinations(
        args.allXorNcDataLists, Array.from(sourceListMap.entries()),
        args.merge_only || false);
    if (listIsEmpty(xorSources) || args.merge_only) {
        return xorSources;
    }
    //**
    // Everything below is only for filter use case (-c and not -t).
    //**
    args.allXorNcDataLists = undefined;
    // OR next
    let or0 = new Date();
    let orArgDataList = buildOrArgDataList(buildSourceListsForUseNcData(
        args.allOrNcDataLists, sourceListMap));
    let or_dur = new Duration(or0, new Date()).milliseconds;
    console.error(` orArgDataList(${orArgDataList.length}) - ${PrettyMs(or_dur)}`);

    // Thoughts on AND compatibility of OrSources:
    // Just because (one sourceList of) an OrSource is AND compatible with an
    // XorSource doesn't mean the OrSource is redundant and can be ignored
    // (i.e., the container cannot be marked as "compatible.") We still need
    // to check the possibility that any of the other XOR-but-not-AND-compatible
    // sourceLists could be AND-compatible with the generated-combo sourceList.
    // So, a container can be marked compatible if and only if there are no
    // no remaining XOR-compatible sourceLists.
    //TODO: markAllANDCompatibleOrSources(xorSourceList, orSourceList);

    /* TODO: C++
    let mark0 = new Date();
    markAllXorCompatibleOrSources(xorSourceList, orArgDataList);
    let mark_dur = new Duration(mark0, new Date()).milliseconds;
    console.error(` mark - ${PrettyMs(mark_dur)}`);
    */
    NativeComboMaker.filterPreparation(orArgDataList);
    return xorSources;
};

export const preCompute = (first: number, last: number, args: any): Result => {
    const maxSum = args.max_sources - 1;

    const begin = new Date();
    args.allXorNcDataLists = args.xor ? buildAllUseNcDataLists(args.xor, maxSum) : [ [] ];
    const d1 = new Duration(begin, new Date()).milliseconds;
    if (args.xor && listIsEmpty(args.allXorNcDataLists)) return { success: false };

    const build2 = new Date();
    args.allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or, maxSum) : [ [] ];
    const d2 = new Duration(build2, new Date()).milliseconds;
    console.error(` buildAllOrNcDataLists(${args.allOrNcDataLists.length})` +
        ` - ${PrettyMs(d2)}`);
    if (args.or && listIsEmpty(args.allOrNcDataLists)) return { success: false };
    
    // TODO: move to C++.
    let sourceListMap = buildKnownNcSourceListMap(2, args.max_sources, args);

    const build3 = new Date();
    const xorSources = buildUseSourceListsFromNcData(sourceListMap, args);
    const d3 = new Duration(build3, new Date()).milliseconds;
    // TODO: sizes?
    if (args.xor && listIsEmpty(xorSources)) return { success: false };

    const d = new Duration(begin, new Date()).milliseconds;
    console.error(`--Precompute - ${PrettyMs(d)}`);
    return { success: true, data: { xor: xorSources, sourceListMap } };
};
