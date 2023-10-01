//
// cm-precompute.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Native      = require('../../../build/experiment.node');
const Peco        = require('../../modules/peco');

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

export interface Data {
    xor: Source.List|number;
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

//////////

const ncAsKnownNcList = (nc: NameCount.Type): NameCount.List => {
    return ClueManager.isKnownNc(nc) ? [nc] : [];
};

const getKnownNcListForName = (name: string): NameCount.List => {
    const countList = ClueManager.getCountListForName(name);
    if (_.isEmpty(countList)) {
        console.error(`not a valid clue name: '${name}'`);
        process.exit(-1);
    }
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
        listArray: ncLists.map(ncList => [...Array(ncList.length).keys()])       // keys of array are 0..ncList.length-1
        //,max: ncLists.reduce((sum, ncList) => sum + ncList.length, 0)          // sum of lengths of nclists
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
    const begin = new Date();
    const result = ncDataLists.filter((ncDataList: NCDataList) => {
        const [largest, ncList] = largestNcDataCountSum(ncDataList);
        if (largest > maxSum) {
            // NOTE this doesn't display the largest ncList
            //console.error(`skipping ${largest}, ${NameCount.listToString(ncList)}`);
        }
        return largest <= maxSum;
    });
    let d = new Duration(begin, new Date()).milliseconds;
    console.error(` buildAllUseNc max(${maxSum}) - ${PrettyMs(d)}`);
    return result;
};

const hasCandidate = (nc: NameCount.Type): boolean => { return nc.count >= 1_000_000; }
const listCandidateCount = (ncList: NameCount.List): number => {
    let count = 0;
    for (let nc of ncList) {
        if (hasCandidate(nc)) ++count;
    }
    return count;
}

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

export const preCompute = (first: number, last: number, args: any): Result => {
    const maxSum = args.max_sources - 1;
    const merge_only = args.merge_only || false;

    const begin = new Date();
    // XOR first
    const xorNcDataLists = args.xor ? buildAllUseNcDataLists(args.xor, maxSum) : [ [] ];
    if (args.xor && listIsEmpty(xorNcDataLists)) return { success: false };

    const xorSourceListOrNumIndices: Source.List|number = 
        Native.mergeCompatibleXorSourceCombinations(xorNcDataLists, merge_only);
    if (merge_only) {
        Assert(typeof(xorSourceListOrNumIndices) !== "number");
        const xorSourceList = xorSourceListOrNumIndices as Source.List;
        return args.xor && listIsEmpty(xorSourceList) ? { success: false } :
            { success: true, data: { xor: xorSourceList } };
    }
    if (args.xor && (xorSourceListOrNumIndices === 0)) return { success: false };

    // OR next
    const orNcDataLists = args.or ? buildAllUseNcDataLists(args.or, maxSum) : [ [] ];
    if (args.or && listIsEmpty(orNcDataLists)) return { success: false };

    Native.setOrArgs(orNcDataLists);
    Native.filterPreparation();

    const d = new Duration(begin, new Date()).milliseconds;
    console.error(`--Precompute - ${PrettyMs(d)}`);
    return { success: true, data: { xor: xorSourceListOrNumIndices } };
};
