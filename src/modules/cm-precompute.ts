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
const JStringify   = require('javascript-stringify').stringify;

import * as Clue from '../types/clue';
import * as ClueManager from './clue-manager';
import * as ComboMaker from './combo-maker';
import * as CountBits from '../types/count-bits-fastbitset';
import * as MinMax from '../types/min-max';
import * as NameCount from '../types/name-count';
import * as Sentence from '../types/sentence';
import * as Source from './source';

export interface Data {
    xor: Source.List|number;
}

/*
export interface Result {
    success: boolean;
    data?: Data;
}
*/

interface NCData {
    ncList: NameCount.List;
    name: string;
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

const combinationNcList = (indexList: number[], ncLists: NameCount.List[]):
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
const combinationNcDataList = (ncIndexList: number[], ncLists: NameCount.List[]):
    NCDataList =>
{
    return ncIndexList.map((ncIndex: number, listIndex: number) =>
        Object({ ncList: ncLists[listIndex][ncIndex]} ));
};

const combinationsToNcDataLists = (combinationNcLists: NameCount.List[], verbose: number):
    NCDataList[] =>
{
    Debug(`combToNcDataLists() combinationNcLists: ${Stringify(combinationNcLists)}`);
    // TODO: NameCount.listsToIndexLists()
    // keys of array are 0..ncList.length-1
    const listArray = combinationNcLists.map(ncList => [...Array(ncList.length).keys()]);
    if (verbose > 1) console.error(` listArray: ${JStringify(listArray)}`);
    let indexLists =  Peco.makeNew({ listArray }).getCombinations();
    if (verbose > 1) console.error(` indexLists: ${JStringify(indexLists)}`);
    return indexLists.map((ncIndexList: number[]) =>
        combinationNcDataList(ncIndexList, combinationNcLists));
};

// TODO:
// it's an open question whether we want to be summing the NC counts or taking
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
};

const displayCombinationNcLists = (combinationNcLists: NameCount.List[][]): void => {
    console.error(' [');
    for (let ncLists of combinationNcLists) {
        let str = '';
        let first = true;
        for (let ncList of ncLists) {
            if (!first) str += ',';
            str += `[${NameCount.listToString(ncList)}]`;
            first = false;
        }
        console.error(`  [${str}]`);
    }
    console.error(' ]');
};

const buildAllUseNcDataLists = (listName: string, maxSum: number, args: any): NCDataList[] => {
    const useArgsList: string[] = args[listName];
    Assert(useArgsList);
    if (args.verbose) {
        console.error(`buildAllUseNcDataLists(${listName}), useArgList(${useArgsList.length})` +
            `, maxSum(${maxSum})`);
    }
    const combinationNcLists = getCombinationNcLists(useArgsList);
    if (args.verbose > 1) {
        console.error(` combinationNcLists(${combinationNcLists.length}):`);
        displayCombinationNcLists(combinationNcLists);
    }
    const ncDataLists = combinationsToNcDataLists(combinationNcLists, args.verbose);
    if (args.verbose > 1) {
        console.error(` ncDataLists(${ncDataLists.length})`);
        console.error(`  [0]: ${JStringify(ncDataLists[0])}`);
    }
    const begin = new Date();
    const result = ncDataLists.filter((ncDataList: NCDataList) => {
        const [largest, ncList] = largestNcDataCountSum(ncDataList);
        if (largest > maxSum) {
            // NOTE this doesn't display the largest ncList ??
            //console.error(`skipping ${largest}, ${NameCount.listToString(ncList)}`);
        }
        return largest <= maxSum;
    });
    if (args.verbose > 1) {
        let d = new Duration(begin, new Date()).milliseconds;
        console.error(` filtered ncDataLists(${result.length}) - ${PrettyMs(d)}`);
    }
    return result;
};

const hasCandidate = (nc: NameCount.Type): boolean => { return nc.count >= 1_000_000; }
const listCandidateCount = (ncList: NameCount.List): number => {
    let count = 0;
    for (let nc of ncList) {
        if (hasCandidate(nc)) ++count;
    }
    return count;
};

export const preCompute = (first: number, last: number, args: any): boolean => {
    const maxSum = args.max_sources;// - 1;
    const merge_only = args.merge_only || false;

    const begin = new Date();
    // XOR first
    const xorNcDataLists = args.xor ? buildAllUseNcDataLists("xor", maxSum, args) : [ [] ];
    if (listIsEmpty(xorNcDataLists)) {
       console.error(`empty xorNcDataLists`);
       return false;
    }

    const num_indices: number = 
        Native.mergeCompatibleXorSourceCombinations(xorNcDataLists, merge_only);
    if (args.xor && !num_indices) {
        console.error(`no compatible XOR sources`);
        return false;
    }
    console.error(`compatible XOR sources: ${num_indices}`);
    if (merge_only) return true;

    // OR next
    const orNcDataLists = args.or ? buildAllUseNcDataLists("or", maxSum, args) : [ [] ];
    if (listIsEmpty(orNcDataLists)) return false;

    // TODO: call mergeCompatible here for OR args same as above?

    if (!Native.filterPreparation(orNcDataLists)) {
        console.error(`no compatible OR sources`);
        return false;
    }

    const d = new Duration(begin, new Date()).milliseconds;
    console.error(`--Precompute - ${PrettyMs(d)}`);
    return true;
};
