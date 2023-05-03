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
import * as PreCompute from './cm-precompute';
import * as Sentence from '../types/sentence';
import * as Source from './source';
import * as Synonym from './synonym';

import { ValidateResult } from './validator';

// TODO: import from somewhere. also defined in clue-manager
const DATA_DIR =  Path.normalize(`${Path.dirname(module.filename)}/../../../data/`);

interface StringAnyMap {
    [key: string]: any;
}

type CountArray = Int32Array;

interface CountArrayAndSize {
    array: CountArray;
    size: number;
}

type MergedSources = Source.ListContainer & Source.CompatibilityData;
type MergedSourcesList = MergedSources[];

let PCD: PreCompute.Data;

//////////

const Stringify = (val: any) => {
    return stringify(val, (value: any, indent: any, stringify: any) => {
        if (typeof value === 'function') return "function";
        return stringify(value);
    }, " ");
}

// TODO: as const;
const Op = {
    and: 1,
    or:  2,
    xor: 3
};
Object.freeze(Op);

const OpName = (opValue: number): string | undefined => {
    return _.findKey(Op, (v: number) => opValue === v);
};

//////////

const listIsEmpty = (list: any[]): boolean => {
    return list.length === 0;
};

const listGetNumEmptySublists = (listOfLists: any[][]) => {
    let numEmpty = 0;
    for (let list of listOfLists) {
        if (listIsEmpty(list)) ++numEmpty;
    }
    return numEmpty;
};

const mergeCompatibleSourceLists2 = (mergedSourcesList: MergedSourcesList,
    sourceList: Source.List): MergedSourcesList =>
{
    let result: MergedSourcesList = [];
    for (const mergedSources of mergedSourcesList) {
        for (const source of sourceList) {
	    if (!Source.isXorCompatible(source, mergedSources)) continue;
	    result.push({
		sourceBits: CountBits.or(mergedSources.sourceBits, source.sourceBits),
		usedSources: Source.mergeUsedSources(mergedSources.usedSources, source.usedSources),
		sourceList: [...mergedSources.sourceList, source]
	    });
        }
    }
    return result;
};

const makeMergedSourcesList = (sourceList: Source.List) : MergedSourcesList => {
    let mergedSourcesList: MergedSourcesList = [];
    for (const source of sourceList) {
	mergedSourcesList.push({
	    sourceBits: CountBits.makeFrom(source.sourceBits),
	    usedSources: source.usedSources.slice(),
	    sourceList: [source]
	});
    }
    return mergedSourcesList;
};

const mergeAllCompatibleSources2 = (ncList: NameCount.List,
    sourceListMap: Map<string, Source.AnyData[]>): MergedSourcesList =>
{
    // because **maybe** broken for > 2 below
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    // TODO: reduce (or some) here
    let mergedSourcesList = makeMergedSourcesList(sourceListMap.get(
	NameCount.toString(ncList[0])) as Source.List);
    for (let ncIndex = 1; ncIndex < ncList.length; ++ncIndex) {
        const nextSourceList = sourceListMap.get(NameCount.toString(ncList[ncIndex])) as Source.List;
        mergedSourcesList = mergeCompatibleSourceLists2(mergedSourcesList, nextSourceList);
        // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
        if (listIsEmpty(mergedSourcesList)) break;
    }
    return mergedSourcesList;
};

const nextIndex = (countList: number[], clueIndexes: number[]): boolean => {
    // increment last index
    let index = clueIndexes.length - 1;
    clueIndexes[index] += 1;

    // while last index is maxed: reset to zero, increment next-to-last index, etc.
    while (clueIndexes[index] === ClueManager.getUniqueClueNameCount(countList[index])) {
        clueIndexes[index] = 0;
        if (--index < 0) {
            return false;
        }
	clueIndexes[index] += 1;
    }
    return true;
};

export interface FirstNextResult {
    done: boolean;
    ncList?: NameCount.List;
    nameList?: string[];
}

const skip = (clueCount: number, clueIndex: number): boolean => {
    if (clueCount < 2) return false;
    const clue = ClueManager.getClueList(clueCount)[clueIndex];
    return Boolean(clue.ignore || clue.skip);
}

export const next = (countList: number[], clueIndexes: number[]): FirstNextResult => {
    for (;;) {
        if (!nextIndex(countList, clueIndexes)) {
            return { done: true };
        }
        let ncList: NameCount.List = [];    // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
        let nameList: string[] = [];        // e.g. [ "pollock", "jackson" ]
        if (!countList.every((count, index) => {
	    if (skip(count, clueIndexes[index])) return false;
            let name = ClueManager.getUniqueClueName(count, clueIndexes[index]);
            nameList.push(name);
            ncList.push(NameCount.makeNew(name, count));
            return true; // every.continue;
        })) {
            continue;
        }
        nameList.sort();
        NameCount.sortList(ncList);
        return { done: false, ncList, nameList };
    }
};

export const first = (countList: number[], clueIndexes: number[]): FirstNextResult => {
    // TODO: _.fill?
    for (let index = 0; index < countList.length; ++index) {
	if (ClueManager.getUniqueClueNameCount(countList[index]) === 0) {
	    return { done: true };
	}
        clueIndexes[index] = 0;
    }
    clueIndexes[clueIndexes.length - 1] = -1;
    return next(countList, clueIndexes);
};

//
// args:
//   synonymMinMax
//
const getCombosForUseNcLists = (sum: number, max: number, pcd: PreCompute.Data,
    args: any): string[] =>
{
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

    // Given a sum, such as 4, and a max # of numbers to combine, such as 2, generate
    // an array of addend arrays ("count lists"), for each 2 <= N <= max, that add up
    // to that sum, such as [ [1, 3], [2, 2] ]
    let countListArray: number[][] = Peco.makeNew({ sum, max }).getCombinations(); 

    // for each countList
    countListArray.forEach((countList: number[]) => {
        comboCount += 1;

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
            const key: string = NameCount.listToString(result.ncList!);

            let cacheHit = false;
	    let mergedSourcesList: MergedSourcesList = [];
            if (!hash[key]) {
		mergedSourcesList = mergeAllCompatibleSources2(result.ncList!, pcd.sourceListMap);
                if (listIsEmpty(mergedSourcesList)) {
		    ++numMergeIncompatible;
		}
                hash[key] = { mergedSourcesList };
            } else {
                mergedSourcesList = hash[key].mergedSourcesList;
                cacheHit = true;
                numCacheHits += 1;
            }

            //console.log(`  found compatible sources: ${!listIsEmpty(mergedSourcesList)}`);

            // failed to find any compatible combos
            if (listIsEmpty(mergedSourcesList)) continue;

            if (hash[key].isCompatible === undefined) {
		isany += 1;
		let flag = false;
                hash[key].isCompatible = NativeComboMaker.isAnySourceCompatibleWithUseSources(mergedSourcesList, flag);
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

    if (args.verbose) {
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
export const makeCombosForSum = (sum: number, max: number, args: any): string[] => {
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

const parallel_makeCombosForRange = (first: number, last: number, args: any): any => {
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

export const makeCombos = (args: any): any => {
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
        const result = PreCompute.preCompute(first, last, args);
        let comboMap = {};
	if (result.success) {
	    PCD = result.data!;
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
	}
        let d = new Duration(begin, new Date()).milliseconds;
	if (!args.verbose) console.error('');
        console.error(`--combos: ${PrettyMs(d)}`);
        Debug(`total: ${total}, filtered(${_.size(comboMap)})`);
        _.keys(comboMap).forEach((nameCsv: string) => console.log(nameCsv));
    }
    return 1;
};
