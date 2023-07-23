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
const NativeComboMaker = require('../../../build/Release/experiment.node');
//const NativeComboMaker = require('../../../build/Debug/experiment.node');

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

interface PerfData {
    calls: number;
    comps: number;
    compat: number;
    range_calls: number;
    ss_attempt: number;
    ss_fail: number;
    full: number;
}

interface CandidateStats {
    sum: number;
    sourceLists: number;
    totalSources: number;
    comboMapIndices: number;
    totalCombos: number;
}

interface StringAnyMap {
    [key: string]: any;
}

type CountArray = Int32Array;

interface CountArrayAndSize {
    array: CountArray;
    size: number;
}

interface OptionalCloneOnMerge {
    cloneOnMerge?: boolean;
}

type MergedSources = Source.ListContainer & Source.CompatibilityData & OptionalCloneOnMerge;
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

/*
const listGetNumEmptySublists = (listOfLists: any[][]) => {
    let numEmpty = 0;
    for (let list of listOfLists) {
        if (listIsEmpty(list)) ++numEmpty;
    }
    return numEmpty;
};
*/

let ms_copy = 0;
let ms_inplace = 0;
let ms_comp = 0;
let ms_compat = 0;

const mergeSourceInPlace = (mergedSources: MergedSources,
    source: Source.Data): MergedSources =>
{
    ++ms_inplace;
    CountBits.orInPlace(mergedSources.sourceBits, source.sourceBits);
    Source.mergeUsedSourcesInPlace(mergedSources.usedSources, source.usedSources);
    mergedSources.sourceList.push(source);
    return mergedSources;
}

const mergeSource = (mergedSources: MergedSources, source: Source.Data):
    MergedSources =>
{
    ++ms_copy;
    return {
        //CountBits.or(mergedSources.sourceBits, source.sourceBits),
        sourceBits: mergedSources.sourceBits.clone().union(source.sourceBits),
        usedSources: Source.mergeUsedSources(mergedSources.usedSources, source.usedSources),
        sourceList: [...mergedSources.sourceList, source]
    };
};

const makeMergedSourcesList = (sourceList: Source.List) : MergedSourcesList => {
    let result: MergedSourcesList = [];
    for (const source of sourceList) {
        result.push({
            // CountBits.makeFrom(source.sourceBits),
            sourceBits: source.sourceBits, // .clone(),
            usedSources: source.usedSources, // Source.cloneUsedSources(source.usedSources),
            sourceList: [source],
            cloneOnMerge: true
        });
    }
    return result;
};

interface MergeData {
    mergedSources: MergedSources;
    sourceList: Source.List;
}

const getCompatibleSourcesMergeData = (mergedSourcesList: MergedSourcesList,
    sourceList: Source.List): MergeData[] =>
{
    let result: MergeData[] = [];
    for (let mergedSources of mergedSourcesList) {
        let mergeData: MergeData = { mergedSources, sourceList: [] };
        for (const source of sourceList) {
            ++ms_comp;
            if (Source.isXorCompatible(mergedSources, source)) {
                ++ms_compat;
                mergeData.sourceList.push(source);
            }
        }
        if (!listIsEmpty(mergeData.sourceList)) {
            result.push(mergeData);
        }
    }
    return result;
};

//let ms_111 = 0;

const mergeSourcesInMergeData = (mergedSourcesList: MergedSourcesList,
    mergeData: MergeData[], flag?: number): MergedSourcesList =>
{
    /* does nothing.
    if ((mergeData.length === 1) && (mergeData[0].sourceList.length === 1)) {
        ++ms_111;
        const data = mergeData[0];
        if (!data.mergedSources.cloneOnMerge) {
            const mergedSources = mergeSourceInPlace(data.mergedSources, data.sourceList[0]);
            return (mergedSourcesList.length === 1) ? mergedSourcesList : [mergedSources];
        } else {
            return [mergeSource(data.mergedSources, data.sourceList[0])];
        }
    }
    */
    let result: MergedSourcesList = [];
    // TODO: since indexing isn't used below, i could use for..of
    for (let i = 0; i < mergeData.length; ++i) {
        let data = mergeData[i];
        for (let j = 0; j < data.sourceList.length; j++) {
            const source = data.sourceList[j];
            /* does nothing
            // if this is the last source to be merged with this mergedSources
            if ((j === data.sourceList.length - 1) && !data.mergedSources.cloneOnMerge) {
                // merge in place
                result.push(mergeSourceInPlace(data.mergedSources, source));
            } else {
            */
            // else copy/merge
            result.push(mergeSource(data.mergedSources, source));
            //}
        }
    }
    return result;
}

const mergeAllCompatibleSources3 = (ncList: NameCount.List,
    sourceListMap: Map<string, Source.AnyData[]>, flag?: number): MergedSourcesList =>
{
    // because **maybe** broken for > 2
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    let mergedSourcesList: MergedSourcesList = [];
    for (let nc of ncList) {
        const sources = sourceListMap.get(NameCount.toString(nc)) as Source.List;
        if (listIsEmpty(mergedSourcesList)) {
            mergedSourcesList = makeMergedSourcesList(sources);
            continue;
        }
        const mergeData = getCompatibleSourcesMergeData(mergedSourcesList, sources);
        if (listIsEmpty(mergeData)) {
            return [];
        }
        mergedSourcesList = mergeSourcesInMergeData(mergedSourcesList, mergeData, flag);
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

// "clueIndex" here is actually "uniqueNameIndex". except if it's a legacy
// clue index, which are (non-uniquely) added to the uniqueNameList first.
// This is done to support ignore/skip properties, which only exist on legacy
// clues, but the same properties don't exist on every instance of the same
// named legacy clue. We needed to differentiate between the various isntances
// of:  ace, hero, north
const skip = (clueCount: number, clueIndex: number): boolean => {
    if (clueCount !== 1) return false;
    const clueList = ClueManager.getClueList(clueCount)
    // only legacy clue indices are currently skippable
    if (clueIndex >= clueList.length) return false;
    const clue = clueList[clueIndex];
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
        // TODO: what if I could delay sorting until compatibility was established
        //nameList.sort();
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
            let mergedSourcesList: MergedSourcesList = [];
            if (!hash[key]) {
                mergedSourcesList = mergeAllCompatibleSources3(result.ncList!, pcd.sourceListMap);
                if (listIsEmpty(mergedSourcesList)) {
                    ++numMergeIncompatible;
                }
                hash[key] = { mergedSourcesList };
            } else {
                mergedSourcesList = hash[key].mergedSourcesList;
                numCacheHits += 1;
            }

            // failed to find any compatible combos
            if (listIsEmpty(mergedSourcesList)) continue;
            
            const combo = result.nameList!.sort().toString();
            const listOrIndex = (hash[key].index === undefined) ?
                mergedSourcesList : hash[key].index;
            hash[key].index = NativeComboMaker.addCandidateForSum(sum, combo, listOrIndex);
        }
        totalVariations += numVariations;
    });

    let duration = (process.hrtime.bigint() - start) / MILLY;
    Debug(`sum(${sum}) combos(${comboCount}) variations(${totalVariations})` +
        ` cacheHits(${numCacheHits}) merge-incompatible(${numMergeIncompatible})` +
        ` use-incompatible(${numUseIncompatible})` +
        ` actual(${totalVariations - numCacheHits - numUseIncompatible}) ${duration}ms`);

    if (args.verbose) {
        console.error(`sum(${sum}) combos(${comboCount})` +
            ` variations(${totalVariations}) cacheHits(${numCacheHits})` +
            ` no-merge(${numMergeIncompatible}) no-use(${numUseIncompatible})` +
            ` actual(${totalVariations - numCacheHits - numUseIncompatible})` +
            ` - ${duration}ms `);
        const cs = NativeComboMaker.getCandidateStatsForSum(sum);
        console.error(`  sourceLists(${cs.sourceLists})` +
            `, totalSources(${cs.totalSources})` +
            `, comboMapIndices(${cs.comboMapIndices})` +
            `, totalCombos(${cs.totalCombos})`);
        
    } else {
        process.stderr.write('.');
    }
    NativeComboMaker.filterCandidatesForSum(sum, args.streams, args.workitems);
    return combos;
};

export const makeCombosForSum = (sum: number, max: number, args: any): string[] => {
    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
        // TODO: whereever this is actually enforced:
        // console.error(`Enforcing max results: ${args.maxResults}`);
    }
    // TODO move this a layer or two out; use "validateArgs" 
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

const makeNameVariationLists = (nameList: string[],
    variations: Sentence.Variations): string[][] =>
{
    return nameList.map(name =>
        [name, ...Sentence.getNameVariations(name, variations)]);
}

// combos: an iterable of nameCsvs
const displayCombos = (combos: any, variations: Sentence.Variations): void => {
    let hash = new Set<string>();
    for (let nameCsv of combos) {
        Peco.makeNew({
            listArray: makeNameVariationLists(nameCsv.split(','), variations)
        }).getCombinations().forEach(nameList => {
            const nameCsv = nameList.sort().toString();
            if (!hash.has(nameCsv)) {
                hash.add(nameCsv);
                console.log(nameCsv);
            }
        });
    }
}

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
                // TODO: no filtering here?
                displayCombos(comboSet.values(), ClueManager.getVariations())
            });
    } else {
        let totals = ClueManager.emptyFilterResult();
        const result = PreCompute.preCompute(first, last, args);
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
                const filterResult = ClueManager.filter(comboList, sum, totals);
            }
        }
        let d = new Duration(begin, new Date()).milliseconds;
        if (!args.verbose) console.error('');
        console.error(`--combos, total(${total}), known(${totals.known})` +
            `, reject(${totals.reject}), dupes(${totals.duplicate})` +
            ` - ${PrettyMs(d)}`);
        if (1) {
            console.error(`merge: copy(${ms_copy}), inplace(${ms_inplace})` +
                `, comp(${ms_comp}), compat(${ms_compat})`);
                //, ms_111(${ms_111})
            /*
            const isany: PerfData = NativeComboMaker.getIsAnyPerfData();
            console.error(`isAny: calls(${isany.calls})` +
                `, range_calls(${isany.range_calls}), full_range(${isany.full})` +
                `, comps(${isany.comps}), compat(${isany.compat})` +
                `, ss_attempt(${isany.ss_attempt}), ss_fail(${isany.ss_fail})`);
            */
        }
        Debug(`total: ${total}, filtered(${_.size(totals.map)})`);
        displayCombos(Object.keys(totals.map), ClueManager.getVariations());
    }
    return 1;
};
