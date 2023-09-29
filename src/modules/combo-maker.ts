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
const NativeComboMaker = require('../../../build/experiment.node');
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

//import { ValidateResult } from './validator';

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

interface OptionalCloneOnMerge {
    cloneOnMerge?: boolean;
}

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

const nextIndex = (countList: number[], clueIndices: number[]): boolean => {
    // increment last index
    let index = clueIndices.length - 1;
    clueIndices[index] += 1;
    // while last index is maxed: reset to zero, increment next-to-last index, etc.
    while (clueIndices[index] === ClueManager.getUniqueClueNameCount(countList[index])) {
        clueIndices[index] = 0;
        if (--index < 0) {
            return false;
        }
        clueIndices[index] += 1;
    }
    return true;
};

// "clueIndex" here is actually "uniqueNameIndex". except if it's a legacy
// clue index, which are (non-uniquely) added to the uniqueNameList first.
// This is done to support ignore/skip properties, which only exist on legacy
// clues, but the same properties don't exist on every instance of the same
// named legacy clue. We needed to differentiate between the various instances
// of:  ace, hero, north
const skip = (clueCount: number, clueIndex: number): boolean => {
    Assert(0, "ClueManager.getClueList(1) is bogus");
    if (clueCount !== 1) return false;
    const clueList = ClueManager.getClueList(clueCount)
    // only legacy clue indices are currently skippable
    if (clueIndex >= clueList.length) return false;
    const clue = clueList[clueIndex];
    return Boolean(clue.ignore || clue.skip);
}

export interface FirstNextResult {
    done: boolean;
    ncList?: NameCount.List;
}

export const next = (countList: number[], clueIndices: number[]): FirstNextResult => {
    for (;;) {
        if (!nextIndex(countList, clueIndices)) {
            return { done: true };
        }
        let ncList: NameCount.List = [];    // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
        if (countList.every((count, index) => {
            //if (skip(count, clueIndices[index])) return false;
            let name = ClueManager.getUniqueClueName(count, clueIndices[index]);
            if (ncList.length) {
                // because we are only comparing to ncList[0].name
                Assert((ncList.length < 2) && "logic broken");
                // no duplicate names allowed
                if (ncList[0].name === name) return false;
            }
            // TODO: ncList.push({ name, count });
            ncList.push(NameCount.makeNew(name, count));
            return true; // every.continue;
        })) {
            NameCount.sortList(ncList);
            return { done: false, ncList };
        }
    }
};

export const first = (countList: number[], clueIndices: number[]): FirstNextResult => {
    // TODO: _.fill?
    for (let index = 0; index < countList.length; ++index) {
        if (ClueManager.getUniqueClueNameCount(countList[index]) === 0) {
            return { done: true };
        }
        clueIndices[index] = 0;
    }
    clueIndices[clueIndices.length - 1] = -1;
    return next(countList, clueIndices);
};

//
// args:
//   synonymMinMax
//
const getCombosForUseNcLists = (sum: number, max: number, args: any): void => {
    let comboCount = 0;
    let totalVariations = 0;
    
    const MILLY = 1000000n;
    const start = process.hrtime.bigint();

    // Given a sum, such as 4, and a max # of numbers to combine, such as 2, generate
    // an array of addend arrays ("count lists"), for each 2 <= N <= max, that add up
    // to that sum, such as [ [1, 3], [2, 2] ]
    let countListArray: number[][] = Peco.makeNew({ sum, max }).getCombinations(); 

    let candidateCount = 0;
    // for each countList
    countListArray.forEach((countList: number[]) => {
        comboCount += 1;

        let clueIndices: number[] = [];
        let result = first(countList, clueIndices);
        if (result.done) return; // continue; 

        let numVariations = 1;

        // this is effectively Peco.getCombinations().forEach()
        let firstIter = true;
        while (!result.done) {
            if (!firstIter) {
                result = next(countList, clueIndices);
                if (result.done) break;
                numVariations += 1;
            } else {
                firstIter = false;
            }
           NativeComboMaker.considerCandidate(result.ncList!, sum);
        }
        totalVariations += numVariations;
    });

    let duration = (process.hrtime.bigint() - start) / MILLY;
    Debug(`sum(${sum}) combos(${comboCount}) variations(${totalVariations})` +
        ` -${duration}ms`);

    if (args.verbose) {
        console.error(`sum(${sum}) combos(${comboCount})` +
            ` variations(${totalVariations}) - ${duration}ms `);
        /*
        const cs = NativeComboMaker.getCandidateStatsForSum(sum);
        console.error(`  sourceLists(${cs.sourceLists})` +
            `, totalSources(${cs.totalSources})` +
            `, comboMapIndices(${cs.comboMapIndices})` +
            `, totalCombos(${cs.totalCombos})`);
        */
    }
    NativeComboMaker.filterCandidatesForSum(sum, args.tpb, args.streams,
      args.stride, args.iters, args.synchronous);
};

export const makeCombosForSum = (sum: number, args: any,
  synchronous: boolean = false): void =>
{
    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
        // TODO: whereever this is actually enforced:
        // console.error(`Enforcing max results: ${args.maxResults}`);
    }
    args.synchronous = synchronous;
    // TODO: Fix this abomination
    args.sum = sum;
    let max = args.max;
    args.max = Math.min(args.max, args.sum);
    getCombosForUseNcLists(sum, max, args);
    args.max = max;
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
            //xormm: args.xormm,
            or: args.or,
            fast: args.fast,
            load_max: ClueManager.getMaxClues(),
            parallel: true
            //,puse_syns: args.use_syns
            //,synonymMinMax: args.synonymMinMax
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
            // seed "incompatible sources" with 2-clue sources to make all
            // subsequent sums faster. but, synchronously.
            makeCombosForSum(2, args, true);
            if (first == 2) ++first;
            for (let sum = first; sum <= last; ++sum) {
                // TODO: return # of combos filtered due to note name match
                makeCombosForSum(sum, args);
            }
            const comboList = NativeComboMaker.getResult();
            total += comboList.length;
            ClueManager.filter(comboList, 0, totals);
        }
        let d = new Duration(begin, new Date()).milliseconds;
        console.error(`--combos, total(${total}), known(${totals.known})` +
            `, reject(${totals.reject}), dupes(${totals.duplicate})` +
            ` - ${PrettyMs(d)}`);
        if (1) {
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
