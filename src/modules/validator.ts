//
// validator.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Peco        = require('../../modules/peco');
const ResultMap   = require('../../types/result-map');

const Assert      = require('assert');
const Debug       = require('debug')('validator');
const Expect      = require('should/as-function');
const Native      = require('../../../build/experiment.node');
const stringify   = require('javascript-stringify').stringify;
const Stringify2  = require('stringify-object');
const Timing      = require('debug')('timing');

import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import * as ClueManager from './clue-manager';
import * as CountBits from '../types/count-bits-fastbitset';
import * as NameCount from '../types/name-count';
import * as OldValidator from './old-validator';
import * as Sentence from '../types/sentence';
import * as Source from './source';

let logLevel         = 0;

function Stringify(val: any) {
    return stringify(val, (value: any, indent: any, stringify: any) => {
        if (typeof value === 'function') return "function";
        return stringify(value);
    }, " ");
}

interface ValidateResultData {
    ncList: NameCount.List;
    nameSrcList: NameCount.List;
    resultMap?: any;
    nameSrcCsv?: string; // TODO: remove; old-validator uses it, stop using old-validator
}

export type ValidateResult = ValidateResultData & Source.CompatibilityData;

export interface ValidateSourcesResult {
    success: boolean;
    list?: ValidateResult[];
}

let spaces = (length: number): string => {
    return ' '.repeat(length);
};

let indent = (): string => {
    return spaces(logLevel);
};

let indentNewline = (): string => {
    return '\n' + indent();
};

const emptyValidateResult = (): ValidateResult => {
    return {
        ncList: [],
        nameSrcList: [],
        //resultMap: ResultMap.makeNew(),
        usedSources: Source.emptyUsedSources()
    };
};

let copyAddNcList = (ncList: NameCount.List, name: string, count: number): NameCount.List => {
    // for non-primary check for duplicate name:count entry
    // technically this is allowable for count > 1 if the there are
    // multiple entries of this clue name in the clueList[count].
    // (at least as many entries as there are copies of name in ncList)
    // TODO: make knownSourceMapArray store a count instead of boolean

    if (!ncList.every(nc => {
        if (nc.count > 1) {
            if ((name === nc.name) && (count === nc.count)) {
                return false;
            }
        }
        return true;
    })) {
        return [];
    }
    let newNcList = ncList.slice();
    newNcList.push(NameCount.makeNew(name, count));
    return newNcList;
}

let chop_copy = (list: any, removeValue: any): any[] => {
    let copy: any[] = [];
    list.forEach((value: any) => {
        if (value === removeValue) {
            removeValue = undefined;
        } else {
            copy.push(value);
        }
    });
    return copy;
};

type NameListContainer = {
    nameList: string[];
}

type CountListContainer = {
    countList: number[];
}

type NcListContainer = {
    ncList: NameCount.List;
}

type VSFlags = {
    validateAll: boolean;
    fast: boolean|undefined;
}

//
//
const getAllSourcesForPrimaryClueName = (name: string,
    allCandidates = ClueManager.getAllCandidates()): number[] =>
{
    let sources: number[] = [];
    // add candidates clue sources
    allCandidates
        .filter((container: Sentence.CandidatesContainer) =>
            _.has(container.nameIndicesMap, name))
        .forEach((container: Sentence.CandidatesContainer) => {
            // add all sources for name to sources list
            const compatibleIndices = container.nameIndicesMap[name];
            for (let index of compatibleIndices) {
                const candidate = container.candidates[index];
                Assert(candidate && _.has(candidate.nameSourcesMap, name));
                Assert(!_.isEmpty(candidate.nameSourcesMap[name]));
                sources.push(...candidate.nameSourcesMap[name]);
            }
        });
    if (_.isEmpty(sources)) throw new Error(`can't find: ${name}`);
    return sources;
};

const addUsedSourcesFromNameSrcList = (usedSources: Source.UsedSources,
    nameSrcList: NameCount.List): boolean =>
{
    return nameSrcList.every(nameSrc =>
        Source.addUsedSource(usedSources, nameSrc.count, true));
}

const addCompoundNc = (to: ValidateResultData, result: ValidateResultData): void => {
    to.ncList.push(...result.ncList);
    to.nameSrcList.push(...result.nameSrcList);
};

const addPrimaryNameSrc = (to: ValidateResultData, nc: NameCount.Type,
    nameSrc: NameCount.Type): void =>
{
    to.ncList.push(nc); // || NameCount.makeNew(nameSrc.name, 1));
    to.nameSrcList.push(nameSrc);
};

interface MergeNcListComboResult {
    success: boolean;
    validateResult?: ValidateResult;
}

type NativeMergeResult = ValidateResult | null;

export let merge_nclc = 0;

const mergeNcListCombo = (ncList: NameCount.List, indexList: number[]):
    MergeNcListComboResult =>
{
    ++merge_nclc;
    let validateResult = emptyValidateResult();
    // indexList value is either an index into a resultMap.list (compound clue)
    // or a primary source (primary clue)
    for (let i = 0; i < indexList.length; ++i) {
        const nc = ncList[i];
        if (nc.count > 1) { // compound clue
            const listIndex = indexList[i];
            //const ncResult = Native.somethingsomething
            const ncResult = ClueManager.getNcResultMap(nc.count)[nc.toString()].list[listIndex];
            if (!Source.isXorCompatible(validateResult, ncResult)) {
                return { success: false };
            }
            Source.mergeUsedSourcesInPlace(validateResult.usedSources, ncResult.usedSources);
            addCompoundNc(validateResult, ncResult);
        } else { // primary clue
            const primarySrc = indexList[i];
            if (!Source.addUsedSource(validateResult.usedSources, primarySrc, true)) {
                return { success: false };
            }
            addPrimaryNameSrc(validateResult, NameCount.makeNew(nc.name, primarySrc), nc);
        }
    }
    return { success: true, validateResult };
};

const list_sizes = (lists: any[][]): number[][] => {
    return lists.map(list => [ list.length ]);
};

type MergeNcListResultsArgs = VSFlags;

export let native_merge_nclc = 0;
export let native_get_num_ncr = 0;
export let merge_nclr = 0;

const mergeNcListResults = (ncListToMerge: NameCount.List,
    args: MergeNcListResultsArgs): ValidateSourcesResult =>
{
    ++merge_nclr;
    let resultList: ValidateResult[] = Native.mergeNcListResults(ncListToMerge);
    
    /*
    let listArray: number[][] = ncListToMerge.map(nc => {
        if (nc.count === 1) {
            // TODO: optimization: these could be cached. i'm not sure it'd
            // matter too much.
            return getAllSourcesForPrimaryClueName(nc.name);
        } else {
            ++native_get_num_ncr;
            //const count = ClueManager.getNcResultMap(nc.count)[nc.toString()].list.length;
            const count = Native.getNumNcResults(nc);
            return [...Array(count).keys()].map(_.toNumber);
        }
    });

    //console.error(`${stringify(list_sizes(listArray))}`);

    ///* NATIVE
    let resultList: ValidateResult[] =
        Native.mergeAllNcListCombinations(ncListToMerge, listArray);

    let resultList: ValidateResult[] = Peco.makeNew({ listArray })
        .getCombinations()
        //.map((indexList: number[]) => mergeNcListCombo(ncListToMerge, indexList))
        //.filter((mergeResult: MergeNcListComboResult) => mergeResult.success)
        //.map((mergeResult: MergeNcListComboResult) => mergeResult.validateResult!);
        ///* NATIVE
        .map((indexList: number[]) => {
            ++native_merge_nclc;
            return Native.mergeNcListCombo(ncListToMerge, indexList);
        })
        .filter((mergeResult: NativeMergeResult) => !!mergeResult);
    */

    resultList.forEach((result: ValidateResult) => {
        result.usedSources = Source.emptyUsedSources();
            result.nameSrcList.forEach((nameSrc: NameCount.Type) => {
                Source.addUsedSource(result.usedSources, nameSrc.count);
        });
    });
    return { success: !_.isEmpty(resultList), list: resultList };
};

type VSForNameCountArgs = NameListContainer & CountListContainer
    & NcListContainer & VSFlags;

let validateSourcesForNameCount = (clueName: string|undefined, srcName: string,
    srcCount: number, args: VSForNameCountArgs): ValidateSourcesResult =>
{
    Debug(`++validateSourcesForNameCount(${clueName}), ${srcName}:${srcCount}` +
        `, validateAll: ${args.validateAll} ${indentNewline()}` +
        `  ncList: ${args.ncList}, nameList: ${args.nameList}`);

    let ncList = copyAddNcList(args.ncList, srcName, srcCount);
    if (_.isEmpty(ncList)) {
        // TODO:
        // duplicate name:count entry. technically this is allowable for
        // count > 1 if the there are multiple entries of this clue name
        // in the clueList[count]. (at least as many entries as there are
        // copies of name in ncList)
        // SEE ALSO: copyAddNcList()
        // NOTE: this should be fixable with some effort if it ever fires.
        console.error(`  duplicate nc, ${srcName}:{srcCount}`);
        return { success: false }; // fail
    }
    Debug(`  added nc ${srcName}:${srcCount}, ncList.length: ${ncList.length}`);
    // If only one name & count remain, we're done.
    // (name & count lists are equal length, just test one)
    if (args.nameList.length === 1) {
        let result: ValidateSourcesResult;
         // NOTE getting rid of this validateAll check might fix --copy-from, --add, etc.
        if (args.fast && args.validateAll) {
            result = mergeNcListResults(ncList, args);
        } else {
            Assert(0, "was curious if this was used, didn't think it was (it shouldn't be)");
            result = OldValidator.checkUniqueSources(ncList, args);
            Debug(`checkUniqueSources --- ${result.success ? 'success!' : 'failure'}`);
        }
        if (result.success) {
            args.ncList.push(NameCount.makeNew(srcName, srcCount));
            Debug(`  added ${srcName}:${srcCount}, ncList(${ncList.length}): ${ncList}`);
        }
        return result;
    }
    
    // nameList.length > 1, remove current name & count,
    // and validate remaining
    Debug(` calling validateSourcesForNameAndCountLists recursively, ncList: ${ncList}`);
    let rvsResult = validateSourcesForNameAndCountLists(clueName,
        chop_copy(args.nameList, srcName), chop_copy(args.countList, srcCount), {
            ncList,
            fast: args.fast,
            validateAll: args.validateAll
        });
    if (!rvsResult.success) {
        Debug('--validateSourcesForNameCount: validateSourcesForNameCountLists failed');
        return rvsResult;
    }
    // does this achieve anything? modifies args.ncList. answer: probably.
    // TODO: probably need to remove why that matters. answer: maybe.
    // TODO: use slice() (or clone()?)
    args.ncList.length = 0;
    ncList.forEach(nc => args.ncList.push(nc));
    Debug(`--validateSourcesForNameCount, add ${srcName}:${srcCount}` +
          `, ncList(${ncList.length}): ${ncList}`);
    return rvsResult;
};

type VSForNameCountListsArgs = NcListContainer & VSFlags;

let validateSourcesForNameAndCountLists = (clueName: string|undefined, nameList: string[],
    countList: number[], args: VSForNameCountListsArgs):
    ValidateSourcesResult =>
{
    logLevel++;
    Debug(`++validateSourcesForNameCountLists, looking for [${nameList}] in [${countList}]`);
    //if (xp) Expect(nameList.length).is.equal(countList.length);

    // optimization: could have a map of count:boolean entries here
    // on a per-name basis (new map for each outer loop; once a
    // count is checked for a name, no need to check it again

    let resultList: ValidateResult[] = [];
    const name = nameList[0];
    // TODO: could do this test earlier, like in calling function, check entire name list.
    if (name === clueName) {
        return { success: false, list: undefined };
    }
    let success = countList
        .filter((count: number) => ClueManager.isKnownNc({ name, count }))
        .some((count: number) => {
            let rvsResult = validateSourcesForNameCount(clueName, name, count, {
                nameList,
                countList,
                ncList: args.ncList,
                fast: args.fast,
                validateAll: args.validateAll
            });
            if (!rvsResult.success) return false; // some.continue;
            Debug(`  validateSourcesForNameCount output for: ${name}`+
                `, ncList(${args.ncList.length}): ${args.ncList}`);
            /*
            // sanity check - why? lost to time
            if (!args.validateAll && (args.ncList.length < 2)) {
                // TODO: add "allowSingleEntry" ?
                // can i check vs. clueNameList.length?
                // throw new Error('list should have at least two entries1');
            }
            */
            resultList = rvsResult.list!;
            return true; // success: some.exit
        });
    --logLevel;
    return {
        success,
        list: success ? resultList : undefined
    };
};

export const validateSources = (clueName: string|undefined, args: any):
    ValidateSourcesResult =>
{
    Debug(`++validateSources(${clueName})` +
          `${indentNewline()}  nameList(${args.nameList.length}): ${args.nameList}` +
          `, sum(${args.sum})` +
          `, count(${args.count})` +
          `, validateAll: ${args.validateAll}`);

    let success = false;
    let resultList: ValidateResult[] = [];
    Peco.makeNew({
        sum:   args.sum,
        count: args.count,
        max:   args.max
    }).getCombinations().some((countList: number[]) => {
        /*
        let rvsResult = validateSourcesForNameAndCountLists(clueName,
          args.nameList, countList, {
            ncList: [],
            fast: args.fast,
            validateAll: args.validateAll
        });
        let sourceList = rvsResult.success ? rvsResult.list : [];
        */
        let sourceList = Native.validateSourcesForNameAndCountLists(clueName,
            args.nameList, countList, []);
        if (sourceList.length) {
            Debug('validateSources: VALIDATE SUCCESS!');
            //if (rvsResult.list) {
            // TODO: return empty array, get rid of .success
            resultList.push(...sourceList);
            //}
            success = true;
            if (!args.validateAll) return true; // found a match; some.exit
            Debug('validateSources: validateAll set, continuing...');
        }
        return false; // some.continue
    });
    Debug('--validateSources');

    return {
        success,
        list: success ? resultList : undefined
    };
};
