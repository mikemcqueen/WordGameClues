//
// validator.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Peco        = require('../../modules/peco');
const ResultMap   = require('../../types/result-map');

//const ClueManager = require('./clue-manager');
const Assert      = require('assert');
const Debug       = require('debug')('validator');
const Expect      = require('should/as-function');
const stringify	  = require('javascript-stringify').stringify;
const Stringify2  = require('stringify-object');
const Timing      = require('debug')('timing');

import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import * as ClueManager from './clue-manager';
import * as NameCount from '../types/name-count';
import * as OldValidator from './old-validator';

function Stringify(val: any) {
    return stringify(val, (value: any, indent: any, stringify: any) => {
	if (typeof value === 'function') return "function";
	return stringify(value);
    }, " ");
}

//

let logLevel         = 0;

//

interface ValidateResultData {
    ncList: NameCount.List;
    resultMap: any;
    nameSrcList: NameCount.List;
    srcBits?: any;
    nameSrcCsv?: string;
    propertyCounts?: Clue.PropertyCounts.Map;
    //primarySrcArray?: CountArray;
}

export type ValidateResult = ValidateResultData & ClueManager.AllCandidatesContainer;

export interface ValidateSourcesResult {
    success: boolean;
    list?: ValidateResult[];
}

export type NumberArray = number[]; // TODO: Int32Array

//
//
let spaces = (length: number): string => {
    return ' '.repeat(length);
};

let indent = (): string => {
    return spaces(logLevel);
};

let indentNewline = (): string => {
    return '\n' + indent();
};

//
//

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

//
//
let chop = (list: any, removeValue: any): any => {
    let copy: any[] = [];
    list.forEach((value: any) => {
        if (value == removeValue) {
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
let getRestrictedPrimaryClueNumber = (nameSrc: NameCount.Type): number => {
    let clue = ClueManager.getPrimaryClue(nameSrc);
    return clue.restrictToSameClueNumber ? clue.num : 0;
};

//
//
let allHaveSameClueNumber = (nameSrcList: NameCount.List, clueNumber: number): boolean => {
    return nameSrcList.every(nameSrc => ClueManager.getPrimaryClue(nameSrc).num === clueNumber);
};

//
//
let getAllSourcesForPrimaryClueName = (name: string, allCandidates: ClueManager.AllCandidates):
    number[] =>
{
//    let clueList: { name: string, src: string }[] = ClueManager.getClueList(1);
    let clueList = ClueManager.getClueList(1) as ClueList.Primary;
    let sources: number[] = clueList.filter(clue => clue.name === name)
	.map(clue => _.toNumber(clue.src));
    // Add "sentence" candidates to sources list.
    // TODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODO
    //
    // TODO: We should probably be removing both the used & incompatible
    // candidates from all lists at this point as well.
    //
    // THIS is probably slow. make it faster with pure for loops.
    //
    // TODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODO
    
    allCandidates
	.filter(container => _.has(container.nameIndicesMap, name))
	.forEach(container => {
	    // add all sources from name to sources list
	    const compatibleIndices = container.nameIndicesMap[name];
	    for (let index of compatibleIndices) {
		if (container.candidates[index]) { // might have been deleted
		    Assert(_.has(container.candidates[index].nameSourcesMap, name));
		    Assert(!_.isEmpty(container.candidates[index].nameSourcesMap[name]));
		    sources.push(...container.candidates[index].nameSourcesMap[name]);
		}
	    }
	    // remove all incompatible candidates from candidate list
	    for (let index = 0; index < container.candidates.length; ++index) {
		if (!_.has(container.candidates[index].nameSourcesMap, name)) {
		    delete container.candidates[index];
		}
	    }
	});

    if (_.isEmpty(sources)) throw new Error(`can't find: ${name}`);
    return sources;
}

type MergeNcListResultsArgs = ClueManager.AllCandidatesContainer & VSFlags;

let mergeNcListResults = (ncListToMerge: NameCount.List,
    args: MergeNcListResultsArgs): ValidateSourcesResult =>
{
    let ncCsv = ncListToMerge.toString();
    let arrayList: NumberArray[] = ncListToMerge.map(nc => {
        if (nc.count === 1) {
            return getAllSourcesForPrimaryClueName(nc.name, args.allCandidates);
        } else {
            const ncResultMap = ClueManager.getNcResultMap(nc.count);
            return [...Array(ncResultMap[nc.toString()].list.length).keys()].map(_.toNumber);
        }
    });
    let resultList: ValidateResult[] = [];
    Peco.makeNew({
        listArray: arrayList,
        max: 99999
    }).getCombinations().forEach((indexList: number[]) => {
        let ncList: NameCount.List = [];
        let nameSrcList: NameCount.List = [];
        let resultMap = ResultMap.makeNew();
        let restrictToClueNumber = 0;
	// TODO: resultIndex kinda bad name, can also be clue source for primary clues
        indexList.forEach((resultIndex, ncIndex) => {
            let nc = ncListToMerge[ncIndex];
            if (nc.count > 1) {
                let result = ClueManager.getNcResultMap(nc.count)[nc.toString()].list[resultIndex];
                ncList.push(...result.ncList);
                nameSrcList.push(...result.nameSrcList);
                resultMap.addNcMapSource(nc, result.resultMap);
            } else {
                ncList.push(nc);
                let nameSrc = NameCount.makeNew(nc.name, resultIndex);
                nameSrcList.push(nameSrc);
                resultMap.addPrimarySource(nameSrc);
                let clueNumber = getRestrictedPrimaryClueNumber(nameSrc);
                if (clueNumber) {
                    restrictToClueNumber = clueNumber;
                }
            }
        });
        if (restrictToClueNumber) {
            if (!allHaveSameClueNumber(nameSrcList, restrictToClueNumber)) {
                return; // forEach.continue;
            }
        }
        // TODO: uniqBy da debil
        if (_.uniqBy(nameSrcList, NameCount.count).length === nameSrcList.length) {
            let nameSrcCsv = _.sortBy(nameSrcList, NameCount.count).toString();
            let result: ValidateResult = {
                ncList,
                resultMap,
                nameSrcList,
                nameSrcCsv,
		allCandidates: args.allCandidates
            };
            resultList.push(result);
	}
    });
    return { list: resultList, success: !_.isEmpty(resultList) };
};

//
//
let test = (ncList: NameCount.List, args: any): ValidateSourcesResult => {
    // can remove this.
    if (!ncList.every(nc => {
        let ncResultMap = ClueManager.getNcResultMap(nc.count);
        let ncStr = nc.toString();
        if (nc.count === 1 || (ncResultMap[ncStr] && ncResultMap[ncStr].list)) {
            return true;
        }
        return false;
    })) throw new Error('no result list');
    return mergeNcListResults(ncList, args);
};

type VSForNameCountArgs = NameListContainer & CountListContainer
    & ClueManager.AllCandidatesContainer & NcListContainer & VSFlags;

let validateSourcesForNameCount = (name: string, count: number, args: VSForNameCountArgs):
    ValidateSourcesResult =>
{
    Debug(`++rvsWorker, ${name}:${count}` +
        `, validateAll: ${args.validateAll} ${indentNewline()}` +
	`  ncList: ${args.ncList}, nameList: ${args.nameList}`);

    let ncList = copyAddNcList(args.ncList, name, count);
    if (_.isEmpty(ncList)) {
        // TODO:
        // duplicate name:count entry. technically this is allowable for
        // count > 1 if the there are multiple entries of this clue name
        // in the clueList[count]. (at least as many entries as there are
        // copies of name in ncList)
        // SEE ALSO: copyAddNcList()
        Debug(`++rvsWorker, duplicate name:count, ${name}:{count}`);
        return { success: false }; // fail
    }
    Debug(`added NC ${name}:${count}, ncList.length: ${ncList.length}`);
    // If only one name & count remain, we're done.
    // (name & count lists are equal length, just test one)
    if (args.nameList.length === 1) {
        let result: ValidateSourcesResult;
        if (args.fast && args.validateAll) { // NOTE getting rid of this validateAll check might fix --copy-from, --add, etc.
            result = mergeNcListResults(ncList, args);
        } else {
            result = OldValidator.checkUniqueSources(ncList, args);
            Debug(`checkUniqueSources --- ${result.success ? 'success!' : 'failure'}`);
        }
        if (result.success) {
            args.ncList.push(NameCount.makeNew(name, count));
            Debug(`add1, ${name}:${count}, ncList(${ncList.length}): ${ncList}`);
        }
        return result;
    }
    
    // nameList.length > 1, remove current name & count,
    // and validate remaining
    Debug(`calling rvs recursively, ncList: ${ncList}`);
    let rvsResult = validateSourcesForNameCountLists(chop(args.nameList, name),
        chop(args.countList, count), {
	    allCandidates: ClueManager.copyAllCandidates(args.allCandidates),
	    ncList,
	    fast: args.fast,
	    validateAll: args.validateAll
	});
    if (!rvsResult.success) {
        Debug('--rvsWorker, recursiveValidateSources failed');
        return rvsResult;
    }
    // does this achieve anything? modifies args.ncList.
    // TODO: probably need to remove why that matters.
    // TODO2: use _clone() until then
    args.ncList.length = 0;
    ncList.forEach(nc => args.ncList.push(nc));
    Debug(`--rvsWorker, add ${name}:${count}` +
          `, ncList(${ncList.length}): ${ncList}`);
    return rvsResult;
};

type VSForNameCountListsArgs = ClueManager.AllCandidatesContainer & NcListContainer & VSFlags;

let validateSourcesForNameCountLists = (nameList: string[], countList: number[],
    args: VSForNameCountListsArgs): ValidateSourcesResult =>
{
    logLevel++;
    Debug(`++recursiveValidateSources, looking for [${nameList}] in [${countList}]`);
    //if (xp) Expect(nameList.length).is.equal(countList.length);

    // optimization: could have a map of count:boolean entries here
    // on a per-name basis (new map for each outer loop; once a
    // count is checked for a name, no need to check it again

    let resultList: ValidateResult[] = [];
    const clueName = nameList[0];
    let success = countList
	.filter(count => _.has(ClueManager.getKnownClueMap(count), clueName))
        .some(count => // .every !!??
    {
        let rvsResult = validateSourcesForNameCount(clueName, count, {
            nameList,
            countList,
	    allCandidates: args.allCandidates,
            ncList: args.ncList,
            fast: args.fast,
            validateAll: args.validateAll
        });
        if (!rvsResult.success) return false; // some.continue;
        Debug(`  rvsWorker output for: ${clueName}, ncList(${args.ncList.length}) ${args.ncList}`);
        // sanity check
        if (!args.validateAll && (args.ncList.length < 2)) {
            // TODO: add "allowSingleEntry" ?
            // can i check vs. clueNameList.length?
            // throw new Error('list should have at least two entries1');
        }
        resultList = rvsResult.list!;
        return true; // success: some.exit
	// TODO::: WAIT WHAT?? EARLY EXIT ON SUCCESS? WITHOUT CHECKING ALL VARIATIONS?
	// I AM NOT SURE IF THAT'S TRUE BUT I SHOULD UNDERSTAND THIS
    });
    --logLevel;

    return {
        success,
        list: success ? resultList : undefined
    };
};

//
//
export const validateSources = (args: any): ValidateSourcesResult => {
    Debug('++validateSources' +
          `${indentNewline()}  nameList(${args.nameList.length}): ${args.nameList}` +
          `, sum(${args.sum})` +
          `, count(${args.count})` +
          `, validateAll: ${args.validateAll}`);

    let found = false;
    let resultList: ValidateResult[] = [];
    Peco.makeNew({
        sum:     args.sum,
        count:   args.count,
        max:     args.max,
        quiet:   args.quiet
    }).getCombinations().some((countList: number[]) => {
        let rvsResult = validateSourcesForNameCountLists(args.nameList, countList, {
	    allCandidates: ClueManager.copyAllCandidates(),
	    ncList: [],
            fast: args.fast,
            validateAll: args.validateAll
        });
        if (rvsResult.success) {
            Debug('validateSources: VALIDATE SUCCESS!');
            if (rvsResult.list) {
		resultList.push(...rvsResult.list);// TODO: return empty array, get rid of .success
	    }
            found = true;
            if (!args.validateAll) return true; // found a match; some.exit
            // validatingg all, continue searching
            Debug('validateSources: validateAll set, continuing...');
        }
        return false; // some.continue
    }, this);
    Debug('--validateSources');

    return {
        success: found,
        list: found ? resultList : undefined
    };
};
