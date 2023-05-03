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
import * as CountBits from '../types/count-bits-fastbitset';
import * as NameCount from '../types/name-count';
import * as OldValidator from './old-validator';
import * as Sentence from '../types/sentence';
import * as Source from './source';

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
    sourceBits?: CountBits.Type;
    usedSources?: Source.UsedSources;
    nameSrcCsv?: string; // TODO: remove; old-validator uses it, stop using old-validator
    //propertyCounts?: Clue.PropertyCounts.Map;
    //primarySrcArray?: CountArray;
}

// TODO: & Source.CompatibilityData (when no longer optional)
export type ValidateResult = ValidateResultData & ClueManager.AllCandidatesContainer;

export interface ValidateSourcesResult {
    success: boolean;
    list?: ValidateResult[];
}

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

// part of restrictToSameClueNumber logic
//
let getRestrictedPrimaryClueNumber = (nameSrc: NameCount.Type): number => {
    let clue = ClueManager.getPrimaryClue(nameSrc);
    return clue?.restrictToSameClueNumber ? clue.num : 0;
};

// part of restrictToSameClueNumber logic
//
let allHaveSameClueNumber = (nameSrcList: NameCount.List, clueNumber: number): boolean => {
    return nameSrcList.every(nameSrc => ClueManager.getPrimaryClue(nameSrc)?.num === clueNumber);
};

//
//
let getAllSourcesForPrimaryClueName = (name: string, allCandidates: ClueManager.AllCandidates):
    number[] =>
{
    //let clueList: { name: string, src: string }[] = ClueManager.getClueList(1);
    let clueList = ClueManager.getClueList(1) as ClueList.Primary;
    let sources: number[] = clueList.filter(clue => clue.name === name)
	.map(clue => _.toNumber(clue.src));

    // TODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODO
    //
    // TODO: We should probably be removing both the used & incompatible
    // candidates from all lists at this point as well.
    //
    // TODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODOTODO
    
    let log = false; //name === 'town';

    // Add "sentence" candidates to sources list.
    allCandidates.filter(container => {
	let has = _.has(container.nameIndicesMap, name);
	if (log) console.error(`filter ${container}, ${name}: ${has}`);
	return has;
    }).forEach(container => {
	// add all sources for name to sources list
	let numValidIndices = 0;
	const compatibleIndices = container.nameIndicesMap[name];
	for (let index of compatibleIndices) {
	    if (!container.candidates[index]) continue; // might have been deleted already
	    numValidIndices += 1;
	    Assert(_.has(container.candidates[index].nameSourcesMap, name));
	    Assert(!_.isEmpty(container.candidates[index].nameSourcesMap[name]));
	    sources.push(...container.candidates[index].nameSourcesMap[name]);
	}
	if (log) console.error(` ${numValidIndices} indices valid of ${[...compatibleIndices].length}`);
    });
    
    if (_.isEmpty(sources)) throw new Error(`can't find: ${name}`);
    return sources;
}

let theName: string = ""

type MergeNcListResultsArgs = ClueManager.AllCandidatesContainer & VSFlags;

const mergeNcListResults = (ncListToMerge: NameCount.List,
    args: MergeNcListResultsArgs): ValidateSourcesResult =>
{
    let ncCsv = ncListToMerge.toString();
    let listArray: number[][] = ncListToMerge.map(nc => {
        if (nc.count === 1) {
            return getAllSourcesForPrimaryClueName(nc.name, args.allCandidates);
        } else {
            const ncResultMap = ClueManager.getNcResultMap(nc.count);
            return [...Array(ncResultMap[nc.toString()].list.length).keys()].map(_.toNumber);
        }
    });
    let resultList: ValidateResult[] = [];
    const combos = Peco.makeNew({
	listArray
    }).getCombinations().forEach((indexList: number[]) => {
	let usedSources: Source.UsedSources = [];
        let ncList: NameCount.List = [];
        let nameSrcList: NameCount.List = [];
        let resultMap = ResultMap.makeNew();
        let restrictToClueNumber = 0;
	// indexList value is either an index into a resultMap.list (compound clue)
	// or a primary source (primary clue)
        for (let i = 0; i < indexList.length; ++i) {
            const nc = ncListToMerge[i];
            if (nc.count > 1) { // compound clue
		const resultListIndex = indexList[i];
                const result = ClueManager.getNcResultMap(nc.count)[nc.toString()]
		    .list[resultListIndex];
		for (let nameSrc of result.nameSrcList) {
		    if (!Source.addUsedSource(usedSources, nameSrc.count, true)) {
			return; // forEach.continue;
		    }
		}
                ncList.push(...result.ncList);
                nameSrcList.push(...result.nameSrcList);
                resultMap.addNcMapSource(nc, result.resultMap);
            } else { // primary clue
		const primarySrc = indexList[i];
		if (!Source.addUsedSource(usedSources, primarySrc, true)) {
		    return; // forEach.continue;
		}
                ncList.push(nc);
                const nameSrc = NameCount.makeNew(nc.name, primarySrc);
                nameSrcList.push(nameSrc);
                resultMap.addPrimarySource(nameSrc);
                const clueNumber = getRestrictedPrimaryClueNumber(nameSrc);
                if (clueNumber) {
                    restrictToClueNumber = clueNumber;
                }
            }
        }
        if (restrictToClueNumber) {
            if (!allHaveSameClueNumber(nameSrcList, restrictToClueNumber)) {
                return; // forEach.continue;
            }
        }
        if (NameCount.listHasCompatibleSources(nameSrcList)) {
            //let nameSrcCsv = _.sortBy(nameSrcList, NameCount.count).toString();
            let result: ValidateResult = {
                ncList,
                resultMap,
                nameSrcList,
		allCandidates: [] // args.allCandidates // TODO avoiding OOM
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
        Debug(`  duplicate nc, ${srcName}:{srcCount}`);
        return { success: false }; // fail
    }
    Debug(`  added nc ${srcName}:${srcCount}, ncList.length: ${ncList.length}`);
    // If only one name & count remain, we're done.
    // (name & count lists are equal length, just test one)
    if (args.nameList.length === 1) {
        let result: ValidateSourcesResult;
        if (args.fast && args.validateAll) { // NOTE getting rid of this validateAll check might fix --copy-from, --add, etc.
	    theName = clueName!;
            result = mergeNcListResults(ncList, args);
        } else {
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
    Debug(`..calling validateSorucesForNameCountLists recursively, ncList: ${ncList}`);
    let rvsResult = validateSourcesForNameCountLists(clueName,
	chop(args.nameList, srcName), chop(args.countList, srcCount), {
	    allCandidates: args.allCandidates, //ClueManager.copyAllCandidates(args.allCandidates),
	    ncList,
	    fast: args.fast,
	    validateAll: args.validateAll
	});
    if (!rvsResult.success) {
        Debug('--validateSourcesForNameCount: validateSourcesForNameCountLists failed');
        return rvsResult;
    }
    // does this achieve anything? modifies args.ncList.
    // TODO: probably need to remove why that matters.
    // TODO2: use _clone() until then
    args.ncList.length = 0;
    ncList.forEach(nc => args.ncList.push(nc));
    Debug(`--validateSourcesForNameCount, add ${srcName}:${srcCount}` +
          `, ncList(${ncList.length}): ${ncList}`);
    return rvsResult;
};

type VSForNameCountListsArgs = ClueManager.AllCandidatesContainer & NcListContainer & VSFlags;

let validateSourcesForNameCountLists = (clueName: string|undefined, nameList: string[],
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
    // could do this test earlier, like in calling function, check entire name list.
    if (name === clueName) {
	return { success: false, list: undefined };
    }
    let success = countList
	.filter(count => ClueManager.isKnownNc({ name, count }))
        .some(count =>
    {
        let rvsResult = validateSourcesForNameCount(clueName, name, count, {
            nameList,
            countList,
	    allCandidates: args.allCandidates,
            ncList: args.ncList,
            fast: args.fast,
            validateAll: args.validateAll
        });
        if (!rvsResult.success) return false; // some.continue;
        Debug(`  validateSourcesForNameCount output for: ${name}`+
	    `, ncList(${args.ncList.length}): ${args.ncList}`);
        // sanity check
        if (!args.validateAll && (args.ncList.length < 2)) {
            // TODO: add "allowSingleEntry" ?
            // can i check vs. clueNameList.length?
            // throw new Error('list should have at least two entries1');
        }
        resultList = rvsResult.list!;
        return true; // success: some.exit
    });
    --logLevel;

    return {
        success,
        list: success ? resultList : undefined
    };
};

//
//
export const validateSources = (clueName: string|undefined, args: any):
    ValidateSourcesResult =>
{
    Debug('++validateSources(${clueName}' +
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
        let rvsResult = validateSourcesForNameCountLists(clueName, args.nameList, countList, {
	    allCandidates: ClueManager.getAllCandidates(), //ClueManager.copyAllCandidates(),
	    ncList: [],
            fast: args.fast,
            validateAll: args.validateAll
        });
        if (rvsResult.success) {
            Debug('validateSources: VALIDATE SUCCESS!');
            if (rvsResult.list) {
		resultList.push(...rvsResult.list);// TODO: return empty array, get rid of .success
	    }
	    //console.error(`resultList(${resultList.length})`);
            found = true;
            if (!args.validateAll) return true; // found a match; some.exit
            // validatingg all, continue searching
            Debug('validateSources: validateAll set, continuing...');
        }
        return false; // some.continue
    });
    Debug('--validateSources');

    return {
        success: found,
        list: found ? resultList : undefined
    };
};
