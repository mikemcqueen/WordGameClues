//
// show-components.js
//
// "-t" support.
//
// Lots of dead unused code in here, generally a mess.
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Native      = require('../../../build/experiment.node');
const Assert      = require('assert');
const Debug       = require('debug')('components');
const Expect      = require('should/as-function');
const JStringify  = require('javascript-stringify').stringify;

import * as ClueList from '../types/clue-list';
import * as ClueManager from './clue-manager';
import * as NameCount from '../types/name-count';
import * as PreCompute from './cm-precompute';

export const addRemove = (names: string[], counts: number[], options: any): number => {
    const count = ClueManager.addRemoveOrReject({
        add:      options.add,
        remove:   options.remove,
    /*  property: options.property,
        reject:   options.reject
    */
    }, names, counts, {
        save : options.save,
        addMax: options.max ? Number(options.max) : 30,
        removeMin: 0
    });
    if (options.add || options.remove) {
        console.log(`${options.add ? "added" : "removed"} ${count} clues` +
            ` ${options.save ? "[SAVED]": ""}`);
    }
    return count;
};

const addClues = (names: string[], nc_list: NameCount.List, options: any): void => {
    let num_added = 0;
    while (nc_list.length) {
        const add = nc_list[0].name;
        const counts = nc_list.filter(nc => nc.name === add).map(nc => nc.count);
        num_added += addRemove(names, counts, {
            add,
            save: options.save,
            addMax: options.max_sources,
            removeMin: 0
        });
        nc_list = nc_list.filter(nc => nc.name !== add);
    }
};

export const show = (options: any): any => {
    Expect(options).is.an.Object();
    Expect(options.test).is.a.String();
    if (options.reject) {
        Expect(options.add).is.undefined();
    }
    options.fast = true; // force fast
    console.log(`test: ${options.test}, fast=${options.fast}`);

    const nameList = options.test.split(',').sort();
    const pc_args = {
        xor: nameList,
        merge_only: true,
        max: 2,
        max_sources: options.max_sources,
        quiet: options.quiet,
        verbose: options.verbose,
        ignoreErrors: options.ignoreErrors
    };
    const pc_result = PreCompute.preCompute(2, options.max_sources, pc_args);
    if (pc_result) {
        const counts = Native.showComponents(nameList);
        options.save = true; // hacky
        addRemove(nameList, counts, options);
    } else {
        console.error(`Precompute failed.`);
    }
    return 0;
};

//////// 

const get_unique_combos = (first: number, last: number): Set<string> => {
    let combos = new Set<string>();
    for (let i = first; i <= last; ++i) {
        const clues = ClueManager.getClueList(i) as ClueList.Compound;
        if (!clues) continue;
        for (let clue of clues) {
            if (!combos.has(clue.src)) {
                combos.add(clue.src);
            }
        }
    }
    return combos;
};

type ConsistencyResultMap = {
    [key: string]: NameCount.List;
};

export const consistency_check = (options: any): void => {
    const version = _.includes(options.flags, '3') ? 2 : 1;
    console.error(`consistency check v${version}`);
    let combos = get_unique_combos(2, options.max_sources);
    let v1_results = new Set<string>();
    let v2_results: ConsistencyResultMap;
    for (let combo of combos) {
        const nameList = combo.split(',').sort();
        let valid = true;
        if (version === 1) {
            const pc_args = {
                xor: nameList,
                merge_only: true,
                max: 2,
                max_sources: options.max_sources,
                quiet: true,
                verbose: options.verbose,
                ignoreErrors: options.ignoreErrors
            };
            valid = PreCompute.preCompute(2, options.max_sources, pc_args);
            if (!valid) {
                console.error(`\nConsistency::Precompute failed at ${combo}.`);
            }
        }
        if (valid) {
            if (!Native.checkClueConsistency(nameList, options.max_sources, version)) {
                if (version === 1) v1_results.add(combo);
            }
        }
        if (valid && !options.quiet && !options.verbose) {
            process.stderr.write('.');
        }
    }
    let count = v1_results.size;
    if (version === 2) {
        v2_results = Native.getConsistencyCheckResults();
        count = Object.keys(v2_results).length;
    }
    if (count) {
        console.error(`\nInconsistent combos:`);
        if (version === 1) {
            for (let result of v1_results) {
                console.error(result);
            }
        } else {
            for (let source_csv of Object.keys(v2_results!)) {
                const nc_list = v2_results![source_csv];
                console.error(`${source_csv}: ${NameCount.listToString(nc_list)}`);
                addClues(source_csv.split(','), nc_list, options);
            }
        }
    } else {
        console.error(`\nNo inconsistent combos`);
    }
};

/*
///////////////////////////////////////////////////////////////////////////////
////////
//////// all below probably should be removed.
////////
///////////////////////////////////////////////////////////////////////////////

function get_clue_names (options: any) {
    let result: string[] = [];
    if (options.count_lo) {
        //console.log(`get_clue_names: lo(${options.count_lo}) hi(${options.count_hi})`);
        for (let count = options.count_lo; count <= options.count_hi; count += 1) {
            result.push(...ClueManager.getClueList(count).map(clue => clue.name));
        }
        result = _.uniq(result);
    }
    return result;
}

function addOrRemove (args, nameList, countSet, options) {
    if (!options.add && !options.remove) return;
    const save = _.isUndefined(options.save) ? true : options.save;
    const count = ClueManager.addRemoveOrReject({
        add:      options.add,
        remove:   options.remove,
        property: options.property,
    }, nameList, countSet, { save });
    console.log(`${options.add ? "added" : "removed"} ${count} clues`);
}

function getCompatiblePrimaryNameSrcList (listOfListOfPrimaryNameSrcLists) {
    const listArray = listOfListOfPrimaryNameSrcLists.map(listOfNameSrcLists =>
        [...Array(listOfNameSrcLists.length).keys()]); // 0..nameSrcList.length-1
    let comboLists = Peco.makeNew({
        listArray,
        max: listOfListOfPrimaryNameSrcLists.reduce((sum, listOfNameSrcLists) =>
            sum + listOfNameSrcLists.length, 0)
    }).getCombinations();
    for (const comboList of comboLists) {
        const nameSrcList = comboList.reduce((nameSrcList, comboListValue, comboListIndex) => {
            let nsList = listOfListOfPrimaryNameSrcLists[comboListIndex][comboListValue];
            if (!nsList || !_.isArray(nsList)) {
                console.log(`nsList: ${nsList}, value ${comboListValue} index ${comboListIndex} lolPnsl(${comboListIndex}):` +
                            ` ${Stringify(listOfListOfPrimaryNameSrcLists[comboListIndex])} nameSrcList ${Stringify(nameSrcList)}`);
                console.log(`lolopnsl: ${Stringify(listOfListOfPrimaryNameSrcLists)}`);
            }
            nameSrcList.push(...nsList);
            return nameSrcList;
        }, []);
        const uniqNameSrcList = _.uniqBy(nameSrcList, NameCount.count);
        if (uniqNameSrcList.length === nameSrcList.length) return uniqNameSrcList;
    }
    return null;
}

function buildSubListFromIndexList (nameList: string[], indexList: number[]):
    string[]
{
    const subList: string[] = [];
    indexList.forEach(index => subList.push(nameList[index]));
    return subList;
}

function addValidResults (validResults, validResultList, options) {
    validResultList.forEach(result => {
        Expect(result.valid).is.ok();
        let names = result.ncList.slice(options.slice_index).map(nc => nc.name).sort().toString();
        if (!_.has(validResults, names)) {
            validResults[names] = [];
        }
        validResults[names].push(result);
        //console.log(`${result.ncList} : VALID (${result.sum}): ${result.compatibleNameSrcList} `);
    });
}

function display_valid_results (validResults) {
    console.error('++display');
    Object.keys(validResults).forEach(key => console.log(key));
}

function showNcLists (ncLists) {
    for (let ncList of ncLists) {
        console.log(`${ncList}`);
    }
}

////////

function readlines(filename): string[] {
    const path = Path.normalize(`${Path.dirname(module.filename)}/tools/${filename}`);
    const readLines = new Readlines(filename);
    let lines: string[] = [];
    let line;
    while ((line = readLines.next()) !== false) {
        lines.push(line.toString().trim());
    }
    return lines;
}

function valid_combos(combo_list: string[], options: any = {}): string[] {
    const combos: string[] = [];
    options.any = true;

    combo_list.forEach(combo_str => {
        //onst combo_str = _.join(combo, ',');
        //Debug(`${typeof(combo)}: ${combo}`);
        Debug(`${typeof(combo_str)}: ${combo_str} (${combo_str.split(',').length})`);
        if (combo_str.split(',').length > 3) return; // continue
//      const result = ClueManager.getCountListArrays(combo_str, options);
        const result = show( {test: combo_str, save: false, fast: true });
        if (!result || !(result.known.length + result.valid.length)) {
            return;
        }
        //showCountListArray(result.invalid, 'INVALID');
        //showCountListArray(result.known, 'PRESENT as', true);
          //showCountListArray(result.clues, 'PRESENT as clue with sources:', true);
        //showCountListArray(result.valid, 'VALID');
        Debug(`valid_combos adding: ${combo_str}`);
        combos.push(combo_str);
    });
    return combos;
}

function all_combos(input_list: string[], word_list: string[]): string[] {
    const combos: string[] = [];
    for (const csvInput of input_list) {
        const input = csvInput.split(',');
        for (const word of word_list) {
            const combo = _.concat(input, word);
            // sort here, then sortedUniq. then no sort at join.
            const uniq = _.sortedUniq(combo.sort());
            if (combo.length != uniq.length) {
                continue;
            }
            Debug(`all_combos adding: ${uniq}`);
            combos.push(_.join(uniq, ','));
        }
    }
    return _.uniq(combos);
}

function validate_sources(lines, options) {
    for (const line of lines) {
        const result: any = {}; // REPLACE: ClueManager.getCountListArrays(line, options);
        if (!result || !result.valid) {
            console.log(`${line} ${!result ? 'doesnt exist??' : result.invalid ? 'invalid' : 'rejected'}`);
        }
    }
}

function validate_combos(lines, options): void {
    const combos: string[] = [];
    let input = lines;
    for (;;) {
        Timing(``);
        const raw_combo_list = all_combos(input, lines);
        Timing(`all_combos (${raw_combo_list.length})`);
        if (_.isEmpty(raw_combo_list)) break;
        Debug(`raw: ${typeof(raw_combo_list)}, len: ${raw_combo_list.length}: ${raw_combo_list}`);
        let valid_combo_list = valid_combos(raw_combo_list);
        if (_.isEmpty(valid_combo_list)) break; 
        Timing(`valid combos (${valid_combo_list.length})`);
        combos.push(...valid_combo_list);
        input = valid_combo_list;
    }
    combos.forEach(combo => {
        console.log(`${combo}`);
    });
}

export function validate (filename, options: any = {}) {
    Assert(0 && "validate broken, don't use ClueManager.getCountListArrays");
    const lines = readlines(filename);
    if (options.combos) {
        validate_combos(lines, options);
    } else {
        validate_sources(lines, options);
    }
}
*/
