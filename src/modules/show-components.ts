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
const Peco        = require('../../modules/peco');

const Assert      = require('assert');
const Debug       = require('debug')('show-components');
const Expect      = require('should/as-function');
const Path        = require('path');
const Readlines   = require('n-readlines');
const stringify   = require('javascript-stringify').stringify;
const Stringify2  = require('stringify-object');
const Timing      = require('debug')('timing');

import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import * as ClueManager from './clue-manager';
import * as NameCount from '../types/name-count';
import * as PreCompute from './cm-precompute';
import * as Source from './source';

///////////

interface CountListNameList {
    countList: number[];
    nameList: string[];
}

function Stringify (val) {
    return stringify(val, (value, indent, stringify) => {
        if (typeof value == 'function') return "function";
        return stringify(value);
    }, " ");
}

/*
const getPrimaryClueNameSources = (name: string): string[] => {
    const nc: NameCount.Type = { name, count: 1};
    const primary_sources: string[] = Native.getSourcesForNc(nc);
    return primary_sources;
    //begin
  well it seems this is all unnecessary, and variations names exist in the
  native nameSourcesMap! and I didn't realize that before I spent an hour
  writing code assuming it didn't. which tells me I don't really understand
  this code very well.

    const variation_sources = new Set<string>();
    const variation_names = ClueManager.getAllNamesForVariationName(name);
    for (let v_name of variation_names) {
        const sources: string[] = Native.getSourcesForNc(nc);
        for (let source of sources) {
            variation_sources.add(source);
        }
    }
    // test if a particular name is both used as a "primary" (non-variation)
    // name and as a variation name.
    // nothing wrong with this, i just don't expect it (yet), and want to be
    // notified when it happens. the likelihood that something is wrong when 
    // it does happen seems about equal to the likelihood that it is benign.
    if ((primary_sources.length && variation_sources.size)) {
        console.error(`name: ${name}`);
        console.error(`primary_sources: ${primary_sources}`);
        console.error(`variation_sources: ${[...variation_sources]}`);
        process.exit(-1);
    }
    for (let source of primary_sources) {
        variation_sources.add(source);
    }
    return [...variation_sources];
};

const getCompoundNcSrcList = (nc: NameCount.Type): string[] => {
    Assert(nc.count > 1);
    return ClueManager.getClueList(nc.count)
        .filter(clue => clue.name === nc.name)
        .map(clue => clue.src);
};

const getNcSrcList = (nc: NameCount.Type): string[] => {
    return nc.count === 1 ? getPrimaryClueNameSources(nc.name)
        : getCompoundNcSrcList(nc);
};
*/

/*
const getCountListArrays = (nameList: string[], pcResult: PreCompute.Result,
    options: any): any =>
{
    let addRemoveSet: Set<number> = new Set<number>();
    //if (options.add || options.remove) {
    //  addRemoveSet = new Set<number>();
    //}
    let valid: number[][] = [];
    let known: CountListNameList[] = [];
    let clues: CountListNameList[] = [];
    let invalid: number[][] = [];
    let nameListStr: string = nameList.toString();
    let hash = {};

    for (const xorSource of pcResult.data!.xor as Source.List) {
        const countList = NameCount.listToCountList(xorSource.ncList);
        // for --verbose, we could allow this:
        const hashKey = countList.toString();
        if (hash[hashKey]) continue;
        hash[hashKey] = true;
        // TODO: in order to support this, we'd need to pass a flag to PreCompute to
        // tell it to preserve the filtered incompatible combinations, or manually
        // walk through all ClueManager.knownSourceMaps looking for a sourceCsv combo,
        // and displaying those that *aren't* in the xor list. the latter should be done
        // in a separate loop probably, not in this loop.

        //if (!result.success) {
        //  console.log(`invalid: ${nameList}  CL ${clueCountList}  x ${x} sum ${sum}  validateAll=${validateAll}`);
        //  invalid.push(clueCountList);
        //} else

        const sum = countList.reduce((a, b) => a + b);
        if (nameList.length === 1) {
            const name = nameList[0];
            let srcList = getNcSrcList({ name, count: sum });
            if (srcList.length) {
                clues.push({ countList: [sum], nameList: srcList });
            } else {
                console.log(`hmm, no sources for ${name}:${sum}`);
            }
        } else {
            const sourceMap = ClueManager.getKnownSourceMap(sum);
            if (!sourceMap) {
                console.error(`!sourceMap(${sum}), nameList: ${nameListStr}`);
                //  + ` xorSource.ncList: ${NameCount.listToString(xorSource.ncList)}`);
                continue;
            }
            let sourceData = sourceMap[nameListStr];
            if (sourceData) {
                known.push({
                    countList,
                    nameList: (sourceData.clues as ClueList.Compound).map(clue => clue.name)
                });
            } else {
                valid.push(countList);
            }
            if ((options.add && (sum <= options.addMaxSum))
                  || (options.remove && (sum >= options.removeMinSum))) {
                addRemoveSet.add(sum);
            }
        }
    }
    return { valid, known, invalid, clues, addRemoveSet };
};

function getClueSources (nameList) {
    return nameList.join(' - ');
}

function getSourceClues (source, countList, nameList) {
    const count = countList.reduce((sum, count) => sum + count, 0);
    if (!Native.isKnownSourceMapEntry(count, source)) {
        let sourceList = source.split(',');
        let s = '';
        sourceList.forEach((source, index) => {
            // [source] is wrong at least for primary clue case, need actual list of sources.
            s += getClueSources([source]);
        });
        return s;
    }
    return getClueSources(nameList);
}

function showCountListArray (name, countListArray, text, hasNameList = false) {
    for (const elem of countListArray) {
        const countList = hasNameList ? elem.countList : elem;
        let sources = '';
        if (name) {
            if (countList.length > 1) {
                // -t name1,name2[,name3,...] (multiple names; name == nameCsv here)
                sources += getSourceClues(name, countList, elem.nameList); // 
            } else {
                // -t name (one name only)
                sources += getClueSources(elem.nameList);
            }
        }
        console.log(`${countList} ${text} ${sources}`);
    }
}

const showCountLists = (nameList: string[], result: any, options: any): any => {
    //showCountListArray(null, result.rejects, 'REJECTED');
    showCountListArray(null, result.invalid, 'INVALID');
    showCountListArray(nameList.toString(), result.known, 'PRESENT as', true);
    showCountListArray(nameList.toString(), result.clues, 'PRESENT as clue with source:', true);
    showCountListArray(null, result.valid, 'VALID');

    //
    // *****
    // TODO: honor options.addMaxSum, options.removeMinSum here
    // *****
    //

    // TODO: extract this to helper function, maybe in clue-manager
    // NOTE: explicit undefined check here is necessary
    const save = _.isUndefined(options.save) ? true : options.save;
    // good lord this function call is something else
    const count = ClueManager.addRemoveOrReject({
        add:      options.add,
        remove:   options.remove,
        property: options.property,
        reject:   options.reject,
        isKnown:  !_.isEmpty(result.known),
        isReject: !_.isEmpty(result.reject)
    }, nameList, result.addRemoveSet, {
        save,
        max: options.max ? Number(options.max) : 0
    });
    if (options.add || options.remove) {
        console.log(`${options.add ? "added" : "removed"} ${count} clues`);
    }
    return Object.assign(result, { added: count });
};
*/

export const show = (options: any): any => {
    Expect(options).is.an.Object();
    Expect(options.test).is.a.String();
    if (options.reject) {
        Expect(options.add).is.undefined();
    }
    if (options.add) {
        options.addMaxSum = 30;
    }
    if (options.remove) {
        options.removeMinSum = 0;
    }
    options.fast = true; // force fast
    console.log(`test: ${options.test}, fast=${options.fast}`);

    const nameList = options.test.split(',').sort();
    const pc_args = {
        xor: nameList,
        merge_only: true, // flag: wrap xorSources on return from Native.merge()
        max: 2,
        max_sources: options.max_sources,
        quiet: options.quiet,
        ignoreErrors: options.ignoreErrors
    };
    // TODO: don't wrap here for -t, but save results in Native.MFD
    const pc_result = PreCompute.preCompute(2, options.max_sources, pc_args);
    if (pc_result) {
        Native.showComponents(nameList);
        /*
          const result = getCountListArrays(nameList, pcResult, options);
          showCountLists(nameList, result, options);
        */
    } else {
        console.error(`Precompute failed.`);
    }
    return 0;
};

//////// all below probably should be removed.

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
