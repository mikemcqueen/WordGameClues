//
// show-components.js
//
// "-t" support.
//
// Lots of dead unused code in here, generally a mess.
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Assert      = require('assert');
const Debug       = require('debug')('show-components');
const Expect      = require('should/as-function');
const Path        = require('path');
const Readlines   = require('n-readlines');
const stringify   = require('javascript-stringify').stringify;
const Stringify2  = require('stringify-object');
const Timing      = require('debug')('timing');

const Peco        = require('../../modules/peco');

import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import * as ClueManager from './clue-manager';
import * as NameCount from '../types/name-count';
import * as PreCompute from './cm-precompute';
import * as Source from './source';

///////////

function Stringify (val) {
    return stringify(val, (value, indent, stringify) => {
        if (typeof value == 'function') return "function";
        return stringify(value);
    }, " ");
}

const getCountListArrays = (nameList: string[], pcResult: PreCompute.Result,
    options: any): any =>
{
    let addRemoveSet;
    if (options.add || options.remove) {
        addRemoveSet = new Set();
    }
    let valid: any[] = [];
    let known: any[] = [];
    let clues: any[] = [];
    let invalid: any[] = [];
    let nameListStr: string = nameList.toString();
    let hash = {};

    for (const xorSource of pcResult.data!.xor) {
        // TODO broken
        const countList: number[] = []; // NameCount.listToCountList(xorSource.ncList);
        // for --verbose, we could allow this:
        const hashKey = countList.toString();
        if (hash[hashKey]) continue;
        hash[hashKey] = true;
        let ncListStr = countList.map((count, index) => NameCount.makeNew(nameList[index], count)).toString();
        // TODO: in order to support this, we'd need to pass a flag to PreCompute to
        // tell it to preserve the filtered incompatible combinations, or manually
        // walk through all ClueManager.knownSourceMaps looking for a sourceCsv combo,
        // and displaying those that *aren't* in the xor list. the latter should be done
        // in a separate loop probably, not in this loop.
        /*
        if (!result.success) {
            //console.log(`invalid: ${nameList}  CL ${clueCountList}  x ${x} sum ${sum}  validateAll=${validateAll}`);
            invalid.push(clueCountList);
        } else
        */
        const sum = countList.reduce((a, b) => a + b);
        if (nameList.length === 1) {
            const name = nameList[0];
            let srcList: string[];
            // this is a bit awkward. I didn't want to write the code to handle
            // candidate clue lookup for ClueList(1) so I hacked it to look
            // at primaryNameSrcList.
            if (sum > 1) {
                srcList = ClueManager.getClueList(sum)
                    .filter(clue => clue.name === name)
                    .map(clue => clue.src);
            } else {
                 srcList = xorSource.primaryNameSrcList
                     .filter(nameSrc => nameSrc.name === name)
                     .map(nameSrc => `${nameSrc.count}`);
            }
            if (srcList.length) {
                clues.push({ countList, nameList: srcList });
            } else {
                console.log('well, nothing');
            }
        } else {
            const sourceMap = ClueManager.getKnownSourceMap(sum);
            let is_known = false;
            if (!sourceMap) {
                console.error(`!sourceMap(${sum}), nameList: ${nameListStr}`);
                //  + ` xorSource.ncList: ${NameCount.listToString(xorSource.ncList)}`);
            } else {
                let sourceData = sourceMap[nameListStr];
                if (sourceData) {
                    known.push({
                        countList,
                        nameList: (sourceData.clues as ClueList.Compound).map(clue => clue.name)
                    });
                    is_known = true;
                }
            }
            if (!is_known) {
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
    const srcMap = ClueManager.getKnownSourceMap(count);
    const results = srcMap[source] ? srcMap[source].results : undefined; // TODO: ?.results
    if (!results) {
        let sourceList = source.split(',');
        let s = '';
        sourceList.forEach((source, index) => {
            // [source] is wrong at least for primary clue case, need actual list of sources.
            s += getClueSources([source]);
            console.error(s);
        });
        console.error('---');
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

const show = (options: any): any => {
    Expect(options).is.an.Object();
    Expect(options.test).is.a.String();
    if (options.reject) {
        Expect(options.add).is.undefined();
    }

    // TODO: move buildAllUseNcDataLists to clue-manager?  currently in combo-maker

    if (options.add) {
        options.addMaxSum = ClueManager.getNumPrimarySources();
    }
    if (options.remove) {
        options.removeMinSum = 0;
    }
    options.fast = true; // force fast
    console.log(`test: ${options.test}, fast=${options.fast}`);

    const nameList = options.test.split(',').sort();
    if (options.fast) {
        // TODO: maybe all of this belongs in ClueManager. Because getCountListArrays()
        //       is called from so many places.
        const args = {
            xor: nameList,
            xor_wrap: true, // wrap xorSources on return from c++ plugin
            max: 2,
            quiet: options.quiet,
            ignoreErrors: options.ignoreErrors
        };
        const pcResult = PreCompute.preCompute(2, ClueManager.getNumPrimarySources(), args);
        const result = getCountListArrays(nameList, pcResult, options);
        showCountLists(nameList, result, options);
        // TODO: return something for valid_combos()
        //process.exit(0);
        return 0;
    } else {
        return slow_show(nameList, options);
    }
};

const slow_show = (nameList: string[], options: any) => {
    const nameCsv = nameList.toString();
    const result = ClueManager.getCountListArrays(nameCsv, options);
    if (!result) {
        console.log('No matches');
        return null;
    }
    return showCountLists(nameList, result, options);
};

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
    //console.log(`${Stringify(listOfListOfPrimaryNameSrcLists)}`);
    const listArray = listOfListOfPrimaryNameSrcLists.map(listOfNameSrcLists =>
        [...Array(listOfNameSrcLists.length).keys()]); // 0..nameSrcList.length
    //console.log(`++++ ListArray\n${Stringify(listArray)}\n---- ListArray`);
    let comboLists = Peco.makeNew({
        listArray,
        max: listOfListOfPrimaryNameSrcLists.reduce((sum, listOfNameSrcLists) =>
            sum + listOfNameSrcLists.length, 0)
    }).getCombinations();

    //console.log(`${Stringify(comboLists)}`);

    for (const comboList of comboLists) {
        const nameSrcList = comboList.reduce((nameSrcList, comboListValue, comboListIndex) => {
            let nsList = listOfListOfPrimaryNameSrcLists[comboListIndex][comboListValue];
            //console.log(`nameSrcList: ${nameSrcList}, clValue ${comboListValue}, clIndex ${comboListIndex}, nsList: ${nsList}`);
            if (!nsList || !_.isArray(nsList)) {
                console.log(`nsList: ${nsList}, value ${comboListValue} index ${comboListIndex} lolPnsl(${comboListIndex}):` +
                            ` ${Stringify(listOfListOfPrimaryNameSrcLists[comboListIndex])} nameSrcList ${Stringify(nameSrcList)}`);
                console.log(`lolopnsl: ${Stringify(listOfListOfPrimaryNameSrcLists)}`);
            }
            nameSrcList.push(...nsList);
            return nameSrcList;
        }, []);
        const uniqNameSrcList = _.uniqBy(nameSrcList, NameCount.count);
        //console.log(`nameSrcList ${Stringify(nameSrcList)} len ${nameSrcList.length} uniqLen ${uniqNameSrcList.length}`);
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

// I don't know what the hell is going on here.
// I tried to get --or work with -t or something?
// And it made me jump through some extra hoops here?
// The code doesn't appear to be working.
function flunky (nameList, /*allOrNcDataList,*/ options) {
    const validResults = {};
    let clueNameList = [...get_clue_names(options), ...options.or];
    const min = 2;
    let max = clueNameList.length;
    const indexList = [...Array(max).keys()]; // 0..max (-1?)
    console.log(`maxArg: ${options.maxArg}`);
    let maxIter = options.maxArg ? options.maxArg : max;
    for (let count = maxIter; count >= min; --count) {
        Timing(`wrapper count(${count})`);
        const listArray = [...Array(count).keys()].map(_ => indexList);
        let peco = Peco.makeNew({
            listArray,
            max: count * max
        });
        let comboCount = 0;
        let comboList;
        for (/*let */comboList = peco.firstCombination(); !_.isNull(comboList); comboList = peco.nextCombination()) {
            comboCount++;
            //console.log(`comboList: ${comboList}`); //***logging
            //if (1) continue;
            //if (_.uniq(comboList).length !== comboList.length) continue;
            //console.log(comboList);
            let subList = buildSubListFromIndexList(clueNameList, comboList);
            let comboNameList = [...nameList, ...subList];
            //console.log(`comboNameList: ${comboNameList}`);
            let validResultList = fast_combos_list(comboNameList, Object.assign(
                _.clone(options), { quiet: true, skip_invalid: true }));
            addValidResults(validResults, validResultList, { slice_index: nameList.length });
        }
        Timing(`comboCount: ${comboCount}`);
        
        // Bigger idea: optional arg(s) to -t[COUNTLO[,COUNTHI]]
        // build list of input clues from all clues of those counts
    }
    display_valid_results(validResults);
}

function fast_combo_wrapper (nameList, /*allOrNcDataList,*/ options) {
    console.error('fast_combo_wrapper');
    console.error(`--or: ${Stringify(options.or)}`);
    if (options.or) {
        flunky(nameList, options);
    } else {
        let counts = fast_combos(nameList, options);
        addOrRemove ({
            add:      options.add,
            remove:   options.remove,
            property: options.property
        }, nameList, counts, options);
    }
}

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

function fast_combos (nameList, options) {
    let counts = new Set();
    fast_combos_list(nameList, options)
        .forEach(result => {
            let message = 'invalid';
            if (result.valid) {
                message = `VALID (${result.sum}) ${result.compatibleNameSrcList} `
                    + `REMAIN(${result.inversePrimarySources.length}): ${result.inversePrimarySources}`;
                counts.add(result.sum);
            }
            console.log(`${result.ncList} ${message}`);
        });
    return counts;
}

function fast_combos_list (nameList, options) {
    const ncLists = ClueManager.buildNcListsFromNameList(nameList);
    if (_.isEmpty(ncLists)) {
        if (!options.quiet) {
            console.log(`No ncLists for ${nameList}`);
        }
        //console.log(`nameList: ${nameList} - EMPTY`);
        return [];
    }
    //console.log(`nameList: ${nameList}`);
    //ncLists.forEach(ncList => console.log(`  ncList: ${ncList}`));
    const lists = ClueManager.buildListsOfPrimaryNameSrcLists(ncLists);
    return lists.reduce((resultList, listOfListOfPrimaryNameSrcLists, index) => {
        let add = false;
        let result: any = {};
        result.compatibleNameSrcList = getCompatiblePrimaryNameSrcList(listOfListOfPrimaryNameSrcLists);
        result.valid = Boolean(result.compatibleNameSrcList);
        result.ncList = ncLists[index];
        if (result.valid) {
            result.sum = ncLists[index].reduce((sum, nc) => sum + nc.count, 0);
            result.inversePrimarySources = ClueManager.getInversePrimarySources(result.compatibleNameSrcList.map(ns => `${ns.count}`));
            add = true;
        } else if (!options.skip_invalid) {
            add = true;
        }
        if (add) resultList.push(result);
        //console.log(`${result.ncList} : ${result.valid ? 'VALID' : 'invalid'}`);
        return resultList;
    }, []);
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

//

function validate (filename, options: any = {}) {
    const lines = readlines(filename);
    if (options.combos) {
        validate_combos(lines, options);
    } else {
        validate_sources(lines, options);
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
        const result = ClueManager.getCountListArrays(line, options);
        if (!result || !result.valid) {
            console.log(`${line} ${!result ? 'doesnt exist??' : result.invalid ? 'invalid' : 'rejected'}`);
        }
    }
}

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

//

module.exports = {
    show,
    validate
};
