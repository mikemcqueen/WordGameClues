//
// show-components.js
//

'use strict';

const _           = require('lodash');

const ClueManager = require('../dist/modules/clue-manager');
const NameCount   = require('../dist/types/name-count');
const Validator   = require('../dist/modules/validator');

const Debug       = require('debug')('show-components');
const Expect      = require('should/as-function');
const Path        = require('path');
const Peco        = require('./peco');
const Readlines   = require('n-readlines');
const stringify   = require('javascript-stringify').stringify;
const Stringify2   = require('stringify-object');
const Timing      = require('debug')('timing');

function Stringify (val) {
    return stringify(val, (value, indent, stringify) => {
	if (typeof value == 'function') return "function";
	return stringify(value);
    }, " ");
}

function getSourceClues (source, countList, nameList) {
    source = source.split(',').sort().toString();
    const count = countList.reduce((sum, count) => sum + count, 0);
    // for each name in namelist, get propertyCount(s) of knownSrcMap[count][param]
    const srcMap = ClueManager.knownSourceMapArray[count];
    const results = srcMap[source].results;
    console.error(`results ${Stringify(results)} len(${results.length})`);
    const synCounts = results.map(result => ClueManager.recursiveGetCluePropertyCount(result.resultMap.internal_map, "synonym"));
    return `${nameList.join(' - ')}, syn(${synCounts})`;
}

let GCS = 0;
function getClueSources (name, countList, nameList) {
    let sources = '';
    const count = countList[0];
    const srcMap = ClueManager.knownSourceMapArray[count];
    if (count === 1) {
        const source = nameList[0];
        const clue = _.find(ClueManager.getClueList(count), { name, src: source });
        //console.log(clue);
        const synCount = clue.synonym ? 1 : 0;
        sources += `${source} syn(${synCount})`;
    } else {
        sources += nameList.map(source => {
            const results = srcMap[source].results;
            if (GCS) {
                if (results.length > 1) {
                    for (const result of results) {
                        console.log(Stringify(result.resultMap.internal_map));
                    }
                }
                GCS = 0;
            }
            const synCounts = results.map(result => ClueManager.recursiveGetCluePropertyCount(result.resultMap.internal_map, "synonym"));
            source += ` syn(${synCounts})`;
            return source;
        }).join(' - ');
    }
    return sources;
}

//
function showCountListArray (param, countListArray, text, hasNameList = false) {
    for (const elem of countListArray) {
        const countList = hasNameList ? elem.countList : elem;
        let sources = '';
        if (param) {
            if (countList.length > 1) {
                // -t xxx,yyy[,zzz] (more than 1 word)
                sources += getSourceClues(param, countList, elem.nameList);
            }
            else {
                // -t xxx (1 word only)
                sources += getClueSources(param, countList, elem.nameList);
            }
        }
        console.log(`${countList} ${text} ${sources}`);
    }
}

//
//

function show (options) {
    Expect(options).is.an.Object();
    Expect(options.test).is.a.String();
    if (options.reject) {
        Expect(options.add).is.undefined();
    }

    //
    // TODO: call ClueManager.getCountLists
    //

    // TODO: move buildAllUseNcDataLists to clue-manager?  currently in combo-maker

    // why not .or?
    if (!_.isEmpty(options.or)) {
	options.fast = true; // force fast
    }

    console.log(`test: ${options.test}`);
    const nameList = options.test.split(',').sort();
    if (nameList.length > 1 && options.fast) {
	return fast_combo_wrapper(nameList, options);
    }
    const result = ClueManager.getCountListArrays(options.test, options);
    if (!result) {
        console.log('No matches');
        return null;
    }

    /*
    nameList.forEach(name => {
        console.log('name: ' + name);
    });

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    let countListArray = ClueManager.getKnownClueIndexLists(nameList);
    Debug(countListArray);
    let resultList = Peco.makeNew({
        listArray: countListArray,
        max:       ClueManager.maxClues
    }).getCombinations();
    if (_.isEmpty(resultList)) {
        console.log('No matches');
        return;
    }

    let addCountSet = new Set();
    let isKnown = false;
    let isReject = false;
    for (const clueCountList of resultList) {
        const sum = clueCountList.reduce((a, b) => a + b);
        const result = Validator.validateSources({
            sum:         sum,
            nameList:    nameList,
            count:       nameList.length,
            validateAll: true
        });
        
        //console.log('validate [' + nameList + ']: ' + result);
        let msg = clueCountList.toString();
        if (!result.success) {
            msg += ': INVALID';
        } else if (ClueManager.isRejectSource(nameList)) {
            msg += ': REJECTED';
            isReject = true;
        } else if (nameList.length === 1) {
            let name = nameList[0];
            let nameSrcList = ClueManager.clueListArray[sum]
                    .filter(clue => clue.name === name)
                    .map(clue => clue.src);
            
            if (nameSrcList.length > 0) {
                //let clueNameList = ClueManager.clueListArray[sum].map(clue => clue.name);
                //if (clueNameList.includes(name)) {
                //
                //ClueManager.clueListArray[sum].forEach(clue => {
                //if (clue.name === name) {
                //clueSrcList.push(`"${clue.src}"`);
                //}
                //});

                msg += ': PRESENT as clue with sources: ' + nameSrcList.join(' - ');
            }
        } else {
            let clueList = ClueManager.knownSourceMapArray[sum][nameList];
            if (clueList) {
                msg += ': PRESENT as ' + clueList.map(clue => clue.name);
                isKnown = true;
            }
            if (options.add || options.remove) {
                addCountSet.add(sum);
            }
        }
        console.log(msg);
    }
    */

    showCountListArray(null, result.rejects, 'REJECTED');
    showCountListArray(null, result.invalid, 'INVALID');
    showCountListArray(options.test, result.known, 'PRESENT as', true);
    showCountListArray(options.test, result.clues, 'PRESENT as clue with source:', true);
    showCountListArray(null, result.valid, 'VALID');

    const save = _.isUndefined(options.save) ? true : options.save;
    const count = ClueManager.addRemoveOrReject({
        add:      options.add,
        remove:   options.remove,
        reject:   options.reject,
        isKnown:  !_.isEmpty(result.known),
        isReject: !_.isEmpty(result.reject)
    }, nameList, result.addRemoveSet, { save });
    if (options.add || options.remove) {
        console.log(`${options.add ? "added" : "removed"} ${count} clues`);
    }
    return Object.assign(result, { added: count });
}

//

function getCompatiblePrimaryNameSrcList (listOfListOfPrimaryNameSrcLists) {
    //console.log(`${Stringify(listOfListOfPrimaryNameSrcLists)}`);
    const listArray = listOfListOfPrimaryNameSrcLists.map(listOfNameSrcLists => [...Array(listOfNameSrcLists.length).keys()]); // 0..nameSrcList.length
    //console.log(`++++ ListArray\n${Stringify(listArray)}\n---- ListArray`);
    let comboLists = Peco.makeNew({
        listArray,
        max: listOfListOfPrimaryNameSrcLists.reduce((sum, listOfNameSrcLists) => sum + listOfNameSrcLists.length, 0)
    }).getCombinations();

    //console.log(`${Stringify(comboLists)}`);

    for (const comboList of comboLists) {
	const nameSrcList = comboList.reduce((nameSrcList, comboListValue, comboListIndex) => {
	    let nsList = listOfListOfPrimaryNameSrcLists[comboListIndex][comboListValue];
	    //console.log(`nameSrcList: ${nameSrcList}, clValue ${comboListValue}, clIndex ${comboListIndex}, nsList: ${nsList}`);
	    if (!nsList || !_.isArray(nsList)) {
		console.log(`nsList: ${nsList}, value ${comboListValue} index ${comboListIndex} lolPnsl(${comboListIndex}): ${Stringify(listOfListOfPrimaryNameSrcLists[comboListIndex])} nameSrcList ${Stringify(nameSrcList)}`);
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

function buildSubListFromIndexList (nameList, indexList) {
    const subList = [];
    indexList.forEach(index => subList.push(nameList[index]));
    return subList;
}

function fast_combo_wrapper (nameList, /*allOrNcDataList,*/ options) {
    console.log('fast_combo_wrapper');
    console.log(`--or: ${Stringify(options.or)}`);
    if (options.or) {
	const validResults = {};
	let clueNameList = [...get_clue_names(options), ...options.or];
	const min = 2;
	let max = clueNameList.length;
	const indexList = [...Array(max).keys()]; // 0..max
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
		let validResultList = fast_combos_list(comboNameList, Object.assign(_.clone(options), { quiet: true, skip_invalid: true }));
		addValidResults(validResults, validResultList, { slice_index: nameList.length });
	    }
	    Timing(`comboCount: ${comboCount}`);

	    // Bigger idea: optional arg(s) to -t[COUNTLO[,COUNTHI]]
	    // build list of input clues from all clues of those counts
	}
	display_valid_results(validResults);
    } else {
	return fast_combos(nameList, options);
    }
}

function get_clue_names (options) {
    let result = [];
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
    fast_combos_list(nameList, options)
	.forEach(result => {
	    let message = 'invalid';
	    if (result.valid) {
		message = `VALID (${result.sum}) ${result.compatibleNameSrcList} `
                    + `REMAIN(${result.inversePrimarySources.length}): ${result.inversePrimarySources}`;
	    }
	    console.log(`${result.ncList} ${message}`);
	});
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

    //console.log(`1`);
    const lists = ClueManager.buildListsOfPrimaryNameSrcLists(ncLists);
    //console.log(`2`);
    //console.log(`len: ${lists.length}`);
    return lists.reduce((resultList, listOfListOfPrimaryNameSrcLists, index) => {
	let add = false;
	let result = {};
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
    console.log('++display');
    Object.keys(validResults).forEach(key => console.log(key));
    console.log('--display');
}

function showNcLists (ncLists) {
    for (let ncList of ncLists) {
	console.log(`${ncList}`);
    }
}

function old_fast_combos (name_list, options) {
    console.log('old_fast_combos');
/*
    let list_array = [];
    const one_max = name_list.length - 1; // not perfect, only works for combos of 2
    for (let one = 0; one < one_max; one += 1) {
	for (let two = one + 1; two < name_list.length; two += 1) {
	    list_array.push([ one, two ]);
	}
    }
    console.log(`list_array: ${Stringify(list_array)}`);
    for (const index_list of list_array) {
	const combo_list = [ name_list[index_list[0]], name_list[index_list[1]] ];
*/
    const combo_list = [ name_list[0], name_list[1] ];
    const test  = combo_list.join(',');
    const addRemove = combo_list.join(' ');
    const save = false;
    
    let result = show({ test, save, add: addRemove });
	
    if (!result || !(result.valid.length + result.known.length)) {
        console.log('No fast matches');
	return null;
    }

    let fastresult = show({
	//test: `${addRemove},${name_list[3 - index_list[0] - index_list[1]]}`,
	test: `${addRemove},${name_list[2]}`,
	save
    });
    
    if (result.added) {
	show({ test, save, remove: addRemove });
    }
    return fastresult;
}

//

function validate (filename, options = {}) {
    const lines = readlines(filename);
    if (options.combos) {
	validate_combos(lines, options);
    } else {
	validate_sources(lines, options);
    }
}

function validate_combos(lines, options) {
    const combos = [];
    let input = lines;
    for (;;) {
	Timing(``);
	const raw_combo_list = all_combos(input, lines);
	Timing(`all_combos (${raw_combo_list.length})`);
	if (_.isEmpty(raw_combo_list)) break;
	Debug(`raw: ${typeof(raw_combo_list)}, len: ${raw_combo_list.length}: ${raw_combo_list}`);
	let valid_combo_list = [];
	valid_combo_list = valid_combos(raw_combo_list);
	if (_.isEmpty(valid_combo_list)) break; 
	Timing(`valid combos (${valid_combo_list.length})`);
	combos.push(...valid_combo_list);
	input = valid_combo_list;
    }
    combos.forEach(combo => {
	console.log(`${combo}`);
    });
}

function valid_combos(combo_list, options = {}) {
    const combos = [];
    options.any = true;

    combo_list.forEach(combo_str => {
	//onst combo_str = _.join(combo, ',');
	//Debug(`${typeof(combo)}: ${combo}`);
	Debug(`${typeof(combo_str)}: ${combo_str} (${combo_str.split(',').length})`);
	if (combo_str.split(',').length > 3) return; // continue
//	const result = ClueManager.getCountListArrays(combo_str, options);
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

function all_combos(input_list, word_list) {
    const combos =u< [];
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

function readlines(filename) {
    const path = Path.normalize(`${Path.dirname(module.filename)}/tools/${filename}`);
    const readLines = new Readlines(filename);
    let lines = [];
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
