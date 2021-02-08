//
// show-components.js
//

'use strict';

const _           = require('lodash');
const ClueManager = require('./clue-manager');
const Debug       = require('debug')('show-components');
const Expect      = require('should/as-function');
const NameCount   = require('../types/name-count');
const Path        = require('path');
const Peco        = require('./peco');
const Readlines   = require('n-readlines');
const Stringify   = require('stringify-object');
const Timing      = require('debug')('timing');
const Validator   = require('./validator');

//

function showCountListArray (countListArray, text, hasNameList = false) {
    for (const elem of countListArray) {
        const countList = hasNameList ? elem.countList : elem;
        console.log(`${countList} ${text} ${hasNameList ? elem.nameList.join(' - ') : ''}`);
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

//    let allOrNcDataLists = options.or ? buildAllUseNcDataLists(args.or) : [];
    if (!_.isEmpty(options.or)) {
	options.fast = true; // force fast
    }

    console.log(`test: ${options.test}`);
    const nameList = options.test.split(',').sort();
    if (nameList.length > 1 && options.fast) {
	return fast_combo_wrapper(nameList, /*allOrNcDataLists,*/ options);
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

    showCountListArray(result.rejects, 'REJECTED');
    showCountListArray(result.invalid, 'INVALID');
    showCountListArray(result.known, 'PRESENT as', true);
    showCountListArray(result.clues, 'PRESENT as clue with source:', true);
    showCountListArray(result.valid, 'VALID');

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
	const max_results = 10;
	const min = 2;
	const max = options.or.length;
	console.log(`min(${min}) max(${max}) results(${max_results})`);

	const indexList = [...Array(options.or.length).keys()]; // 0..max
	for (let count = max; count >= min; count -= 1) {
	    const listArray = [...Array(count).keys()].map(_ => indexList);
	    console.log(`count(${count}):\n${Stringify(listArray)}`);
	    Peco.makeNew({
		listArray,
		max: count * max
	    }).getCombinations()
		.filter(indexList => _.uniq(indexList).length === indexList.length)
		.forEach(indexList => {
		    let orNameList = buildSubListFromIndexList(options.or, indexList);
		    orNameList = _.concat(nameList, orNameList);
		    //console.log(`orNameLists: ${orNameList}`);
		    
		    // TODO: here, i probably just want to build an array of results data
		    // that i can sort (by total clue count, for example)
		    // and filter (by valid/invalid)
		    // and display, independently of the loggin in fast_combos.
		    return fast_combos(orNameList, Object.assign(_.clone(options), { quiet: true }));
		});
	}
    } else {
	// TODO: will need to add a display call here as well
	return fast_combos(nameList, options);
    }
}

function fast_combos (nameList, options) {
    const ncLists = ClueManager.buildNcListsFromNameList(nameList);
    if (_.isEmpty(ncLists)) {
	if (!options.quiet) {
	    console.log(`No ncLists for ${nameList}`);
	}
	return;
    }
    const lists = ClueManager.buildListsOfPrimaryNameSrcLists(ncLists);
    console.log(`len: ${lists.length}`);
    lists.forEach ((listOfListOfPrimaryNameSrcLists, index) => {
	const compatibleNameSrcList = getCompatiblePrimaryNameSrcList(listOfListOfPrimaryNameSrcLists);
	let sum = 0;
	let inversePrimarySources;
	if (compatibleNameSrcList) {
	    sum = ncLists[index].reduce((sum, nc) => sum + nc.count, 0);
	    inversePrimarySources = ClueManager.getInversePrimarySources(compatibleNameSrcList.map(ns => `${ns.count}`));
	}
	console.log(`${ncLists[index]} ` + (compatibleNameSrcList ? `VALID (${sum}): ${compatibleNameSrcList} `
            + `REMAIN(${inversePrimarySources.length}): ${inversePrimarySources}` : 'invalid'));
    });
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
    const combos = [];
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
//    countListArray: showCountListArray,
    show,
    validate
};
