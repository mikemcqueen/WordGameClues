//
// show-components.js
//

'use strict';

const _           = require('lodash');
const ClueManager = require('./clue-manager');
const Debug       = require('debug')('show-components');
const Expect      = require('should/as-function');
const Path        = require('path');
const Peco        = require('./peco');
const Readlines   = require('n-readlines');
const Validator   = require('./validator');

//

function showCountListArray (countListArray, text, hasNameList = false) {
    for (const elem of countListArray) {
	const countList = hasNameList ? elem.countList : elem;
	console.log(`${countList} ${text} ${hasNameList ? elem.nameList : ''}`);
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

    const nameList = options.test.split(',').sort();
    const result = ClueManager.getCountListArrays(options.test, options);
    if (!result) {
	console.log('No matches');
	return;
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
    showCountListArray(result.clues, 'PRESENT as clue with sources:', true);
    showCountListArray(result.valid, 'VALID');

    const count = ClueManager.addRemoveOrReject({
	add:      options.add,
	remove:   options.remove,
	reject:   options.reject,
	isKnown:  !_.isEmpty(result.known),
	isReject: !_.isEmpty(result.reject)
    }, nameList, result.addRemoveSet, { save: true });
    if (options.add || options.remove) {
	console.log(`${options.add ? "added" : "removed"} ${count} clues`);
    }
}

//

function validate(filename, options = {}) {
    const path = Path.normalize(`${Path.dirname(module.filename)}/tools/${filename}`);
    const readLines = new Readlines(filename);
    let line;
    while ((line = readLines.next()) !== false) {
	line = line.toString().trim();
	const result = ClueManager.getCountListArrays(line, options);
	if (!result || !result.valid) {
	    console.log(`${line} ${!result ? 'doesnt exist??' : result.invalid ? 'invalid' : 'rejected'}`);
	}
    }
}

//

module.exports = {
//    countListArray: showCountListArray,
    show,
    validate
};
