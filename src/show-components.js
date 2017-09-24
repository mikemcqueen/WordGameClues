//
// show-components.js
//

'use strict';

let _           = require('lodash');
let ClueManager = require('./clue-manager');
let Debug       = require('debug')('show-components');
let Expect      = require('should/as-function');
let Peco        = require('./peco');
let Validator   = require('./validator');

//
//

function show (options) {
    Expect(options).is.an.Object();
    Expect(options.test).is.a.String();
    if (options.reject) {
	Expect(options.add).is.undefined();
    }

    let nameList = options.test.split(',').sort();
    nameList.forEach(name => {
	console.log('name: ' + name);
    });

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    let countListArray = ClueManager.getKnownClueIndexLists(nameList);
    console.log(countListArray);

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
		
		/*
		 ClueManager.clueListArray[sum].forEach(clue => {
		 if (clue.name === name) {
		 clueSrcList.push(`"${clue.src}"`);
		 }
		 });
		 */
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
    ClueManager.addRemoveOrReject({
	add:    options.add,
	remove: options.remove,
	reject: options.reject,
	isKnown,
	isReject
    }, nameList, addCountSet);
}

//

module.exports = {
    show
};
