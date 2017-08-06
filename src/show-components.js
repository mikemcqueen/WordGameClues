//
// show-components.js
//

'use strict';

let _           = require('lodash');
let ClueManager = require('./clue-manager');
let Expect      = require('should'); // require('chai').expect;
let Peco        = require('./peco');
let Validator   = require('./validator');

//

function addClues (countSet, name, src) {
    Expect(countSet).is.instanceof(Set);
    Expect(name).is.a.String();
    Expect(src).is.a.String();
    countSet.forEach(count => {
	if (ClueManager.addClue(count, {
	    name: name,
	    src:  src
	}, true, true)) { // save, nothrow
	    console.log(`${count}: updated`);
	} else {
	    console.log(`${count}: already present`);
	}
    });
}

//

function addReject (nameList) {
    if (ClueManager.addReject(nameList, true)) {
	console.log('updated');
    }
    else {
	console.log('update failed');
    }
}


// each count list contains the clueMapArray indexes in which
// each name appears
// this could live in ClueManager.
function getKnownClueIndexLists (nameList) {
    let countListArray = Array(_.size(nameList)).fill().map(() => []);
    //console.log(countListArray);
    for (let count = 1; count <= ClueManager.maxClues; ++count) {
	const map = ClueManager.knownClueMapArray[count];
	if (!_.isUndefined(map)) {
	    nameList.forEach((name, index) => {
		if (_.has(map, name)) {
		    countListArray[index].push(count);
		}
	    });
	}
	else {
	    console.log('missing known cluemap #' + count);
	}
    }
    // verify that all names were found
    nameList.forEach((name, index) => {
	Expect(countListArray[index], `cannot find clue, ${name}`).exists;
    });
    return countListArray;
}

//
//

function show (options) {
    Expect(options, 'options').is.an.Object();
    Expect(options.test, 'options.test').is.a.String();
    if (options.reject) {
	Expect(options.add, 'cannot specify both --add and --reject').is.undefined;
    }

    let nameList = options.test.split(',').sort();
    nameList.forEach(name => {
	console.log('name: ' + name);
    });

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    let countListArray = getKnownClueIndexLists(nameList);
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
    let known = false;
    let reject = false;
    resultList.forEach(clueCountList => {
	let sum = clueCountList.reduce((a, b) => a + b);
	let result = Validator.validateSources({
	    sum:      sum,
	    nameList: nameList,
	    count:    nameList.length,
	    validateAll: true
	});
	//console.log('validate [' + nameList + ']: ' + result);
	let msg = clueCountList.toString();
	if (!result.success) {
	    msg += ': INVALID';
	} else if (ClueManager.isRejectSource(nameList)) {
	    msg += ': REJECTED';
	    reject = true;
	} else {
	    if (nameList.length === 1) {
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
		if (!_.isUndefined(clueList)) {
		    msg += ': PRESENT as ' + clueList.map(clue => clue.name);
		    known = true;
		}
		if (options.add) {
		    addCountSet.add(sum);
		}
	    }
	}
	console.log(msg);
    });

    if (!_.isUndefined(options.add)) {
	if (nameList.length === 1) {
	    console.log('WARNING! ignoring --add due to single source');
	} else if (reject) {
	    console.log('WARNING! cannot add known clue: already rejected, ' + nameList);
	} else {
	    addClues(addCountSet, options.add, nameList.toString());
	}
    } else if (options.reject) {
	if (nameList.length === 1) {
	    console.log('WARNING! ignoring --reject due to single source');
	} else if (known) {
	    console.log('WARNING! cannot add reject clue: already known, ' + nameList);
	} else {
	    addReject(nameList.toString());
	}
    }
}

//

module.exports = {
    show
};
