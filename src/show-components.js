//
// show-components.js
//

'use strict';

let _           = require('lodash');
//let AltSources  = require('./alt_sources');
//let ClueList    = require('./clue_list');
let ClueManager = require('./clue_manager');
//let Clues       = require('./clue-types');
//let ComboMaker  = require('./combo_maker');
//let ComboSearch = require('./combo_search');
//let Components  = require('./show-components');
//let Duration    = require('duration');
let Expect      = require('chai').expect;
//let NameCount   = require('./name_count');
let Peco        = require('./peco');
//let PrettyMs    = require('pretty-ms');
//let Show        = require('./show');
//let ResultMap   = require('./resultmap');
let Validator   = require('./validator');


//

function addClues(countSet, name, src) {
    Expect(countSet).to.be.a('Set');
    Expect(name).to.be.a('string');
    Expect(src).to.be.a('string');
    countSet.forEach(count => {
	if (ClueManager.addClue(count, {
	    name: name,
	    src:  src
	}, true)) {
	    console.log('updated ' + count);
	}
	else {
	    console.log('update of ' + count + ' failed.');
	}
    });
}

//

function addReject(nameList) {
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
	Expect(countListArray[index], `cannot find clue, ${name}`).to.exist;
    });
    return countListArray;
}

//
//

function show(options) {
    Expect(options, 'options').to.be.an('object');
    Expect(options.test, 'options.test').to.be.a('string');
    if (options.reject) {
	Expect(options.add, 'cannot specify both --add and --reject').to.be.undefined;
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
