//
// CLUE-TYPES.JS
//

'use strict';

// export a singleton

const _              = require('lodash');
const expect         = require('chai').expect;

const Options = [
    ['p', 'poem=NUM',   'use poem clues, sentence NUM' ],
    ['m', 'meta',       'use metamorphosis clues' ],
    ['y', 'synthesis',  'use synthesis clues' ],
    ['r', 'harmony',    'use harmony clues' ],
    ['f', 'final',      'use final clues' ]
];

const POEM_1 = {
    sentence:       1,
    baseDir:        'poem/1',
    resultDir:      'poem',
    MAX_CLUE_COUNT: 12,
    REQ_CLUE_COUNT: 12
};

const POEM_3 = {
    sentence:       3,
    baseDir:        'poem/3',
    resultDir:      'poem',
    MAX_CLUE_COUNT: 4,
    REQ_CLUE_COUNT: 4
};

const META = {
    baseDir:        'meta',
    MAX_CLUE_COUNT: 9,
    REQ_CLUE_COUNT: 9
};

const SYNTH = {
    baseDir:        'synth',
    MAX_CLUE_COUNT: 4,
    REQ_CLUE_COUNT: 4
};

const HARMONY = {
    baseDir:        'harmony',
    MAX_CLUE_COUNT: 3,
    REQ_CLUE_COUNT: 3
};

const FINAL = {
    baseDir:        'final',
    MAX_CLUE_COUNT: 2,
    REQ_CLUE_COUNT: 2
};

//

function getByOptions(options) {
    let src = META;
    if (options.synthesis) {
	src = SYNTH;
    } else if (options.harmony) {
	src = HARMONY;
    } else if (options.final) {
	src = FINAL;
    } else if (!_.isUndefined(options.poem)) {
	switch(_.toNumber(options.poem)) {
	case 1: src = POEM_1; break;
	//case 2: src = POEM_2; break;
	case 3: src = POEM_3; break;
	//case 4: src = POEM_4; break;
	//case 5: src = POEM_5; break;
	//case 6: src = POEM_6; break;
	//case 7: src = POEM_7; break;
	//case 8: src = POEM_8; break;
	//case 9: src = POEM_9; break;
	default:
	    throw Error(`POEM_${src} not supported`);
	}
    }
    return src;
}

const ALL_TYPES = [ META, SYNTH, HARMONY, FINAL ];

// name: 'm' or 'meta', for example, used by tools/merge.js
// and frankly should probably only exist in merge.js
//
function getByBaseDirOption (name) {
    for (const type of ALL_TYPES) { 
	if (name === type.baseDir || name === type.baseDir.charAt(0)) {
	    return type;
	}
    }
    throw new Error(`invalid type name, ${name}`);
}

//
	    
function isValidBaseDirOption (name) {
    try {
	getBaseDirOption(name);
	return true;
    } catch (err) {
	return false;
    }
}

//
/*
function isClueType (name, type) {
    return getFullName(name) === type.name;
}

function isMeta (name) {
    return isClueType(name, META);
}

function isSynth (name) {
    return isClueType(name, SYNTH);
}

function isHarmony (name) {
    return isClueType(name, HARMONY);
}

function isFinal (name) {
    return isClueType(name, FINAL);
}
*/

module.exports = {
    getByBaseDirOption,
    getByOptions,

  /*
   isMeta,
   isSynth,
   isHarmony,
   isFinal,
   */

    isValidBaseDirOption,
    Options,

    POEM_1,
    POEM_3,
    META,
    SYNTH,
    HARMONY,
    FINAL
};
