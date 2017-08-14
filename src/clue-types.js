//
// clue-types.js
//

'use strict';

//

const _              = require('lodash');
const expect         = require('chai').expect;

const Options = [
    ['p', 'poem=NUM',   'use poem clues, sentence NUM' ],
    ['m', 'meta',       'use metamorphosis clues' ],
    ['y', 'synthesis',  'use synthesis clues' ],
    ['r', 'harmony',    'use harmony clues' ],
    ['f', 'final',      'use final clues' ]
];

const POEM = {
    "1" : {
	sentence:       1,
	baseDir:        'poem/1',
	resultDir:      'poem',
	clueCount:      12,
	REQ_CLUE_COUNT: 12
    },

    "2": {
	sentence:       2,
	baseDir:        'poem/2',
	resultDir:      'poem',
	clueCount:      9,
	REQ_CLUE_COUNT: 9
    },

    "3": {
	sentence:       3,
	baseDir:        'poem/3',
	resultDir:      'poem',
	clueCount:      4,
	synthClueCount: 9,
	REQ_CLUE_COUNT: 4
    },

    "5": {
	sentence:       5,
	baseDir:        'poem/5',
	resultDir:      'poem',
	clueCount:      15,
	REQ_CLUE_COUNT: 15
    },

    "6": {
	sentence:       6,
	baseDir:        'poem/6',
	resultDir:      'poem',
	clueCount:      8,
	REQ_CLUE_COUNT: 8
    }
};

const META = {
    baseDir:        'meta',
    clueCount: 9,
    REQ_CLUE_COUNT: 9
};

const SYNTH = {
    baseDir:        'synth',
    clueCount: 4,
    REQ_CLUE_COUNT: 4
};

const HARMONY = {
    baseDir:        'harmony',
    clueCount: 3,
    REQ_CLUE_COUNT: 3
};

const FINAL = {
    baseDir:        'final',
    clueCount: 2,
    REQ_CLUE_COUNT: 2
};

//

function getByOptions(options) {
    let src;
    if (options.meta) {
	src = META;
    } else if (options.synthesis) {
	src = SYNTH;
    } else if (options.harmony) {
	src = HARMONY;
    } else if (options.final) {
	src = FINAL;
    } else if (!_.isUndefined(options.poem)) {
	src = POEM[options.poem[0]];
	if (_.isUndefined(src)) throw new Error(`POEM[${options.poem}] not supported`);
	if (options.poem.slice(1, options.poem.length) === 's') {
	    src = cloneAsSynth(src);
	}
    } else {
	throw new Error('No clue option supplied');
    }
    return src;
}

//

function cloneAsSynth (config) {
    config = _.clone(config);
    config.baseDir += '/synth';
    config.clueCount = config.synthClueCount || 9; // hax
    config.REQ_CLUE_COUNT = 2;
    return config;
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
    cloneAsSynth,
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

    POEM,
    META,
    SYNTH,
    HARMONY,
    FINAL
};
