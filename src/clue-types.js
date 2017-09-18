//
// clue-types.js
//

//'use strict';

//

const _              = require('lodash');

const Options = [
    ['p', 'apple=NUM',  'use apple clues, sentence NUM' ],
    ['m', 'meta',       'use metamorphosis clues' ],
    ['y', 'synthesis',  'use synthesis clues' ],
    ['r', 'harmony',    'use harmony clues' ],
    ['f', 'final',      'use final clues' ]
];

const APPLE = {
    '1' : {
	sentence:       1,
	clueCount:      12,
	REQ_CLUE_COUNT: 12
    },

    '2': {
	sentence:       2,
	clueCount:      9,
	REQ_CLUE_COUNT: 9
    },

    '3': {
	sentence:       3,
	clueCount:      4,
	synthClueCount: 9,
	REQ_CLUE_COUNT: 4
    },

    '5': {
	sentence:       5,
	clueCount:      15,
	REQ_CLUE_COUNT: 15
    },

    '6': {
	sentence:       6,
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

function metamorph (src) {
    const name = arguments.callee.name;
    if (!src.resultDir) {
	let dir = '';
	let index = name.length - 2;
	let next = 0;
	while (index >= 0) {
	    dir += name.charAt(index);
	    next = next === 0 ? 2 : next === 2 ? 4 : 1; 
	    index -= next;
	}
	src.resultDir = dir;
	src.baseDir = `${dir}/${src.sentence}`;
    }
    return src;
 }

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
    } else if (!_.isUndefined(options.apple)) {
	src = metamorph(APPLE[options.apple[0]]);
	if (_.isUndefined(src)) throw new Error(`APPLE[${options.apple}] not supported`);
	if (options.apple.slice(1, options.apple.length) === 's') {
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
	getByBaseDirOption(name);
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
    isValidBaseDirOption,
    Options

    /*
     isMeta,
     isSynth,
     isHarmony,
     isFinal,
     */
    
    /*
     APPLE,
     META,
     SYNTH,
     HARMONY,
     FINAL
     */
};
