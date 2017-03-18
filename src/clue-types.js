//
// CLUE-TYPES.JS
//

'use strict';

// export a singleton

const _              = require('lodash');
const expect         = require('chai').expect;

// I would put these in ClueManager 'cept it's exported as a singleton.
// I should undo that, export functions & maybe this data.

const META = {
    name:           'meta',
    MAX_CLUE_COUNT: 9,
    REQ_CLUE_COUNT: 9
};

const SYNTH = {
    name:           'synth',
    MAX_CLUE_COUNT: 4,
    REQ_CLUE_COUNT: 4
};

const HARMONY = {
    name:           'harmony',
    MAX_CLUE_COUNT: 3,
    REQ_CLUE_COUNT: 3
};

const FINAL = {
    name:           'final',
    MAX_CLUE_COUNT: 2,
    REQ_CLUE_COUNT: 2
};

const ALL_TYPES = [ META, SYNTH, HARMONY, FINAL ];

// name: 'm' or 'meta', for example
//
function getFullName (name) {
    for (const type of ALL_TYPES) { 
	if (name === type.name || name === type.name.charAt(0)) {
	    return type.name;
	}
    }
    throw new Error(`invalid type name, ${name}`);
}

//
	    
function isValidName (name) {
    try {
	getFullName(name);
	return true;
    } catch (err) {
	return false;
    }
}

//

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


module.exports = {
    getFullName,
    isMeta,
    isSynth,
    isHarmony,
    isFinal,
    isValidName,

    META,
    SYNTH,
    HARMONY,
    FINAL
};
