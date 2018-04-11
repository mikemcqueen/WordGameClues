//
// clue-types.js
//

//'use strict'; // because arguments.callee

//

const _              = require('lodash');
const Expect         = require('should/as-function');
const Path           = require('path');
const Stringify      = require('stringify-object');

const Options = [
    ['p', 'apple=NUM',  'use apple clues, sentence NUM' ],
    ['m', 'meta',       'use metamorphosis clues' ],
    ['y', 'synthesis',  'use synthesis clues' ],
    ['r', 'harmony',    'use harmony clues' ],
    ['f', 'final',      'use final clues' ]
];

const DATA_DIR              =  Path.normalize(`${Path.dirname(module.filename)}/../../data/`);

const APPLE = {
    '1' : {
        sentence:       1,
        clueCount:      12,
        REQ_CLUE_COUNT: 12
    },

    '1.1' : {
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
        sentence:         3,
        clueCount:        4,
        REQ_CLUE_COUNT:   4
    },

    '4': {
        sentence:         4,
        clueCount:        13,
        REQ_CLUE_COUNT:   13
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
    },

    '7': {
        sentence:       7,
        clueCount:      7,
        REQ_CLUE_COUNT: 7
    },

    '8': {
        sentence:       8,
        clueCount:      8,
        REQ_CLUE_COUNT: 8
    },

    '8.1': {
        sentence:       8,
        clueCount:      9,
        REQ_CLUE_COUNT: 8
    },

    '8.2': {
        sentence:       8,
        clueCount:      9,
        REQ_CLUE_COUNT: 8
    },

    '9': {
        sentence:       9,
        clueCount:      8,
        REQ_CLUE_COUNT: 8
    }

};

const META = {
    baseDir: 'meta',
    clueCount: 9,
    REQ_CLUE_COUNT: 9
};

const SYNTH = {
    baseDir: 'synth',
    clueCount: 6,
    REQ_CLUE_COUNT: 6
};

const HARMONY = {
    baseDir: 'harmony',
    clueCount: 3,
    REQ_CLUE_COUNT: 3
};

const FINAL = {
    baseDir: 'final',
    clueCount: 2,
    REQ_CLUE_COUNT: 2
};

//

function metamorph (variety) {
    const name = arguments.callee.name;
    const src = APPLE[variety];
    if (!src.baseDir) {
        let dir = '';
        let index = name.length - 2;
        let next = 0;
        while (index >= 0) {
            dir += name.charAt(index);
            next = next === 0 ? 2 : next === 2 ? 4 : 1; 
            index -= next;
        }
        //src.resultDir = dir;
        src.baseDir = `${dir}/${variety}`;
        src.variety = variety;
    }
    return src;
}

//

function getByOptions (options) {
    let src;
    if (options.meta) {
        src = META;
    } else if (options.synthesis) {
        src = SYNTH;
    } else if (options.harmony) {
        src = HARMONY;
    } else if (options.final) {
        src = FINAL;
    } else if (options.apple) {
        let variety = options.apple[0]; // M
        if (options.apple[1] === '.') variety += options.apple[1]; // M.
        if (_.toNumber(options.apple[2]) > 0) variety += options.apple[2]; // M.X
        if (_.toNumber(options.apple[3]) > 0) variety += options.apple[3]; // M.XY
        src = metamorph(variety);
        if (_.isUndefined(src)) throw new Error(`APPLE[${options.apple}] not supported`);
        if (_.size(options.apple) > _.size(variety)) {
            src = cloneAsType(src, getTypeFromSuffix(options.apple.slice(_.size(variety), _.size(options.apple))));
        }
    } else {
        throw new Error('No clue-type option supplied');
    }
    return src;
}

//

function getTypeFromSuffix (type) {
    switch (type) {
    case 's': return SYNTH;
    case 'h': return HARMONY;
    case 'f': return FINAL;
    default:
        throw new Error(`invalid type suffix, ${type}`);
    }
}

//

function getTypeSuffix (config) {
    const index = _.lastIndexOf(config.baseDir, '/') + 1;
    if (index === 0) return undefined;
    const firstLetter = config.baseDir.slice(index, index + 1);
    const number = _.toNumber(firstLetter);
    return number > 0 ? 'p' : firstLetter;
}

//

function getNextType (config) {
    switch (getTypeSuffix(config)) {
    case 'p': return SYNTH;
    case 's': return HARMONY;
    case 'h': return FINAL;
    default:
        throw new Error(`next type not implemented for: ${Stringify(config)}`);
    }
}

//

function cloneAsNextType (config) {
    return cloneAsType(config, getNextType(config));
}

//

function cloneAsType (config, otherType) {
    config = _.clone(config);
    config.baseDir += '/' + otherType.baseDir;
    config.clueCount = otherType.clueCount;
    config.REQ_CLUE_COUNT = otherType.REQ_CLUE_COUNT;
    config.clone = true;
    return config;
}

const ALL_TYPES = [ META, SYNTH, HARMONY, FINAL ];

// name: 'm' or 'meta', for example, used by tools/merge.js
// and frankly should probably only exist in merge.js
// doesn't work with apples
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

// e.g. p3s, p8.1

function getShorthand (clueType) {
    const dir = clueType.baseDir;
    Expect(dir.charAt(0)).is.equal('p');
    return `${dir.charAt(0)}${clueType.variety}${getTypeSuffix(clueType)}`;
}

// e.g. p3s.c2-6.x2

// TODO: getTypeClueCount: synthcClueCount = getByType(clueType).clueCount;
// TODO: test

function getLonghand (clueType, max = 2) {
    return `${getShorthand(clueType)}.c2-${clueType.synthClueCount}.x${max}`;
}

//

function getDirectory (clueType) {
    return `${DATA_DIR}${clueType.baseDir}`;
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
    cloneAsNextType,
    getByOptions,  
    getByBaseDirOption,
    getDirectory,
    getShorthand,
    getLonghand,
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
