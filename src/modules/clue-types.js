//
// clue-types.js
//

//'use strict'; // because arguments.callee

//

const _              = require('lodash');
const Debug          = require('debug')('clue-types');
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
        clueCount:      16,
        REQ_CLUE_COUNT: 16
    },

    '1.2' : {
        sentence:       1,
        clueCount:      17,
        REQ_CLUE_COUNT: 17
    },

    '1.98': {
        sentence:       2,
        clueCount:      15,
        REQ_CLUE_COUNT: 15
    },

    '2': {
        sentence:       2,
        clueCount:      9,
        REQ_CLUE_COUNT: 9
    },

    '2.0': {
        comment:        'all clues from 2 with broken down compound words',
        sentence:       2,
        clueCount:      17,
        REQ_CLUE_COUNT: 17
    },

    '2.1': {
        sentence:       2,
        clueCount:      14,
        REQ_CLUE_COUNT: 14
    },

    '2.2': {
        sentence:       2,
        clueCount:      14,
        REQ_CLUE_COUNT: 14
    },

    '2.98': {
        sentence:       2,
        clueCount:      15,
        REQ_CLUE_COUNT: 15
    },

    '2.99': {
        sentence:       2,
        clueCount:      14,
        REQ_CLUE_COUNT: 14
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

    '4.1': {
        sentence:         4,
        clueCount:        13,
        REQ_CLUE_COUNT:   13
    },

    '4.2': {
        sentence:         4,
        clueCount:        17,
        REQ_CLUE_COUNT:   17
    },

    '4.2': {
        sentence:         4,
        clueCount:        17,
        REQ_CLUE_COUNT:   17
    },

    '4.3': {
        sentence:         4,
        clueCount:        17,
        REQ_CLUE_COUNT:   17
    },

    '4.4': {
        sentence:         4,
        clueCount:        17,
        REQ_CLUE_COUNT:   17
    },

    '4.5': {
        sentence:         4,
        clueCount:        21,
        REQ_CLUE_COUNT:   17
    },

    '4.51': {
        sentence:         4,
        clueCount:        21,
        REQ_CLUE_COUNT:   17
    },

    '4.6': {
        sentence:         4,
        clueCount:        20,
        REQ_CLUE_COUNT:   17
    },
    '4.61': {
        sentence:         4,
        clueCount:        20,
        REQ_CLUE_COUNT:   17
    },
    '4.62': {
        sentence:         4,
        clueCount:        20,
        REQ_CLUE_COUNT:   17
    },

    '4.7': {
        sentence:         4,
        clueCount:        20,
        REQ_CLUE_COUNT:   17
    },
    '4.71': {
        sentence:         4,
        clueCount:        20,
        REQ_CLUE_COUNT:   17
    },
    '4.72': {
        sentence:         4,
        clueCount:        20,
        REQ_CLUE_COUNT:   17
    },

    '4.8': {
        sentence:         4,
        clueCount:        7,
        REQ_CLUE_COUNT:   7
    },

    '4.81': {
        sentence:         4,
        clueCount:        10,
        REQ_CLUE_COUNT:   10
    },

    '5': {
        sentence:       5,
        clueCount:      15,
        REQ_CLUE_COUNT: 15
    },

    '5.1': {
        sentence:       5,
        clueCount:      20,
        REQ_CLUE_COUNT: 20
    },

    '5.2': {
        sentence:       5,
        clueCount:      22,
        REQ_CLUE_COUNT: 22
    },

    '5.3': {
        sentence:       5,
        clueCount:      17,
        REQ_CLUE_COUNT: 17
    },

    '5.4': {
        sentence:       5,
        clueCount:      18,
        REQ_CLUE_COUNT: 18
    },

    '5.41': {
        sentence:       5,
        clueCount:      14,
        REQ_CLUE_COUNT: 14
    },

    '5.42': {
        sentence:       5,
        clueCount:      14,
        REQ_CLUE_COUNT: 14
    },

    '6': {
        sentence:       6,
        clueCount:      8,
        REQ_CLUE_COUNT: 8
    },

    '6.1': {
        sentence:       6,
        clueCount:      10,
        REQ_CLUE_COUNT: 10
    },

    '6.2': {
        sentence:       6,
        clueCount:      10,
        REQ_CLUE_COUNT: 10
    },

    '6.3': {
        sentence:       6,
        clueCount:      10,
        REQ_CLUE_COUNT: 10
    },

    '7': {
        sentence:       7,
        clueCount:      7,
        REQ_CLUE_COUNT: 7
    },

    '7.1': {
        sentence:       7,
        clueCount:      10,
        REQ_CLUE_COUNT: 10
    },

    '7.2': {
        sentence:       7,
        clueCount:      10,
        REQ_CLUE_COUNT: 10
    },

    '8': {
        sentence:       8,
        clueCount:      8,
        REQ_CLUE_COUNT: 8
    },

    '8.1': {
        sentence:       8,
        clueCount:      9,
        REQ_CLUE_COUNT: 9
    },

    '8.2': {
        sentence:       8,
        clueCount:      9,
        REQ_CLUE_COUNT: 9
    },

    '8.3': {
        sentence:       8,
        clueCount:      8,
        REQ_CLUE_COUNT: 8
    },

    '9': {
        sentence:       9,
        clueCount:      8,
        REQ_CLUE_COUNT: 8
    },

    '9.0': {
        sentence:       9,
        clueCount:      10,
        REQ_CLUE_COUNT: 10
    },

    '9.1': {
        sentence:       9,
        clueCount:      10,
        REQ_CLUE_COUNT: 10
    },

    '9.2': {
        sentence:       9,
        clueCount:      9,
        REQ_CLUE_COUNT: 9
    },

    '9.3': {
        sentence:       9,
        clueCount:      10,
        REQ_CLUE_COUNT: 10
    },

    '9.98': {
        sentence:       9,
        clueCount:      15,
        REQ_CLUE_COUNT: 15
    },

    '9.99': {
        sentence:       9,
        clueCount:      5,
        REQ_CLUE_COUNT: 5
    },

    'f': {
        sentence:       10,
        clueCount:      9,
        REQ_CLUE_COUNT: 9
    },

    'f.1': {
        sentence:       10,
        clueCount:      9,
        REQ_CLUE_COUNT: 9
    },

    'f.2': {
        sentence:       10,
        clueCount:      20,
        REQ_CLUE_COUNT: 17
    },

    'f.21': {
        sentence:       10,
        clueCount:      16,
        REQ_CLUE_COUNT: 16
    },

    'f.22': {
        sentence:       10,
        clueCount:      13,
        REQ_CLUE_COUNT: 13
    },

    'f.23': {
        sentence:       10,
        clueCount:      14,
        REQ_CLUE_COUNT: 14
    },

    'f.24': {
        sentence:       10,
        clueCount:      8,
        REQ_CLUE_COUNT: 8
    },

    'f.25': {
        sentence:       10,
        clueCount:      12,
        REQ_CLUE_COUNT: 12
    },
    'f.26': {
        sentence:       10,
        clueCount:      14,
        REQ_CLUE_COUNT: 14
    },

    'f.27': {
        sentence:       10,
        clueCount:      14,
        REQ_CLUE_COUNT: 14
    },

    'f.3': {
        sentence:       10,
        clueCount:      23,
        REQ_CLUE_COUNT: 23
    },

    'f.31': {
        sentence:       10,
        clueCount:      29,
        REQ_CLUE_COUNT: 29
    },

    'f.4': {
        sentence:       10,
        clueCount:      21,
        REQ_CLUE_COUNT: 21
    },

    'f.5': {
        sentence:       10,
        clueCount:      42,
        REQ_CLUE_COUNT: 42
    },

    'f.51': {
        sentence:       10,
        clueCount:      29,
        REQ_CLUE_COUNT: 29
    },

    'f.52': {
        sentence:       10,
        clueCount:      31,
        REQ_CLUE_COUNT: 31
    },

    'f.53': {
        sentence:       10,
        clueCount:      31,
        REQ_CLUE_COUNT: 31
    },

    'f.54': {
        sentence:       10,
        clueCount:      32,
        REQ_CLUE_COUNT: 32
    },

    'f.55': {
        sentence:       10,
        clueCount:      32,
        REQ_CLUE_COUNT: 32
    },

    'f.56': {
        sentence:       10,
        clueCount:      32,
        REQ_CLUE_COUNT: 32
    },

    'f.57': {
        sentence:       10,
        clueCount:      28,
        REQ_CLUE_COUNT: 32
    },

    'f.6': {
        sentence:       10,
        clueCount:      28,
        REQ_CLUE_COUNT: 32
    },

    'f.61': {
        sentence:       10,
        clueCount:      28,
        REQ_CLUE_COUNT: 32
    },

    'f.62': {
        sentence:       10,
        clueCount:      28,
        REQ_CLUE_COUNT: 32
    },

    'f.63': {
        sentence:       10,
        clueCount:      35,
        REQ_CLUE_COUNT: 32
    },

    'f.64': {
        sentence:       10,
        clueCount:      35,
        REQ_CLUE_COUNT: 32
    },

    'f.65': {
        sentence:       10,
        clueCount:      35,
        REQ_CLUE_COUNT: 32
    },

    'f.66': {
        sentence:       10,
        clueCount:      35,
        REQ_CLUE_COUNT: 32
    },

    'f.67': {
        sentence:       10,
        clueCount:      35,
        REQ_CLUE_COUNT: 32
    },

    'f.68': {
        sentence:       10,
        clueCount:      35,
        REQ_CLUE_COUNT: 32
    },

    'f.69': {
        sentence:       10,
        clueCount:      35,
        REQ_CLUE_COUNT: 32
    },

    'f.70': {
        sentence:       10,
        clueCount:      35,
        REQ_CLUE_COUNT: 32
    },

    'f.71': {
        sentence:       10,
        clueCount:      55,
        REQ_CLUE_COUNT: 32
    },

    'f.72': {
        sentence:       10,
        clueCount:      55,
        REQ_CLUE_COUNT: 32
    },

    'f.73': {
        sentence:       10,
        clueCount:      55,
        REQ_CLUE_COUNT: 32
    },

    'f.9': {
        sentence:       10,
        clueCount:      43,
        REQ_CLUE_COUNT: 31
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
    if (!APPLE[variety]) throw Error(`Bad apple variety: ${variety}`);
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
	src = getByVariety(options.apple);
    } else {
        throw new Error('No clue-type option supplied');
    }
    return src;
}

function getByVariety (apple) {
    // TODO: same code in note.js, refactor it out to getVariety()
    let variety = _.clone(apple);
    Debug(`Apple: ${apple} Variety: ${variety}`);
//    Expect(_.toNumber(variety.charAt(0))).is.above(0); // M
    let count = 1;
    if (variety.charAt(count) === '.') {
	count += 1;
	if (isVarietyDigit(variety.charAt(count))) count += 1; // M.X
	if (isVarietyDigit(variety.charAt(count))) count += 1; // M.XY
	Expect(count > 3).is.true();
    }
    variety = variety.slice(0, count);
    let src = metamorph(variety);
    if (_.isUndefined(src)) throw new Error(`APPLE[${variety}] not supported`);
/* REMOVED 3/7/2020 no longer suppor this shit
    if (_.size(apple) > _.size(variety) + 1) {
        src = cloneAsType(src, getTypeFromSuffix(apple.slice(_.size(variety), _.size(apple))));
    }
*/
    return src;
}

function isVarietyDigit (n) {
    return "0123456789".includes(n);
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
//    Expect(dir.charAt(0)).is.equal('p');
    return `${dir.charAt(0)}${clueType.variety}${getTypeSuffix(clueType)}`;
}

// e.g. p3s.c2-6.x2

// TODO: getTypeClueCount: synthClueCount = getByType(clueType).clueCount;
// TODO: test

function getLonghand (clueType, max = 2) {
    return `${getShorthand(clueType)}.c2-${clueType.synthClueCount}.x${max}`;
}

//

function getDirectory (clueType) {
    return `${DATA_DIR}${clueType.baseDir}`;
}


module.exports = {
    cloneAsNextType,
    getByBaseDirOption,
    getByOptions,  
    getByVariety,
    getDirectory,
    getShorthand,
    getLonghand,
    isValidBaseDirOption,
    isVarietyDigit,
    Options
};
