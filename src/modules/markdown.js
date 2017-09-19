//
// markdown.js
//

'use strict';

//

const Debug  = require('debug')('markdown');
const Expect = require('should/as-function');

//

const Suffix =  {
    clue:   'c',
    reject: 'x'
};

Suffix.any = [Suffix.clue, Suffix.reject];

function isValidSuffix (suffix) {
    return Suffix.any.includes(suffix);
}

//

function getSuffix (line) {
    Expect(line).is.a.String();

    let suffix;
    let match = false;
    let index = line.lastIndexOf(',');
    if (index > -1) {
	suffix = line.slice(index + 1, line.length).trim();
	if (!isValidSuffix(suffix)) {
	    Debug(`invalid suffix, ${suffix}`);
	    suffix = undefined;
	} else {
	    line = line.slice(0, index);
	}
    }
    return [line, suffix];
}

// process last comma suffix, e.g. ",x"

function hasSuffix (line, suffix) {
    Expect(line).is.a.String();
    Expect(suffix).is.a.String().and.not.empty();

    const [base, lineSuffix] = getSuffix(line);
    const has = (lineSuffix === suffix);
    return [has, base];
}

//

module.exports = {
    getSuffix,
    hasSuffix,
    isValidSuffix,
    Suffix
};
