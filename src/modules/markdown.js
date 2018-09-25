//
// markdown.js
//

'use strict';

//

const _      = require('lodash');
const Debug  = require('debug')('markdown');
const Expect = require('should/as-function');

//

const Prefix = {
    source: '@',
    maybe:  ':',
    known:  '#',
    remove: '-'
};
Prefix.any  = _.values(Prefix);
Prefix.clue = [Prefix.known, Prefix.remove];

const Suffix =  {
    clue:   'c',
    reject: 'x',
};
Suffix.any = _.values(Suffix);

//

function hasPrefix (line, prefix) {
    return line.charAt(0) === prefix;
}

//

function hasSourcePrefix (line) {
    return hasPrefix(line, Prefix.source);
}

//

function hasMaybePrefix (line) {
    return hasPrefix(line, Prefix.maybe);
}

//

function hasKnownPrefix (line) {
    return hasPrefix(line, Prefix.known);
}

//

function hasRemovePrefix (line) {
    return hasPrefix(line, Prefix.remove);
}

//

function isValidSuffix (suffix) {
    return Suffix.any.includes(suffix);
}

//

function isValidPrefix (prefix) {
    return Prefix.any.includes(prefix);
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
            if (suffix.length === 1) {
                Debug(`invalid suffix, ${suffix}`);
            }
            suffix = undefined;
        } else {
            line = line.slice(0, index);
        }
    }
    return [line, suffix];
}

//

function getPrefix (line, allowed = Prefix.any) {
    Expect(line).is.a.String();
    if (!_.isArray(allowed)) allowed = [allowed];
    let prefix;
    let firstChar = line.charAt(0);
    if (allowed.includes(firstChar)) {
        prefix = firstChar;
        line = line.slice(1, line.length);
    }
    return [line, prefix];
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
    getPrefix,
    getSuffix,
    hasPrefix,
    hasSuffix,
    hasKnownPrefix,
    hasMaybePrefix,
    hasRemovePrefix,
    hasSourcePrefix,
    isValidSuffix,
    Prefix,
    Suffix
};
