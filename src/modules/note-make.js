/*
 * note-make.js
 */

'use strict';

const _                = require('lodash');
const Debug            = require('debug')('note-make');
const Expect           = require('should/as-function');
const EncodeURL        = require('encodeurl');
const Filter           = require('./filter');
const Fs               = require('fs-extra');
const He               = require('he');
const My               = require('util');
const Options          = require('./options').options;
const Readlines        = require('n-readlines');

//

const Note = {
    point_size: function () { return Options.point_size || 14; },
    open:       function (checkbox = '') { return `<div>${checkbox}<span><font style="font-size: ${Note.point_size()}pt;">`; },
    close:      '</font></span></div>',
    emptyLine:  '<div><br/></div>',
    checkbox:   function (checked) { return `<en-todo checked="${checked ? 'true' : 'false'}"/>&#xA0;&#xA0;`; },
    twoCheckboxes: function (checked) {
        const todo = c => `<en-todo checked="${c ? 'true' : 'false'}"/>`;
        return `${todo(checked)}&#xA0;Y&#xA0;&#xA0;${todo(false)}&#xA0;N&#xA0;&#xA0;`;
    }
};

//

function url (line) {
    const useNamedReferences = true;
    //let decoded = decodeURI(line);
    let decoded = line.replace('%3A', ':').replace('%2F', '/');
    decoded = decoded.replace('&', '&amp;'); //.replace('?redirect=no', ''); // parens
    let encoded = He.encode(line, { useNamedReferences });
    encoded = encoded.replace('(', '%28').replace(')', '%29'); // parens
    //console.log(`url: ${decoded}\n  encoded: ${encoded}`);
    return `<a href="${encoded}">${decoded}</a>`;
}

//

function writeEmptyLine (dest) {
    return dest + Note.emptyLine;
}

//

function writeUrl (dest, line, suffix) {
    const _suffix = suffix ? `,${suffix}` : '';
    return dest + `${Note.open()}${url(line)}${_suffix}${Note.close}`;
}

//

function writeText (dest, line, checkbox, checked = false, twoCheckboxes = false) {
    const _cb = twoCheckboxes ? Note.twoCheckboxes(checked)
        : checkbox ? Note.checkbox(checked) : '';
    return `${dest}${Note.open(_cb)}${line}${Note.close}`;
}

//

function loadYesPairs (filename) {
    const pairs = new Set();
    if (!filename) return pairs;

    const readLines = new Readlines(filename);
    while (true) {
        const nextLine = readLines.next();
        if (nextLine === false) break;

        const pair = nextLine.toString().trim();
        if (_.isEmpty(pair)) continue;

        const names = pair.split(',');
        if (names.length !== 2) {
            throw new Error(`invalid pair in ${filename}: ${pair}`);
        }
        pairs.add(pair);
        pairs.add(`${names[1]},${names[0]}`);
    }
    return pairs;
}

// this function is dumb anyway.  Fitler.parse => list -> makefromFilterList

async function makeFromFilterFile (filename, options = {}) {
    console.log(`options.point_size: ${Options.point_size}`);

    Expect(filename).is.a.String();
    Debug(`filename: ${filename}`);

    let dest = '';
    const anyCheckbox = options.checkbox || options.twoCheckboxes;
    const yesPairs = anyCheckbox ? loadYesPairs(options.yesPairsFile) : new Set();
    let readLines = new Readlines(filename);
    if (options.outerDiv) {
        dest += '<div>';
    }
    while (true) {
        let line = readLines.next();
        if (line === false) break;
        line = line.toString();
        if (_.isEmpty(line)) {
            dest = writeEmptyLine(dest);
        } else if (_.startsWith(line, 'http')) {
            dest = writeUrl(dest, line);
        } else {
            dest = writeText(dest, line, options.checkbox, yesPairs.has(line),
                             options.twoCheckboxes);
        }
    }
    dest = writeEmptyLine(dest);
    if (options.outerDiv) {
        dest += '</div>';
    }
    return dest;
}

// move to Filter? and maybe the writeXXX methods stay here,
// or in modules/note-markup
//
// changes to this probably require similar changes to
// Filter.dumpList()
//
function makeFromFilterList (list, options = {}) {
    Expect(list).is.an.Array();

    let result = '';
    if (options.outerDiv) {
        result += '<div>';
    }
    for (const sourceElem of list) {
        if (options.verbose) Debug(`${sourceElem.source}`);
        let source = sourceElem.source || sourceElem;
        if (sourceElem.suffix) source += `,${sourceElem.suffix}`;
        result = writeText(result, source, sourceElem.suffix);
        for (const urlElem of sourceElem.urls || []) {
            let url = urlElem.url || urlElem;
            result = writeUrl(result, url, urlElem.suffix);
            for (const clueElem of urlElem.clues || []) {
                const text = Filter.getClueText(clueElem, options);
                if (!text) continue;
                result = writeText(result, text);
            }
        }
        result = writeEmptyLine(result);
    }
    if (options.outerDiv) {
        result += '</div>';
        result = writeEmptyLine(result);
    }
    return result;
}

//

module.exports = {
    makeFromFilterFile,
    makeFromFilterList
}
