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
    open:       function () { return `<div><span><font style="font-size: ${Note.point_size()}pt;">`; },
    close:      '</font></span></div>',
    emptyLine:  '<div><br/></div>'
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

function writeText (dest, line) {
    return `${dest}${Note.open()}${line}${Note.close}`;
}

// this function is dumb anyway.  Fitler.parse => list -> makefromFilterList

async function makeFromFilterFile (filename, options = {}) {
    console.log(`options.point_size: ${Options.point_size}`);

    Expect(filename).is.a.String();
    Debug(`filename: ${filename}`);

    let dest = '';
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
            dest = writeText(dest, line);
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
