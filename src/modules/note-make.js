/*
 * note-make.js
 */

'use strict';

const _                = require('lodash');
const Debug            = require('debug')('note-make');
const Expect           = require('should/as-function');
const Fs               = require('fs-extra');
const He               = require('he');
const Readlines        = require('n-readlines');

//

const Note = {
    Open:      '<div><span><font style="font-size: 12pt;">',
    Close:     '</font></span></div>',
    EmptyLine: '<div><br/></div>'
};

//

function url (line) {
    const useNamedReferences = true;
    line = He.encode(line, { useNamedReferences });
    return `<a href="${line}">${line}</a>`;
}

//

async function write (dest, text) {
    const output = `${text}`; // + \n`;
    if (_.isString(dest)) {
	return `${dest}${output}`;
    }
    if (_.isNumber(dest)) {
	return Fs.write(dest, output);
    }
    throw new Error(`bad dest, ${dest}`);
}

//

function writeEmptyLine (dest) {
    return write(dest, Note.EmptyLine);
}

//

function writeUrl (dest, line) {
    return write(dest, `${Note.Open}${url(line)}${Note.Close}`);
}

//

function writeText (dest, line) {
    return write(dest, `${Note.Open}${line}${Note.Close}`);
}

//

async function makeFromFile (inputFilename, options = {}) {
    Expect(inputFilename).is.a.String();
    Debug(`filename: ${inputFilename}`);

    // write to fd if supplied; else build & return string
    let dest = options.fd || '';
    let readLines = new Readlines(inputFilename);
    if (options.outerDiv) {
	await write(dest, '<div>').then(result => {
	    if (_.isString(dest)) dest = result;
	});
    }
    while (true) {
	let promise;
	let line = readLines.next();
	if (line === false) break;
	line = line.toString();
	if (_.isEmpty(line)) {
	    promise = writeEmptyLine(dest);
	} else if (_.startsWith(line, 'http')) {
	    promise = writeUrl(dest, line);
	} else {
	    promise = writeText(dest, line);
	}
	await promise.then(result => {
	    if (_.isString(dest)) dest = result;
	});
    }
    if (options.outerDiv) {
	await write(dest, '</div>').then(result => {
	    if (_.isString(dest)) dest = result;
	});
    }
    await writeEmptyLine(dest).then(result => {
	if (_.isString(dest)) dest = result;
    });
    return dest;
}

//

function makeFromFilterList (list, options = {}) {
    Expect(list).is.an.Array();

    let result = '';
    if (options.outerDiv) {
	result = write(result, '<div>');
    }
    for (const sourceElem of list) {
	result = writeText(result, sourceElem.source);
	for (const urlElem of sourceElem.urls) {
	    result = writeUrl(result, urlElem.url);
	    for (const clue of urlElem.clues) {
		result = writeText(result, clue);
	    }
	}
	result = writeEmptyLine(result);
    }
    if (options.outerDiv) {
	result = write(result, '</div>');
	result = writeEmptyLine(result);
    }
    return result;
}

//

module.exports = {
    makeFromFile,
    makeFromFilterList
}
