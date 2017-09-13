/*
 * note-make.js
 */

'use strict';

const _                = require('lodash');
const Debug            = require('debug')('note-make');
const Expect           = require('should/as-function');
const Fs               = require('fs');
const He               = require('he');
const Promise          = require('bluebird');
const Querystring      = require('querystring');
const Readlines        = require('n-readlines');

//

const FsWrite          = Promise.promisify(Fs.write);

//

function open () {
    return '<div><span><font style="font-size: 12pt;">';
}

//

function close () {
    return '</font></span></div>';
}

//

function url (line) {
    const useNamedReferences = true;
    line = He.encode(line, { useNamedReferences });
    return `<a href="${line}">${line}</a>`;
}

//

function write (fd, text) {
    return FsWrite(fd, `${text}\n`);
}

//

function writeEmptyLine (fd) {
    return write(fd, '<div><br/></div>');
}

//

function writeUrl (fd, line) {
    return write(fd, `${open()}${url(line)}${close()}`);
}

//

function writeText (fd, line) {
    return write(fd, `${open()}${line}${close()}`);
}

// TODO: outputFd support

async function make (inputFilename, fd) {
    Expect(inputFilename).is.a.String();
    Expect(fd).is.a.Number();

    Debug(`filename: ${inputFilename}, fd: ${fd}`);

    let readLines = new Readlines(inputFilename);
    await write(fd, '<div>');
    while (true) {
	let line = readLines.next();
	if (line === false) break;
	line = line.toString();
	if (_.isEmpty(line)) {
	    await writeEmptyLine(fd);
	} else if (_.startsWith(line, 'http')) {
	    await writeUrl(fd, line);
	} else {
	    await writeText(fd, line);
	}
    }
    await write(fd, '</div>');
    await writeEmptyLine(fd);
}

//

module.exports = {
    make
}
