/*
 * note-merge.js
 */

'use strict';

const _                = require('lodash');
const Debug            = require('debug')('note-merge');
const Expect           = require('should/as-function');
const Fs               = require('fs-extra');
const Filter           = require('./filter');
const Note             = require('./note');
const NoteMaker        = require('./note-make');
const NoteParse        = require('./note-parse');
const Path             = require('path');
const Promise          = require('bluebird');

//

function loadNoteFilterLists(filename, noteName, options) {
    return Promise.join(
	Note.get(noteName, { content: true, notebookGuid: options.notebookGuid }),
	Filter.parseFile(filename, { urls: true, clues: true }),
	(note, filterData) => {
	    return [note, NoteParse.parse(note.content, { urls: true /*, clues: true */}), filterData];
	});

}

//

function compareSource(a, b) {
    let [as, bs] = [a.source, b.source];
    return as < bs ? -1 : as > bs ? 1 : 0; // jsperf for chrome
}

//

async function mergeFilterFile (filename, noteName, options) {
    // const [note, listFromNote, listFromFilterResult] =
    return loadNoteFilterLists(filename, noteName, options)
	.then(([note, listFromNote, listFromFilterResult]) => {
	    Debug(`note ${note && note.title}`);
	    const diffList = Filter.diff(listFromNote.sort(compareSource),
					 listFromFilterResult.sort(compareSource));

	    const expectedDiffCount = listFromFilterResult.length - listFromNote.length;
	    Expect(diffList.length).is.equal(expectedDiffCount);
	    Debug(`note(${listFromNote.length}), filter(${listFromFilterResult.length})` +
			`, diffList(${diffList.length}), diffExpected(${expectedDiffCount})`);

	    if (_.isEmpty(diffList)) {
		console.log('no differences');
		return undefined;
	    }
	    if (options.verbose) {
		console.log('diffList:');
		for (const elem of diffList) {
		    console.log(elem);
		}
	    }
	    const diffBody = NoteMaker.makeFromFilterList(diffList, options);
	    if (options.verbose) {
		console.log(`diffBody:\n${diffBody}`);
	    }
	    return Note.append(note, diffBody, options);
	});
}

//

module.exports = {
    mergeFilterFile
}
