/*
 * note-merge.js
 */

'use strict';

const _                = require('lodash');
const Debug            = require('debug')('note-make');
const Expect           = require('should/as-function');
const Fs               = require('fs-extra');
const Filter           = require('./filter');
const Note             = require('./note');
const NoteParse        = require('./note-parse');
const Path             = require('path');

//

function loadNoteFilterLists(filename, options) {
    const noteName = Path.basename(filename);
    const nbName = Note.getNotebookName(noteName, { strict: true });
    return Promise.join(
	Note.get(nbName, noteName, { content: true }),
	Filter.parseFile(filename, { urls: true, clues: true }),
	(note, filterData) => {
	    return [note, NoteParse.parse(note.content), filterData];
	});

}

//

function compareSource(a, b) {
    let [a, b] = [a.source, b.source];
    return a < b ? -1 : a > b ? 1 : 0; // jsperf for chrome
}

//

function dumpDiff (diffList) {
    for (const elem of diffList) {
	console.log(elem);
    }
}

//

async function mergeFilterFile (filename, options) {
    // const [note, listFromNote, listFromFilterResult] =
    loadNoteFilterLists(filename, options) // .catch(err => { throw err; })
	.then((note, listFromNote, listFromFilterResult) => {
	    const diffList = Filter.diff(listFromNote.sort(compareSource),
					 listFromFilterResult.sort(compareSource));
	    dumpDiff(diffList);
	    console.log(`note(${listFromNote.length}), filter(${listFromFilterResult.length})` +
			`, diff(${listFromFilterResult.length - listfromNote.length})`);
	});
    
}

//

module.exports = {
    mergeFilterFile
}
