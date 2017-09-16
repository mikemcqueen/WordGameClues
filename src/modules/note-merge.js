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
    return loadNoteFilterLists(filename, noteName, options) // .catch(err => { throw err; })
	.then(([note, listFromNote, listFromFilterResult]) => {
	    Debug(`note ${note && note.title}` +
		  `, noteList: ${listFromNote}` + //  && listFromNote.length}`
		  `, filterList: ${listFromFilterResult} `); //  && listFromFilterResult.length}`);

	    console.log('noteList:');
	    console.log(listFromNote);
	    console.log('filterList:');
	    console.log(listFromFilterResult);

	    const diffList = Filter.diff(listFromNote.sort(compareSource),
					 listFromFilterResult.sort(compareSource));
	    console.log(`note(${listFromNote.length}), filter(${listFromFilterResult.length})` +
			`, difList(${diffList.length})` +
			`, diffActual(${listFromFilterResult.length - listFromNote.length})`);
	    return diffList;
	});
    
}

//

module.exports = {
    mergeFilterFile
}
