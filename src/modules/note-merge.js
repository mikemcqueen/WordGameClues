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
const Stringify        = require('stringify-object');

//

function loadNoteFilterLists(filename, noteName, options) {
    const parseOpt = { urls: true, clues: true };
    return Promise.join(
	Note.get(noteName, { content: true, notebookGuid: options.notebookGuid }),
	Filter.parseFile(filename, parseOpt),
	(note, filterData) => {
	    return [note, NoteParse.parse(note.content, parseOpt), filterData];
	});
}

/*
//

function compareSource(a, b) {
    let [as, bs] = [a.source, b.source];
    return as < bs ? -1 : as > bs ? 1 : 0; // jsperf for chrome
}
*/

//

function mergeFilterFile (filename, noteName, options) {
    // const [note, listFromNote, listFromFilter] =
    return loadNoteFilterLists(filename, noteName, options)
	.then(([note, listFromNote, listFromFilter]) => {
	    Debug(`title: ${note && note.title}, noteList(${listFromNote && listFromNote.length})` +
		 `, filterList(${listFromFilter && listFromFilter.length})`);
	    if (options.verbose) {
		console.log(`listFromNote:\n${Stringify(listFromNote)}`);
		console.log(`listFromFilter:\n${Stringify(listFromFilter)}`);
	    }
	    const diffList = Filter.diff(listFromNote, listFromFilter);

	    const expectedDiffCount = listFromFilter.length - listFromNote.length;
	    Expect(diffList.length).is.equal(expectedDiffCount);
	    Debug(`note(${listFromNote.length}), filter(${listFromFilter.length})` +
		  `, diffList(${diffList.length}), diffExpected(${expectedDiffCount})`);

	    Debug(`diffList(${diffList.length})`);
	    if (options.verbose) {
		console.log(`diffList: ${Stringify(diffList)}`);
                /*
                for (const elem of diffList || []) {
                     console.log(elem);
                }
		 */
	    }
	    let [filteredListFromNote, filtered] = options.filter_urls
		    ? Filter.filterUrls(listFromNote, listFromFilter, options) : [,0];
	    Debug(`filtered: ${filtered}`);
	    if (options.filter_urls && filtered) { 
		// some changes -- concat lists and build new note
		Debug(`building new note body`);
		filteredListFromNote.push(...diffList);
		const noteBody = NoteMaker.makeFromFilterList(filteredListFromNote, { outerDiv: true });
		Note.setContentBody(note, noteBody);
		return Note.update(note, options);
	    }

	    // original note unchanged -- only append if there are diffs
	    if (_.isEmpty(diffList)) {
		Debug(`no diffs, no changes - nothing to do`);
		return false;
	    }
	    const diffBody = NoteMaker.makeFromFilterList(diffList, options);
	    if (options.verbose) {
		console.log(`diffBody:\n${diffBody}`);
	    }
	    Debug(`appending new results to note`);
	    return Note.append(note, diffBody, options);
	});
}

//

module.exports = {
    mergeFilterFile
}
