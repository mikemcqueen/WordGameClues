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

async function merge(note, listFromNote, listFromFilter, options) {
    Debug(`title: ${note && note.title}, noteList(${listFromNote && listFromNote.length})` +
	  `, filterList(${listFromFilter && listFromFilter.length})`);
    if (options.verbose) {
	console.log(`listFromNote:\n${Stringify(listFromNote)}`);
	console.log(`listFromFilter:\n${Stringify(listFromFilter)}`);
    }
    // filter before creating difflist
    if (!options.noFilterSources) { 
	let [filteredList, count] = Filter.filterSources(listFromNote, options);
	Debug(`filtered sources: ${count}`);
	listFromNote = filteredList;
    }
    let [filteredList, filteredUrlCount] = Filter.filterUrls(listFromNote, listFromFilter, options);
    const filteredSrcCount = listFromNote.length - filteredList.length;
    if (!options.noFilterUrls) {
	Debug(`filtered urls: ${filteredUrlCount}, sources: ${filteredSrcCount}`);
	listFromNote = filteredList;
    }

    // create diff list
    const diffList = Filter.diff(listFromNote, listFromFilter);
    const expectedDiffCount = listFromFilter.length - listFromNote.length;
    Debug(`note(${listFromNote.length}), filter(${listFromFilter.length})` +
	  `, actualDiff(${diffList.length}), diffExpected(${expectedDiffCount})`);
    Expect(diffList.length).is.equal(expectedDiffCount);
    if (options.verbose) {
	console.log(`diffList: ${Stringify(diffList)}`);
        /*
         for (const elem of diffList || []) {
         console.log(elem);
         }
	 */
    }

    // if listFromNote was filtered, concat diffList and build note from result
    if (filteredUrlCount || filteredSrcCount) {
	Debug(`note changed - building new note body`);
	listFromNote.push(...diffList);
	const noteBody = NoteMaker.makeFromFilterList(filteredList, { outerDiv: true });
	Note.setContentBody(note, noteBody);
	return Note.update(note, options);
    }
    
    // original note unchanged -- only append if there are diffs
    Debug(`note unchanged`);
    if (_.isEmpty(diffList)) {
	Debug(`diffList empty - nothing to do`);
	return note;
    }
    const diffBody = NoteMaker.makeFromFilterList(diffList, options);
    if (options.verbose) {
	console.log(`diffBody:\n${diffBody}`);
    }
    Debug(`appending new results to note`);
    return Note.append(note, diffBody, options);
}

//

function mergeFilterFile (filename, noteName, options) {
    // const [note, listFromNote, listFromFilter] =
    return loadNoteFilterLists(filename, noteName, options)
	.then(([note, listFromNote, listFromFilter]) => merge(note, listFromNote, listFromFilter, options));
}

//

module.exports = {
    mergeFilterFile
}
