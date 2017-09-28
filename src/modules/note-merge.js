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
// i question this being a separate file. should probably be in modules/filter
// although this whole thing deals with filterlists, so maybe filter-list is
// a new module.

// unnecessesarily convoluted

async function loadNoteFilterLists(filename, noteName, options) {
    Debug(`++loadNoteFilterLists()`);
    const parseOpt = { urls: true, clues: true };
    const getOpt = _.clone(options);
    getOpt.content = true;
    return Promise.join(
	Note.get(noteName, getOpt),
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
    Debug(`++merge() title: ${note && note.title}, noteList(${listFromNote && listFromNote.length})` +
	  `, filterList(${listFromFilter && listFromFilter.length})`);
    if (options.verbose) {
	console.log(`listFromNote:\n${Stringify(listFromNote)}`);
	console.log(`listFromFilter:\n${Stringify(listFromFilter)}`);
    }
    
    // should these be in filterMerge (which calls merge?)
    // is there ever a merge without a filter?
    // filter rejects before creating diffList
    let filteredSrcCount = 0;
    if (!options.noFilterSources) { 
	let [filteredList, count] = Filter.filterRejectSources(listFromNote, options);
	Debug(`filtered sources: ${count}`);
	filteredSrcCount = count;
	listFromNote = filteredList;
    }
    let filteredUrlCount = 0;
    if (!options.noFilterUrls) {
	let [filteredList, count] = Filter.filterRejectUrls(listFromNote, listFromFilter, options);
	Debug(`filtered urls: ${count}`);
	// add however many sources were filtered due to all URLs being removed
	filteredSrcCount += listFromNote.length - filteredList.length;
	filteredUrlCount = count;
	listFromNote = filteredList;
    }
    // create diff list
    const diffList = Filter.diff(listFromNote, listFromFilter);
    Debug(`note(${listFromNote.length}), filter(${listFromFilter.length})` +
	  `, diff(${diffList.length})`);
    if (options.verbose) {
	console.log(`diffList: ${Stringify(diffList)}`);
    }

    let addedClueCount = Filter.addKnownClues(listFromNote);
    Filter.addKnownClues(listFromFilter);

    if (options.verbose) {
	console.log(`filteredUrls(${filteredUrlCount}), filteredSources(${filteredSrcCount})` +
		    `, addedClues(${addedClueCount})`);
	Filter.dumpList(listFromNote, { all: true, fd: process.stdout.fd });
    }

    // if listFromNote was filtered, concat diffList and build note from result
    if (filteredUrlCount || filteredSrcCount || addedClueCount) {
	Debug(`base note changed - building new note body`);
	listFromNote.push(...diffList);
	const noteBody = NoteMaker.makeFromFilterList(listFromNote, { outerDiv: true });
	Note.setContentBody(note, noteBody);
	return Note.update(note, options);
    }
    
    // well, actually, at this point it's still possible there are removed clues in
    // listFromNote. we found them earlier. how do we get them here.

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
