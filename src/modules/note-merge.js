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
const NoteParser       = require('./note-parse');
const Path             = require('path');
const Promise          = require('bluebird');
const Stringify        = require('stringify-object');

//
// i question this being a separate file. should probably be in modules/filter
// although this whole thing deals with filterlists, so maybe filter-list is
// a new module.

// get a note by name. if it doesn't exist, create it. return note with content.

async function getOrCreate (noteName, options) {
    let created = false;
    return Note.get(noteName, Object.assign(_.clone(options), { content: true, nothrow: true }))
	.then(note => {
	    if (note) return note;
	    created = true;
	    return Note.create(noteName, '<div></div>', options);
	}).then (note => {
	    if (!note) throw new Error(`error creating note, ${noteName}`);
	    if (!created) return note;
	    if (!note.guid) throw new Error(`created note has no guid, ${noteName}`);
	    return Note.get(null, Object.assign(_.clone(options), { guid: note.guid }));
	});
}

// unnecessesarily convoluted

async function loadNoteFilterLists(filename, noteName, options) {
    Debug(`++loadNoteFilterLists()`);
    return Promise.join(
	// TODO: Note.getOrCreate(
	// TODO: get rid of this. if no note, don't create. wait until Note.append.
	getOrCreate(noteName, options),
	Filter.parseFile(filename),
	(note, listFromFile) => {
	    return [note,
		    Filter.parseLines(NoteParser.parseDom(note.content, options), options),
		    listFromFile];
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
    
    // should these be in filterMerge() (which calls filter + merge?)
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

    const addedClueCount = Filter.addKnownClues(listFromNote);
    Filter.addKnownClues(listFromFilter);

    // at this point it's still possible there are removed clues in listFromNote
    const removedClueMap = Filter.getRemovedClues(listFromNote);
    let removedClueCount = 0;
    if (!_.isEmpty(removedClueMap)) {
	// remove clues from removedClueMap that are known (i.e. they weren't removed)
	// why is this in Filter? not sure.
	Filter.removeKnownClues(removedClueMap, options);
	// remove clues from listFromNote that are in removedClueMap
	removedClueCount = Filter.removeRemovedClues(listFromNote, removedClueMap, options);
    }
    if (options.verbose) {
	console.log(`filteredUrls(${filteredUrlCount}), filteredSources(${filteredSrcCount})` +
		    `, addedClues(${addedClueCount}), removedClues(${removedClueCount})`);
	Filter.dumpList(listFromNote, { /*all: true,*/ fd: process.stdout.fd });
    }

    // if listFromNote was modified, concat diffList and build note from result
    if (filteredUrlCount || filteredSrcCount || addedClueCount || removedClueCount) {
	Debug(`base note changed - building new note body`);
	listFromNote.push(...diffList);
	const noteBody = NoteMaker.makeFromFilterList(listFromNote, { outerDiv: true });
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
    // TODO: options.force_create could be used here.
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
