//
// test-note-merge
//

'use strict';

const _            = require('lodash');
const Expect       = require('should/as-function');
const Filter       = require('../filter');
const Fs           = require('fs-extra');
const My           = require('../util');
const Note         = require('../note');
const NoteMerge    = require('../note-merge');
const NoteParser   = require('../note-parse');
const Path         = require('path');
const Stringify    = require('stringify-object');
const Test         = require('./test');

const TestNotebookName = 'test';

// TODO: Note.getContent(note)

function mergeTest (baseNoteName, updateNoteFilename, expect, done) {
    return Note.get(baseNoteName, { content: true, notebook: TestNotebookName })
	.then(baseNote => {
	    console.log(`base   notebook guid ${baseNote.notebookGuid}`);
	    return Note.create(`${baseNoteName}_copy`, baseNote.content.toString(), {
		content:      true, // i.e. supplied body is actually full content
		notebookGuid: baseNote.notebookGuid
	    });
	}).then(copyBaseNote => {
	    console.log(`copy   notebook guid ${copyBaseNote.notebookGuid}`);
	    return NoteMerge.mergeFilterFile(updateNoteFilename, copyBaseNote.title, {
		notebookGuid: copyBaseNote.notebookGuid,
		filter_urls:  true // filter reject urls
	    });
	}).then(mergedNote => {
	    console.log(`merged notebook guid ${mergedNote.notebookGuid}`);
	    console.log(`note guid ${mergedNote.guid}`);
	    return Promise.all([mergedNote, Note.getContent(mergedNote.guid)]);
	}).then(([mergedNote, content]) => {
	    const filterList = NoteParser.parse(content, { urls: true, clues: true });
	    //TODO: move deleteNote to post test cleanup step
	    return Promise.all([Filter.count(filterList), Note.deleteNote(mergedNote.guid)]);
	}).then(([count, _]) => {		
	    console.log('copy deleted');
	    Expect(count.sources).is.equal(expect.sources);
	    Expect(count.urls).is.equal(expect.urls);
	    Expect(count.clues).is.equal(expect.clues);
	    done();
	}).catch(err => {
	    console.log('error');
	    console.log(err, err.stack);
	    done(err);
	});
}

//

describe ('note-merge tests', function() {
    this.timeout(20000);
    this.slow(4000);

    ////////////////////////////////////////////////////////////////////////////////
    //
    // test Result.fileScoreSaveCommit
    //
    // TODO: all the removing/adding is unnecessary. just make sure the file is there
    // in before(), then writeAdd if it isn't
    //
    it ('should merge notes', function (done) {
	const baseNoteName = 'test-note-merge';
	const baseNoteNameMd = 'test-note-merge-markdown';
	//const updateNoteName = 'test-note-merge_update';
	const updateNoteFilename = Test.file('test-note-merge.update');

	// suboptimal (does note.get twice) but dancing around questionable
	// logic used in note-merge, namely the paired note/filter file loading

	const expect =  {
	    sources: 24,
	    urls:    95,
	    clues:   0
	};
	// before: delete all notes named baseNoteName_copy

	mergeTest(baseNoteName, updateNoteFilename, expect, done);

	// after: delete all notes named baseNoteName_copy
    });


    /*
    it ('should merge note files', function (done) {
	const baseNoteFilename = 'note-merge-test';
	const mdNoteFilename = 'note-merge-test.md';
	const updateNoteFilename = 'note-merge-test.update';

	return Fs.readFile(Test.File(baseNoteFilename))
	
	    .then(([copyBaseNote, updateNote]) => {
		const mergedNote = NoteMerge.merge(copyBaseNote, updateNote);
		const count = Note.count(mergedNote);
		Expect(count.sources);
		Expect(count.urls);
		Expect(count.clues);
		console.log('done');
		done();
	    }).catch(err => {
		console.log(`error, ${err}`);
		done(err);
	    });
    });
*/
});

