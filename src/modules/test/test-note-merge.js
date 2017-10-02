//
// test-note-merge
//

'use strict';

const _            = require('lodash');
const Debug        = require('debug')('test-note-merge');
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
const BaseNoteName =     'test-note-merge';
const BaseNoteNameMd =   'test-note-merge-markdown';

const BaseCount = {
    sources: 7,
    urls:    40,
    clues:   0
};
const UpdateCount =  {
    sources: 17,
    urls:    55,
    clues:   0
};

//

function compareCount (actual, expected) {
    Debug(`actual:   ${Stringify(actual)}`);
    Debug(`expected: ${Stringify(expected)}`);
    Expect(actual.sources).is.equal(expected.sources);
    Expect(actual.urls).is.equal(expected.urls);
    Expect(actual.clues).is.equal(expected.clues);
}

// TODO: Note.getContent(note)

function mergeTest (baseNoteName, updateNoteFilename, expectedCount, done) {
    return Note.get(baseNoteName, { content: true, notebook: TestNotebookName })
	.then(baseNote => {
	    console.log(`base   notebook guid ${baseNote.notebookGuid}`);
	    const count = Filter.count(Filter.parseLines(NoteParser.parseDom(baseNote.content)));
	    compareCount(count, expectedCount.base);
	    return Note.create(`${baseNoteName}_copy`, baseNote.content.toString(), {
		content:      true, // i.e. supplied body is actually full content
		notebookGuid: baseNote.notebookGuid
	    });
	}).then(copyBaseNote => {
	    console.log(`copy   notebook guid ${copyBaseNote.notebookGuid}`);
	    return NoteMerge.mergeFilterFile(updateNoteFilename, copyBaseNote.title, {
		notebookGuid: copyBaseNote.notebookGuid
	    });
	}).then(mergedNote => {
	    console.log(`merged notebook guid ${mergedNote.notebookGuid}`);
	    console.log(`note guid ${mergedNote.guid}`);
	    return Promise.all([mergedNote, Note.getContent(mergedNote.guid)]);
	}).then(([mergedNote, content]) => {
	    const filterList = Filter.parseLines(NoteParser.parseDom(content));
	    //TODO: move deleteNote to post test cleanup step
	    return Promise.all([Filter.count(filterList), Note.deleteNote(mergedNote.guid)]);
	}).then(([count, _]) => {		
	    console.log('copy deleted');
	    compareCount(count, expectedCount.merged);
	    done();
	}).catch(err => {
	    console.log('error');
	    console.log(err, err.stack);
	    done(err);
	});
}

//

describe ('note-merge tests', function() {
    this.timeout(5000);
    this.slow(1500);

    ////////////////////////////////////////////////////////////////////////////////
    //
    it ('should merge-append a note', function (done) {
	//const updateNoteName = 'test-note-merge_update';
	const updateNoteFilename = Test.file('test-note-merge.update');

	// suboptimal (does note.get twice) but dancing around questionable
	// logic used in note-merge, namely the paired note/filter file loading

	const base = BaseCount;
	const merged =  {
	    sources: 24,
	    urls:    95,
	    clues:   0
	};
	// before: delete all notes named baseNoteName_copy

	mergeTest(BaseNoteName, updateNoteFilename, { base, merged }, done);

	// after: delete all notes named baseNoteName_copy
    });


    ////////////////////////////////////////////////////////////////////////////////
    //
    it ('should merge-body a note', function (done) {
	//const updateNoteName = 'test-note-merge_update';
	const updateNoteFilename = Test.file('test-note-merge.md-update');

	// suboptimal (does note.get twice) but dancing around questionable
	// logic used in note-merge, namely the paired note/filter file loading

	const base = BaseCount;
	const merged =  {
	    sources: 22, // 24 - 2 rejected
	    urls:    84, // 95 - 7 (url,x) - 4 (2 rejected sources)
	    clues:   0
	};
	// before: delete all notes named baseNoteName_copy

	mergeTest(BaseNoteNameMd, updateNoteFilename, { base, merged }, done);

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

