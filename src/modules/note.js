/*
 * note.js
 */

'use strict';

//

const _                = require('lodash');
const Clues            = require('../clue-types');
const Evernote         = require('evernote');
const EvernoteConfig   = require('../../data/evernote-config.json');
const Expect           = require('should/as-function');
const Fs               = require('fs-extra');
const Log              = require('./log')('note');
const Path             = require('path');
const Promise          = require('bluebird');
const Stringify        = require('stringify-object');
 
//

function NodeNote (options) {
    let client = new Evernote.Client({
	token: options.token,
	sandbox: options.sandbox
    });
    this.userStore = client.getUserStore();
    this.noteStore = client.getNoteStore();
}

const ProductionDev   = new NodeNote({ token: EvernoteConfig.production.developerToken, sandbox: false });
const ProductionOauth = new NodeNote({ token: EvernoteConfig.production.oauthToken, sandbox: false });
const SandboxDev      = new NodeNote({ token: EvernoteConfig.sandbox.developerToken, sandbox: true });
const SandboxOauth    = new NodeNote({ token: EvernoteConfig.sandbox.oauthToken, sandbox: true });

//

function getNotestore (production) {
    return production ? ProductionOauth.noteStore : SandboxOauth.noteStore;
}

//

async function getNotebook (name, options = {}) {
    if (!name) return undefined;
    const noteStore = getNotestore(options.production);
    return noteStore.listNotebooks()
	.then(nbList => {
	    for(const nb of nbList) {
		Log.debug(`notebook: ${nb.name}`);
		let match = false;
		if (options.relaxed) {
		    match = _.includes(nb.name, name);
		} else {
		    match = (nb.name === name);
		}
		if (match ) {
		    Log.debug(`match`);
		    return nb;
		}
	    }
	    Log.debug(`notebook not found, ${name}`);
	    return undefined;
	});
}

// probably belongs in clue-types, or somewhere else

function getWorksheetName (noteNameOrClueType) {
    let noteName = noteNameOrClueType;
    if (_.isObject(noteName)) {
	noteName = Clues.getShorthand(noteName); // noteName isa clueType
    } else {
	const appleExpr = /p[0-9]s?/;
	const result = appleExpr.exec(noteName);
	if (!result || (result.index !== 0)) return undefined;
    }
    Expect(noteName).is.a.String();
    Expect(noteName.charAt(0)).is.equal('p');
    Expect(_.toNumber(noteName.charAt(1))).is.above(0);
    let count = 2;
    if (noteName.charAt(2) === 's') count += 1;
    const wsName = `Worksheets.${noteName.slice(0, count)}`;
    Log.debug(`worksheet name: ${wsName}`);
    return wsName;
}

// dumb name. should be getNotebook({ guid }, options);

async function getNotebookByGuid (guid, options = {}) {
    const noteStore = getNotestore(options.production);
    return noteStore.getNotebook(guid);
}

//

async function getNotebookByOptions (options) {
    if (options.notebookGuid) return getNotebookByGuid(options.notebookGuid, options);
    if (!options.notebook) {
	Log.debug(`NO NOTEBOOK - THROW?`);
	return undefined;
    }
    return getNotebook(options.notebook, options)
	.then(nb => {
	    if (!nb) throw new Error(`no notebook matches: ${options.notebook}`);
	    return nb;
	});
}

//
/* noteSpec:
 includeContent
 includeResourcesData
 includeResourcesRecognition
 includeResourcesAlternateData
 includeSharedNotes
 includeNoteAppDataValues
 includeResourceAppDataValues
 includeAccountLimits
 */

/* metaSpec:
 includeTitle
 includeContentLength
 includeCreated
 includeUpdated
 includeDeleted
 */

// TODO: remove title as param, go off options purely

async function get (title, options = {}) {
    const noteSpec = { includeContent: true };
    const noteStore = getNotestore(options.production);
    if (options.guid) {
	Log.info(`Note.get: ${options.guid}`);
	return noteStore.getNoteWithResultSpec(options.guid, noteSpec)
	    .then(note => {
		if (!note && !options.nothrow) {
		    throw new Error(`note not found, ${options.guid}`);
		}
		return note;
	    });
    }
    return getNotebookByOptions(options)
	.then(notebook => {
	    Log.debug(`get from notebook: ${notebook.title}, ${notebook.guid}`);
	    const filter = { notebookGuid: notebook.guid };
	    const metaSpec = { includeTitle: true };
	    return noteStore.findNotesMetadata(filter, 0, 250, metaSpec);
	}).then(findResult => {
	    for (const metaNote of findResult.notes) {
		Log.debug(`note: ${metaNote.title}`);
		// TODO: check for duplicate named notes (option)
		if (metaNote.title === title) {
		    Log.debug(`match`);
		    return noteStore.getNoteWithResultSpec(metaNote.guid, noteSpec);
		}
	    }
	    return undefined;
	}).then(note => {
	    if (!note && !options.nothrow) throw new Error(`note not found, ${title}`);
	    return note;
	});
}

//
// options:
//   notebookGuid  only filter notes in this notebook
//   title         notes matching title exactly
//
// filterFunc:
//   user-defined metadata filter 
//
// to return all notes in a notebook, set options.notebookGuid = GUID, filterFunc = undefined
// to return a subset, filter by providing filterFunc
//
function getSome (options, filterFunc = undefined) {
    const noteStore = getNotestore(options.production);
    // strange way to do this. might be more than just 1 options that control notebook (like --default)
    // so, hasNotebookOption(options)
    return Promise.resolve(options.notebook ? getNotebookByOptions(options) : false)
	.then(notebook => {
	    let filter = {};
	    if (notebook) {
		Log.debug(`getSome from notebook: ${notebook.name}, ${notebook.guid}`);
		filter.notebookGuid = notebook.guid;
	    }
	    // include title
	    const metaSpec = { includeTitle: true };
	    return noteStore.findNotesMetadata(filter, 0, 250, metaSpec);
	}).then(findResult => findResult.notes.filter(note => {
	    Log.debug(`note: ${note.title}, guid: ${note.guid}`);
	    // TODO: check for duplicate named notes (option)
	    if (options.title) {
		if (options.title !== note.title) return false;
		Log.debug(`title match`);
	    }
	    const keep = !filterFunc || filterFunc(note);
	    Log.debug(keep ? 'keeping' : 'discarding');
	    return keep;
	})).then(metaNoteList => Promise.map(metaNoteList, note => {
	    const noteSpec = { includeContent: true };
	    return noteStore.getNoteWithResultSpec(note.guid, noteSpec);
	}, { concurrency: 1 }));
}

//
// options:
//   notebookGuid  only filter notes in this notebook
//   title         notes matching title exactly
//
// filterFunc:
//   user-defined metadata filter 
//
// to return all notes in a notebook, set options.notebookGuid = GUID, filterFunc = undefined
// to return a subset, filter by providing filterFunc
//
function getSomeMetadata (options, filterFunc = undefined) {
    const noteStore = getNotestore(options.production);
    // strange way to do this. might be more than just 1 options that control notebook (like --default)
    // so, hasNotebookOption(options)
    return Promise.resolve(options.notebook ? getNotebookByOptions(options) : false)
	.then(notebook => {
	    let filter = {};
	    if (notebook) {
		Log.debug(`all notes from notebook: ${notebook.name}, ${notebook.guid}`);
		filter.notebookGuid = notebook.guid;
	    }
	    // include title
	    const metaSpec = { includeTitle: true };
	    return noteStore.findNotesMetadata(filter, 0, 250, metaSpec);
	}).then(findResult => findResult.notes.filter(note => {
	    Log.debug(`note: ${note.title}, guid: ${note.guid}`);
	    // TODO: check for duplicate named notes (option)
	    if (options.title) {
		if (options.title !== note.title) return false;
		Log.debug(`title match`);
	    }
	    const keep = !filterFunc || filterFunc(note);
	    Log.debug(keep ? 'keeping' : 'discarding');
	    return keep;
	}));
}

//

function setContentBody (note, body) {
    Expect(note).is.an.Object();
    Expect(body).is.a.String();
    note.content = `<?xml version="1.0" encoding="UTF-8"?>` +
    `<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">` +
    `<en-note>${body}</en-note>`;
}


async function create (title, body, options = {}) {
    Expect(title).is.a.String();
    Expect(body).is.a.String();

    let note = {};
    return getNotebookByOptions(options)
	.then(notebook => {
	    note.title = title;
	    note.notebookGuid = notebook.guid;
	    /*
	     let noteAttributes;
	     noteAttributes.author = author;
	     noteAttributes.sourceURL = sourceURL;
	     note.attributes = noteAttributes;
	     */
	    if (options.content) {
		note.content = body;
	    } else {
		setContentBody(note, body);
	    }
	    return getNotestore(options.production).createNote(note);
	}); // .then(_ => note); // return the note.  why not? coz it's not really the note.
}

// file is enml content, passed as 'body' to Note.create

function createFromFile (filename, options = {}) {
    return Fs.readFile(filename)
	.then(content => {
	    // TODO: check if note exists
	    // TODO: specify notebook in options
	    return create(Path.basename(filename), content.toString(), options);
	});
}

//

function update (note, options = {}) {
    const noteStore = getNotestore(options.production);
    return noteStore.updateNote(note);
}

//

function splitForAppend (content, options) {
    const match = /<\/div>/g; // inserttion point is after the last closing div
    let prevResult;
    let result;
    do {
	prevResult = result;
	result = match.exec(content);
    } while (result != null);
    Expect(prevResult).is.ok();
    if (options.verbose) {
	console.log(`split at:\n${content.slice(prevResult.index, content.length)}`);
    }
    const pos = _.indexOf(content, '>', prevResult.index) + 1;
    return [content.slice(0, pos), content.slice(pos, content.length)];
}

//

function append (note, chunk, options = {}) {
    // TODO: xml-validate chunk
    const [before, after] = splitForAppend(note.content, options);
    note.content = `${before}${chunk}${after}`;
    return update(note, options);
}

//

function deleteNote (guid, options = {}) {
    return getNotestore(options.production).deleteNote(guid);
}

//

function getContent (guid, options = {}) {
    return getNotestore(options.production).getNoteContent(guid);
}

//

module.exports = {
    append,
    create,
    createFromFile,
    deleteNote,
    get,
    getContent,
    getNotebook,
//    getSome,
    getSomeMetadata,
    getWorksheetName,
    setContentBody,
    update
};
