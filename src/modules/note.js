/*
 * note.js
 */

'use strict';

//

const _                = require('lodash');
const Clues            = require('../clue-types');
const Debug            = require('debug')('note');
const Evernote         = require('evernote');
const EvernoteConfig   = require('../../data/evernote-config.json');
const Expect           = require('should/as-function');
const Fs               = require('fs-extra');
const Path             = require('path');
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

const Production = new NodeNote({ token: EvernoteConfig.production.token, sandbox: false });
const Sandbox    = new NodeNote({ token: EvernoteConfig.develop.token, sandbox: true });

//

function getNotestore (production) {
    return production ? Production.noteStore : Sandbox.noteStore;
}

//

function getNotebook (name, options = {}) {
    if (!name) return undefined;
    const noteStore = getNotestore(options.production);
    return noteStore.listNotebooks()
	.then(nbList => {
	    let match = false;
	    for(const nb of nbList) {
		if (options.relaxed) {
		    match = _.includes(nb.name, name);
		} else {
		    match = (nb.name === name);
		}
		if (match ) {
		    Debug(`notebook match: ${nb.name}`);
		    return nb;
		} else {
		    Debug(`not notebook: ${nb.name}`);
		}
	    }
	    Debug(`notebook not found, ${name}`);
	    return undefined;
	});
}

//

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
    Debug(`worksheet name: ${wsName}`);
    return wsName;
}

//

async function getNotebookGuidByOptions (options) {
    if (options.notebookGuid) return options.notebookGuid;
    if (!options.notebook) return undefined;
    return getNotebook(options.notebook, options)
	.then(nb => {
	    if (!nb) throw new Error(`no notebook matches: ${options.notebook}`);
	    return nb.guid;
	});
}

//

async function get (title, options = {}) {
    let filter = {};
    // todo: could combine a couple of these awaits into a chain

    filter.notebookGuid = await getNotebookGuidByOptions(options).catch(err => { throw err; });
    Debug(`notebookGuid: ${filter.notebookGuid}`);
    /*
     includeContent
     includeResourcesData
     includeResourcesRecognition
     includeResourcesAlternateData
     includeSharedNotes
     includeNoteAppDataValues
     includeResourceAppDataValues
     includeAccountLimits
     */
    const noteStore = getNotestore(options.production);

    let spec = {};
    if (options.content) {
	spec.includeContent = true;
    }

    let metaSpec = {};
    let result = await noteStore.findNotesMetadata(filter, 0, 250, metaSpec).catch(err => { throw err; });
    //Debug(Stringify(result));
    for (const metaNote of result.notes) {
	Debug(`GUID: ${metaNote.guid}`);
	let note = await noteStore.getNoteWithResultSpec(metaNote.guid, spec).catch(err => { throw err; });
	if (!note) continue;
	// TODO: check for duplicate named notes (option)
	if (note.title === title) {
	    Debug(`note match: ${note.title}`);
	    return note;
	} else {
	    Debug(`not note: ${note.title}`);
	}
    }
    throw new Error(`note not found, ${title}`);
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
    return getNotebookGuidByOptions(options)
	.then(guid => {
	    note.title = title;
	    note.notebookGuid = guid;
	    /*
	     let noteAttributes;
	     noteAttributes.author = author;
	     noteAttributes.sourceURL = sourceURL;
	     note.attributes = noteAttributes;
	     */
	    setContentBody(note, body);
	    return getNotestore(options.production).createNote(note);
	}).then(_ => note);
}

//

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

module.exports = {
    append,
    create,
    createFromFile,
    get,
    getNotebook,
    getWorksheetName,
    setContentBody,
    update
};
