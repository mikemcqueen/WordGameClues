/*
 * note.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('note');
const Evernote         = require('evernote');
const EvernoteConfig   = require('../../data/evernote-config.json');
const Expect           = require('should/as-function');
const Fs               = require('fs-extra');
const Path             = require('path');
const Stringify        = require('stringify-object');
 
//

const DATA_DIR      =  Path.normalize(`${Path.dirname(module.filename)}/../../data/`);

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
    const noteStore = getNotestore(options.production);
    return noteStore.listNotebooks()
	.then(nbList => {
	    let match = false;
	    for(const nb of nbList) {
		if (options.strict) {
		    match = (nb.name === name);
		} else {
		    match = _.includes(nb.name, name);
		}
		if (match ) {
		    Debug(`notebook match: ${nb.name}`);
		    return nb;
		}
	    }
	    return undefined;
	});
}

//

function getNotebookName (noteName) {
    Expect(noteName[0]).equals('p');
    Expect(noteName[1]).is.a.Number();
    return `Worksheets.${noteName.slice(0, 2)}`;
}

//

async function getOptionsNotebookGuid (options) {
    if (options.notebookGuid) return options.notebookGuid;
    if (!options.notebookName) return undefined;
    return getNotebook(options.notebookName, options)
	.then(nb => {
	    if (!nb) throw new Error(`no notebook matches: ${options.notebookName}`);
	    return nb.guid;
	});
}

//

async function get (title, options = {}) {
    let filter = {};
    // todo: could combine a couple of these awaits into a chain

    filter.notebookGuid = await getOptionsNotebookGuid(options).catch(err => { throw err; });
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
	}
    }
    return undefined;
}

//

async function create (title, body, options = {}) {
    Expect(title).is.a.String();
    Expect(body).is.a.String();

    let note = {};
    return getOptionsNotebookGuid(options)
	.then(guid => {
	    note.title = title;
	    note.notebookGuid = guid;
	    /*
	     let noteAttributes;
	     noteAttributes.author = author;
	     noteAttributes.sourceURL = sourceURL;
	     note.attributes = noteAttributes;
	     */
	    note.content = '<?xml version="1.0" encoding="UTF-8"?>';
	    note.content += '<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">';
	    note.content += '<en-note>';
	    note.content += body;
	    note.content += '</en-note>';
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
    getNotebookName,
    update
};
