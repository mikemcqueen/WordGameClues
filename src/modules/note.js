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
//const Evernote = Production;
const NoteStore = Sandbox.noteStore;

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

async function get (title, options = {}) {
    if (!options.notebookGuid) {
	if (options.notebookName) {
	    const nb = await getNotebook(options.notebookName, options);
	    if (!nb) {
		throw new Error(`no notebook matches: ${nbName}`);
	    }
	    options.notebookGuid = nb.guid;
	}
    }
    //Debug(Stringify(notebook));
 
    let filter = {} ; // new Evernote.Types.NoteFilter();
    filter.notebookGuid = options.notebookGuid;
/*
includeContent	bool
includeResourcesData	bool
includeResourcesRecognition	bool
includeResourcesAlternateData	bool
includeSharedNotes	bool
includeNoteAppDataValues	bool
includeResourceAppDataValues	bool
includeAccountLimits
*/
    const noteStore = getNotestore(options.production);

    let spec = {};
    if (options.content) {
	spec.includeContent = true;
    }

    let metaSpec = {};
    let result = await NoteStore.findNotesMetadata(filter, 0, 250, metaSpec);
    //Debug(Stringify(result));
    for (const metaNote of result.notes) {
	Debug(`GUID: ${metaNote.guid}`);
	let note = await noteStore.getNoteWithResultSpec(metaNote.guid, spec)
		.catch(err => {
		    console.log(err, err.stack);
		});
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

function create (title, body, options = {}) {
    Expect(title).is.a.String();
    Expect(body).is.a.String();

    const noteStore = getNotestore(options.production);
    let note = {};

    note.title = title;
    if (options.notebookGuid) {
	note.notebookGuid = options.notebookGuid;
    }

    let noteAttributes;
    //noteAttributes.author = author;
    //noteAttributes.sourceURL = sourceURL;
    note.attributes = noteAttributes;
    note.content = '<?xml version="1.0" encoding="UTF-8"?>';
    note.content += '<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">';
    note.content += '<en-note>';
    note.content += body;
    note.content += '</en-note>';

    return noteStore.createNote(note);
}

//

function createFromFile (filename, options = {}) {
    return Fs.readFile(filename)
	.then(content => {
	    // TODO: check if note exists
	    // TODO: specify notebook?
	    return create(Path.basename(filename), content.toString(), options);
	});
}

//

function update (guid, title, body, options = {}) {
    const noteStore = getNotestore(options.production);

    let note = {};

    note.guid = guid;
    note.title = title;

    let noteAttributes;
    //noteAttributes.author = author;
    //noteAttributes.sourceURL = sourceURL;
    note.attributes = noteAttributes;
    note.content = '<?xml version="1.0" encoding="UTF-8"?>';
    note.content += '<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">';
    note.content += '<en-note>';
    note.content += body;
    note.content += '</en-note>';

    return noteStore.updateNote(note);
}

module.exports = {
    create,
    createFromFile,
    get,
    getNotebook,
    getNotebookName,
    update
};
