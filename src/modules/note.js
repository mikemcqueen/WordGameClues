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
	    for(const nb of nbList) {
		if (_.includes(nb.name, name)) {
		    Debug(`notebook match: ${nb.name}`);
		    return nb;
		}
	    }
	    return undefined;
	});
}

//

async function get (nbName, title, options = {}) {
    const notebook = await getNotebook(nbName, options);
    if (!notebook) {
	console.log(`no notebook matches: ${nbName}`);
	return undefined;
    }
    //Debug(Stringify(notebook));
 
    let filter = {} ; // new Evernote.Types.NoteFilter();
    filter.notebookGuid = notebook.guid;
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
    let result = await NoteStore.findNotesMetadata(filter, 0, 250, spec);
    //Debug(Stringify(result));
    for (const note of result.notes) {
	console.log(`GUID: ${note.guid}`);
	// console.log(`Title: ${note.title}`);//  is null
	let note2  = await noteStore.getNoteWithResultSpec(note.guid, spec)
		.catch(err => {
		    console.log(err, err.stack);
		});
	if (!note2) continue;
	//console.log(`Title: ${note2.title}`);
	//Debug(Stringify(note.content));
	if (note2.title === title) {
	    Debug(`note match: ${note2.title}`);
	    return note2;
	}
    }
    return undefined;
}

//

function create (title, body, options = {}) {
    const noteStore = getNotestore(options.production);

    Expect(title).is.a.String();
    Expect(body).is.a.String();

    let note = {};

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
    update
};
