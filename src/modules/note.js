/*
 * note.js
 */

'use strict';

//

const _                = require('lodash');
const Clues            = require('./clue-types');
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
        token:   options.token,
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
    Log.debug('listing');
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
        noteName = Clues.getShorthand(noteName); // convert clueType to, e.g, 'p8s'
	Log.info(`noteName: ${noteName}`);
    } else {
        const appleExpr = /p[0-9](?:\.[0-9])?s?/;
        const result = appleExpr.exec(noteName);
        if (!result || (result.index !== 0)) return undefined;
    }
    Expect(noteName).is.a.String();
    Expect(noteName.charAt(0)).is.equal('p');
//    Expect(_.toNumber(noteName.charAt(1))).is.above(0);
    let count = 2;
    if (noteName.charAt(count) === '.') {
	count += 1;
	if (Clues.isVarietyDigit(noteName.charAt(count))) count += 1;
	if (Clues.isVarietyDigit(noteName.charAt(count))) count += 1;
	Expect(count > 3).is.true();
    }
    if (noteName.charAt(count) === 's') count += 1;
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

async function getNotebookByOptions (options = {}) {
    Log.debug(`++getNotebookByOptions, guid:${options.notebookGuid}`);
    if (options.notebookGuid) return getNotebookByGuid(options.notebookGuid, options);
    if (!options.notebook) {
	throw new Error('No notebook specified.');
        //Log.debug(`NO NOTEBOOK - THROW?`);
	//return undefined;
    }
    Log.debug(`options : ${Stringify(options)}`);
    return getNotebook(options.notebook, options)
        .then(nb => {
            if (!nb) throw new Error(`no notebook matches: ${options.notebook}`);
            Log.debug(`--getNotebookByOptions, guid:${nb.guid}`);
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

function get (title, options = {}) {
    return getNoteGuid(title, options)
	.then(guid => {
            Log.info(`note guid: ${guid}`);
	    if (!guid) return undefined;
	    return getNotestore(options.production).getNoteWithResultSpec(guid, {});
	}).then(note => {
            if (!note) return undefined;
	    Log.debug(`note header: ${Stringify(note)}`);
	    if (options.updated_after) {
		const noteLastUpdated = new Date(note.updated);
		const updatedAfter = new Date(options.updated_after);
 		Log.debug(`noteLastUpdated: ${noteLastUpdated}`);
 		Log.debug(`updatedAfter: ${updatedAfter}`);
		Log.debug(`note ${noteLastUpdated < updatedAfter ? "is current" : "needs updating"}`);
		if (noteLastUpdated < updatedAfter) return false; // skip: note is current
	    }
	    return getNotestore(options.production).getNoteWithResultSpec(note.guid, { includeContent: true });
	}).then(note => {
            if (_.isUndefined(note) && !options.nothrow) {
                throw new Error(`note not found, title: ${title}, guid: ${options.guid}`);
            }
            return note;
        });
}

async function getNoteGuid (title, options = {}) {
    if (options.guid) return options.guid;
    return getNotebookByOptions(options)
        .then(notebook => {
            Log.debug(`get from notebook: ${notebook.name}, ${notebook.guid}`);
            const filter = { notebookGuid: notebook.guid };
            const metaSpec = { includeTitle: true ,  includeResourcesData: true };
	    const noteStore = getNotestore(options.production);
            return noteStore.findNotesMetadata(filter, 0, 250, metaSpec);
        }).then(findResult => {
            for (const metaNote of findResult.notes) {
                Log.debug(`note: ${metaNote.title}`);
                // TODO: check for duplicate named notes (option)
                if (metaNote.title === title) {
                    Log.debug(`match`);
		    Log.debug(`meta: ${Stringify(metaNote)}`);
		    return metaNote.guid;
                }
            }
            return undefined;
        });
}

//
// options:
//   notebook
//   notebookGuid  only filter notes in this notebook
//
// to return metadata for all notes in a notebook, set options.notebookGuid = GUID
//
function getSomeMetadata (options) {
    // strange way to do this. might be more than just 1 options that control notebook (like --default)
    // so, hasNotebookOption(options)
    return Promise.resolve(options.notebook ? getNotebookByOptions(options) : false)
        .then(notebook => {
            let filter = {};
            if (notebook) {
                Log.info(`all notes from notebook: ${notebook.name}, ${notebook.guid}`);
                filter.notebookGuid = notebook.guid;
            }
            // include title
            const metaSpec = { includeTitle: true };
            return getNotestore(options.production).findNotesMetadata(filter, 0, 250, metaSpec);
        }).then(findResult => findResult.notes);
}

// params:
//   note only contains metadata (no content)

function chooser (note, options) {
    Log.debug(`chooser, note: ${note.title}, guid: ${note.guid}`);
    if (options.all) return true;

    // TODO: check for duplicate named notes (option)
    if (options.title) {
        if (options.title !== note.title) return false;
        Log.debug(`title match`);
        return true;
    }
    
    const prefix = options.match || Clues.getLonghand(Clues.getByOptions(options)) + '.';
    Log.debug(`prefix: ${prefix}`);
    // if --match is specified, choose all notes that match
    // otherwise, don't choose 'article' suffixed notes (TODO: unless --article is specified)
    // and don't choose 'remaining' suffixed notes (unless --remaining is specified)

    return _.startsWith(note.title, prefix) && (options.match ||
        (!_.endsWith(note.title, 'article') &&
        (!options.remaining || !_.endsWith(note.title, 'remaining'))));
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

    Log.debug(`note.create`);
    let note = {};
    return getNotebookByOptions(options)
        .then(notebook => {
            Log.debug(`nb guid: ${notebook.guid}`);
            Log.debug(`body: ${body}`);
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
    chooser,
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
