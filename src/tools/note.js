/*
 * note.js
 */

'use strict';

//

const _              = require('lodash');
const Clues          = require('../clue-types');
const Debug          = require('debug')('note');
const Evernote       = require('evernote');
const EvernoteConfig = require('../../data/evernote-config.json');
const Filter         = require('../modules/filter');
const Fs             = require('fs-extra');
const Getopt         = require('node-getopt');
const My             = require('../modules/util');
const Note           = require('../modules/note');
const NoteMaker      = require('../modules/note-make');
const NoteParser     = require('../modules/note-parse');
const Path           = require('path');
const Promise        = require('bluebird');
const Stringify      = require('stringify-object');
const Update         = require('../modules/update');

//


const Commands = { create, get, parse, update };
const Options = Getopt.create(_.concat(Clues.Options, [
    ['', 'create=FILE',   'create note from filter result file'],
    ['', 'get=TITLE',     'get (display) a note'],
    ['', 'notebook=NAME', 'specify notebook name'],
    ['', 'parse=TITLE',   'parse note into filter file format'],
//    ['', 'parse-file=FILE','parse note file into filter file format'],
    ['', 'json',          '  output in json (parse, parse-file)'],
    ['', 'production',    'use production note store'],
    ['', 'title=TITLE',   'specify note title (used with --create, --parse)'],
    ['', 'update[=NOTE]', 'update all results in worksheet, or a specific NOTE if specified'],
    ['h','help', '']
])).bindHelp(
    `usage: node note --${_.keys(Commands)} [options]\n[[OPTIONS]]\n`
);

//

const DATA_DIR      =  Path.normalize(`${Path.dirname(module.filename)}/../../data/`);

//

function usage (msg) {
    console.log(msg + '\n');
    Options.showHelp();
    process.exit(-1);
}

//

function NodeNote (options) {
    let client = new Evernote.Client({
	token: options.token,
	sandbox: options.sandbox
    });
    
    this.userStore = client.getUserStore();
    this.noteStore = client.getNoteStore();
}

//const Production = new NodeNote({token: EvernoteConfig.production.token, sandbox: false});
const Sandbox    = new NodeNote({ token: EvernoteConfig.develop.token, sandbox: true });
//const Evernote = Production;
const NoteStore = Sandbox.noteStore;

//

function getNotebook (name) {
    return NoteStore.listNotebooks()
	.then(nbList => {
	    for(const nb of nbList) {
		if (_.includes(nb.name, name)) return nb;
		Debug(`No match: ${nb.name}`);
		//Debug(Stringify(nb));
	    }
	    return undefined;
	});
}

//

async function getNote (nbName, title) {
    const notebook = await getNotebook(nbName);
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
    let spec = {
    };
    let result = await NoteStore.findNotesMetadata(filter, 0, 250, spec);
    Debug(Stringify(result));
    for (const note of result.notes) {
	console.log(`GUID: ${note.guid}`);
	// console.log(`Title: ${note.title}`);//  is null
	let note2  = await NoteStore.getNoteWithResultSpec(note.guid, spec)
		.catch(err => {
		    console.log(err, err.stack);
		});
	if (!note) continue;
	console.log(`Title: ${note2.title}`);
	//Debug(Stringify(note.content));
	if (note2.title === title) return note2;
    }
    return undefined;
}

function loadBody(filename) {
    return Fs.readFileSync(DATA_DIR + filename).toString();
}

//

function createNote (title, body) {
    let note = {};

    note.title = title;

    let noteAttributes;
    //noteAttributes.author = author;
//    noteAttributes.sourceURL = sourceURL;
    note.attributes = noteAttributes;
    note.content = '<?xml version="1.0" encoding="UTF-8"?>';
    note.content += '<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">';
    note.content += '<en-note>';
    note.content += body;
    note.content += '</en-note>';

    return NoteStore.createNote(note);
}

//

function updateNote (guid, title, body) {
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

    return NoteStore.updateNote(note);
}


async function createTestNote () {
    let body;

    body = loadBody('test-body.enml');
    return createNote('test', body);
}

//

async function updateTestNote (guid) {
    let body;

    body = loadBody('update-body.enml');
    return updateNote(guid, 'test', body);
}

//

function get (options) {
    options.content = true;
    return Note.get(options.get, options)
	.then(note => {
	    if (!note) usage(`note not found, ${options.get}`);
	    console.log(note.content);
	});
}

//

function create (options) {
    const title = options.title || Path.basename(filename);
    return NoteMaker.makeFromFile(options.create, { outerDiv: true })
	.then(body => Note.create(title, body))
	.then(note => {
	    console.log(Stringify(note));
	});
}

//

function getAndParse (noteName, options) {
    Debug(`getAndParse ${noteName}`);
    options.urls = true;
    options.clues = true;
    options.content = true;
    return Note.get(noteName, options)
	.then(note => Promise.all([note, NoteParser.parse(note.content, options)]));
}

//

function parse (options) {
    return getAndParse(options.parse, options)
	.then(([note, resultList]) => {
	    if (!note) usage(`note not found, ${options.parse}`);
	    if (_.isEmpty(resultList)) {
		console.log('no results');
	    } else {
		const fd = process.stdout.fd;
		Filter.dumpList(resultList, { fd });
	    }
	});
}

//

async function getParseSaveCommit (noteName, options) {
    const [note, resultList] = await getAndParse(noteName, options).catch(err => { throw err; });
    //Debug(resultList);
    const filepath = `${Clues.getDirectory(Clues.getByOptions(options))}/updates/${noteName}`;
    Debug(`saving ${noteName} to: ${filepath}`);
    return Fs.open(filepath, 'w')
	.then(fd => {
	    return Filter.dumpList(resultList, { fd });
	}).then(fd => Fs.close(fd))
	.then(_ => {
	    // no return = no await completion = OK
	    My.gitAddCommit(filepath, 'parsed live note');
	    return filepath;
	});
}

//

async function update (options) {
    Debug('update');
    const clueType = Clues.getByOptions(options);
    if (!_.isEmpty(options.update)) {
	const noteName = options.update;
	const nbName = Note.getWorksheetName(clueType);
	return Note.getNotebook(nbName, options)
	    .then(nb => {
		if (nb) {
		    options.notebookGuid = nb.guid;
		} else {
		    if (options.production || !options.default) {
			throw new Error(`notebook not found, ${nbName}`);
		    }
		}
		return getParseSaveCommit(noteName, options);
	    }).then(path => Update.updateFromFile(path, options));
    } else {
	const nbName = Note.getWorksheetName(clueType);
	console.log(`notebook: ${nbName}`);
    }
}

//

async function main () {
    const opt = Options.parseSystem();
    const options = opt.options;
    if (opt.argv.length > 0) {
	usage(`invalid non-option parameter(s) supplied, ${opt.argv}`);
    }
    if (options.production) console.log('---PRODUCTION--');

    let cmd;
    for (const key of _.keys(Commands)) {
	if (_.has(options, key)) {
	    cmd = key;
	    Debug(`command: ${key}`);
	    break;
	} else Debug(`not: ${key}`);
    }
    if (!cmd) usage(`missing command`);
    return Commands[cmd](options);
}

//

main().catch(err => {
    console.error(err, err.stack);
});
