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
    ['v','verbose',       'more noise'],
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

function get (options) {
    options.content = true;
    return Note.get(options.get, options)
	.then(note => {
	    if (!note) usage(`note not found, ${options.get}`);
	    console.log(note.content);
	});
}

//

function old_create (options) {
    const title = options.title;
    
    return NoteMaker.makeFromFilterFile(options.create, { outerDiv: true })
	.then(body => {
	    Debug(`body: ${body}`);
	    Note.create(title, body, options);
	}).then(note => {
	    console.log(Stringify(note));
	});
}

//

function create (options) {
    const title = options.title;
    
    const list = Filter.parseFile(options.create, { urls: true, clues: true });
    Debug(`filterList: ${list}`);
    const body = NoteMaker.makeFromFilterList(list, { outerDiv: true });
    Debug(`body: ${body}`);
    return Note.create(title, body, options)
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
		return console.log('no results');
	    } else {
		const fd = process.stdout.fd;
		return Filter.dumpList(resultList, { fd });
		//console.log(`${Stringify(resultList)}`);
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
