/*
 * note.js
 */

'use strict';

//

const _              = require('lodash');
const ClueManager    = require('../clue-manager');
const Clues          = require('../clue-types');
const Debug          = require('debug')('note');
const Evernote       = require('evernote');
const EvernoteConfig = require('../../data/evernote-config.json');
const Expect         = require('should/as-function');
const Filter         = require('../modules/filter');
const Fs             = require('fs-extra');
const Getopt         = require('node-getopt');
const Markdown       = require('../modules/markdown');
const My             = require('../modules/util');
const Note           = require('../modules/note');
const NoteMaker      = require('../modules/note-make');
const NoteParser     = require('../modules/note-parse');
const Path           = require('path');
const Promise        = require('bluebird');
const Stringify      = require('stringify-object');
const Update         = require('../modules/update');

//

const Commands = { create, get, parse, update, count };
const Options = Getopt.create(_.concat(Clues.Options, [
    ['', 'count=NAME',    'count sources/clues/urls in a note'],
    ['', 'create=FILE',   'create note from filter result file'],
    ['', 'force',         'force execution: update single note with removed clue'],
    ['', 'get=TITLE',     'get (display) a note'],
    ['', 'notebook=NAME', 'specify notebook name'],
    ['', 'parse=TITLE',   'parse note into filter file format'],
    ['', 'compare',       '  parse old + dom  and show differences'],
    ['', 'dom',           '  parse dom (change this to old: use old parse method)'],
//    ['', 'parse-file=FILE','parse note file into filter file format'],
    ['', 'json',          '  output in json (parse, parse-file)'],
    ['', 'production',    'use production note store'],
    ['', 'quiet',         'less noise'],
    ['', 'title=TITLE',   'specify note title (used with --create, --parse)'],
    ['', 'update[=NOTE]', 'update all results in worksheet, or a specific NOTE if specified'],
    ['', 'save',          '  save cluelist files (used with --update)'],
    ['v','verbose',       'more noise'],
    ['h','help', '']
])).bindHelp(
    `usage: node note --${_.keys(Commands)} [options]\n[[OPTIONS]]\n`
);

//

const DATA_DIR =  Path.normalize(`${Path.dirname(module.filename)}/../../data/`);

//

function usage (msg) {
    console.log(msg + '\n');
    Options.showHelp();
    process.exit(-1);
}

//

async function get (options) {
    options.content = true;
    return Note.get(options.get, options)
	.then(note => {
	    if (!options.quiet) {
		console.log(note.content);
	    }
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
	    if (!options.quiet) {
		console.log(Stringify(note));
	    }
	});
}

//

async function create (options) {
    const title = options.title;
    if (!title) usage('--title is required');
    const list = Filter.parseFile(options.create, { urls: true, clues: true });
    //Debug(`filterList: ${list}`);
    const body = NoteMaker.makeFromFilterList(list, { outerDiv: true }, options);
    //Debug(`body: ${body}`);
    return Note.create(title, body, options)
	.then(note => {
	    if (!options.quiet) {
		console.log(Stringify(note));
	    }
	});
}

//

async function getAndParse (noteName, options) {
    Debug(`getAndParse ${noteName}`);
    options.urls = true;
    options.clues = true;
    options.content = true;
    return Note.get(noteName, options)
	.then(note => [note, NoteParser.parse(note.content, options)]);
}

//

async function saveLines (lines, path) {
    return new Promise((resolve, reject) => {
	// what on earth happens if this fails.
	const stream = Fs.createWriteStream(path);
	stream.on('open', _ => {
	    for (const line of lines) {
		stream.write(line + '\n');
	    }
	    stream.end();
	}).on('close', _ => {
	    resolve(path);
	}).on('error', err => {
	    reject(err);
	});
    });
}

//

async function saveLinesAndList (filename, lines, filterList) {
    const path = Path.dirname(module.filename) +`/tmp/${filename}`;
    return Promise.join(saveLines(lines, path + '-lines'),
			Filter.save(filterList, path + '-list'),
			(linePath, listPath) => [linePath, listPath]);

}

//

async function parse (options) {
    if (options.compare) {
	return Note.get(options.parse, options)
	    .then(note => {
		let lines = NoteParser.parseDom(note.content, options);
		options.urls = true;
		options.clues = true;
		options.content = true;
		let filterList = NoteParser.parse(note.content, options);
		return saveLinesAndList(note.title, lines, filterList);
	    }).then(([linePath, listPath]) => {
		console.log(`lines: ${linePath}`);
		console.log(`list:  ${listPath}`);
	    });
    } else if (options.dom) {
	return Note.get(options.parse, options)
	    .then(note => {
		let lines = NoteParser.parseDom(note.content, options);
		lines.forEach(line => console.log(line));
	    });
    } else {
	return getAndParse(options.parse, options)
	    .then(([note, resultList]) => {
		if (_.isEmpty(resultList)) {
		    return console.log('no results');
		} else {
		    const fd = process.stdout.fd;
		    //console.log(`${Stringify(resultList)}\n------`);
		    return Filter.dumpList(resultList, { json: options.json, /*all: true,*/ fd });
		}
	    });
    }
}

//

function getParseSaveCommit (noteName, options) {
    return getAndParse(noteName, options)
	.then(([note, resultList]) => {
	    return Filter.saveAddCommit(noteName, resultList, options);
	});
}

//

function updateOneClue (noteName, options) {
    return getAndParse(noteName, options)
	.then(([note, resultList]) => {
	    const removedClueMap = Filter.getRemovedClues(resultList);
	    if (!_.isEmpty(removedClueMap)) {
		if (!options.force) {
		    for (const key of removedClueMap.keys()) {
			console.log(`removed clue: ${key} -> ${Array.from(removedClueMap.get(key).values())}`);
		    }
		    usage(`can't update single note with removed clues (${removedClueMap.size})`);
		}
		removeAllClues(removedClueMap, options);
	    }
	    return Filter.saveAddCommit(noteName, resultList, options);
	}).then(path => Update.updateFromFile(path, options));
}

//

async function getSomeAndParse (options, filterFunc) {
    Expect(filterFunc).is.a.Function();
    Debug(`getSomeAndParse`);
    options.urls = true;
    options.clues = true;
    options.content = true;
    return Note.getSome(options, filterFunc)
	.then(noteList => {
	    if (_.isEmpty(noteList)) throw new Error('no notes found');
	    Debug(`found ${noteList.length} notes`);
	    // map list of notes to list of { note, filterList }
	    // use _ prefix because filterFunc may likely have conflicting 'note' param
	    return noteList.map(note => {
		return { note, filterList: NoteParser.parse(note.content, options) };
	    });
	});
}

// to, from: are Map() type. must be the builtin class type, not generic objects

function addRemovedClues (to, from) {
    for (let source of from.keys()) {
	Debug(`${source} has ${_.size(from.get(source))} removed clue(s)`);
	if (!to.has(source)) {
	    to.set(source, from.get(source));
	} else {
	    for (let clueName of from.get(source).values()) {
		to.get(source).add(clueName);
	    }
	}
    }
}

//

function getAllRemovedClues (resultList) {
    let allRemovedClues = new Map();
    for (let result of resultList) {
	console.log(`note: ${result.note.title}`);
	const removedClues = Filter.getRemovedClues(result.filterList);
	if (!_.isEmpty(removedClues)) {
	    addRemovedClues(allRemovedClues, removedClues);
	    Debug(`found ${_.size(removedClues)} source(s) with removed clues` +
		  `, total unique: ${_.size(allRemovedClues)}`);
	}
    }
    return allRemovedClues;
}

//

function removeAllClues (removedClues, options = {}) {
    Debug('removeAllClues');
    let total = 0;
    for (let source of removedClues.keys()) {
	let nameCsv = Markdown.hasSourcePrefix(source)
		? source.slice(1, source.length) : source;
	const nameList = nameCsv.split(',').sort();
	const result = ClueManager.getCountListArrays(nameCsv, { remove: true });
	if (!result) {
	    Debug(`no matches for source: ${source}`);
	    continue;
	}
	// for each clue word in set
	for (let clue of removedClues.get(source).keys()) {
	    let removed = ClueManager.addRemoveOrReject(
		{ remove: clue }, nameList, result.addRemoveSet, options);
	    if (removed > 0) {
		Debug(`removed ${removed} instance(s) of [${clue}] : ${source}`);
	    }
	    total += removed;
	}
    }
    Debug(`total removed: ${total}`);
    return total;
}

// return array of save paths

async function saveAddCommitAll (resultList, options) {
    Expect(resultList).is.an.Array();
    Expect(options).is.an.Object();
    return Promise.map(resultList, result => {
	return Filter.saveAddCommit(result.note.title, result.filterList, options);
    }, { concurrency: 1 });
}

//

async function updateAllClues (options) {
    const prefix = Clues.getShorthand(Clues.getByOptions(options));
    const bignoteSuffix = '.article';
    return getSomeAndParse(options, note => {
	return _.startsWith(note.title, prefix) && !_.endsWith(note.title, bignoteSuffix);
    }).then(resultList => {
	let allRemovedClues = getAllRemovedClues(resultList);
	removeAllClues(allRemovedClues, options);
	return saveAddCommitAll(resultList, options);
    }).then(pathList => {
	// NOTE: updateFromPathList should do something with options.save, don't pass it to
	// updateFromFile, just call save at the end.
	return Update.updateFromPathList(pathList, options);
    });
}

//

async function update (options) {
    // but i could support multiple: just call updateOneClue in a for() loop
    // but not exactly that simple due to removed clues
    if (_.isArray(options.update)) usage('multiple --updates not yet supported');
    Debug(`update, ${options.update}`);
    // TODO: what if noteobok is undefined
    return Note.getNotebook(options.notebook, options)
	.then(nb => {
	    if (!nb && !options.default) {
		throw new Error(`notebook not found, ${options.notebook}`);
	    }
	    console.log(`notebook: ${nb.title}, guid: ${nb.guid}`);
	    options.notebookGuid = nb.guid;

	    ClueManager.loadAllClues({ clues: Clues.getByOptions(options) });

	    if (_.isEmpty(options.update)) {
		// no note name supplied, update all notes
		return updateAllClues(options);
	    }
	    // update supplied note name(s)
	    return updateOneClue(options.update, options);
	});
}

//

async function count (options) {
    Debug('count');
    // options.notebook = options.notebook || Note.getWorksheetName(Clues.getByOptions(options));
    return getAndParse(options.count, options)
	.then(([note, filterList]) => {
	    const count = Filter.count(filterList);
	    console.log(`sources: ${count.sources}, urls: ${count.urls}, clues: ${count.clues}`);
	});
}

//

async function main () {
    const opt = Options.parseSystem();
    const options = opt.options;
    if (opt.argv.length > 0) {
	usage(`invalid non-option parameter(s) supplied, ${opt.argv}`);
    }
    if (options.production) console.log('---PRODUCTION--');

    options.notebook = options.notebook || Note.getWorksheetName(Clues.getByOptions(options));

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
    process.exit(-1);
});
