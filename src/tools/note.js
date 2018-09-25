/*
 * note.js
 */

'use strict';

//

const _              = require('lodash');
const ClueManager    = require('../modules/clue-manager');
const Clues          = require('../modules/clue-types');
const Debug          = require('debug')('note');
const Duration       = require('duration');
const Evernote       = require('evernote');
const EvernoteConfig = require('../../data/evernote-config.json');
const Expect         = require('should/as-function');
const Filter         = require('../modules/filter');
const Fs             = require('fs-extra');
const Getopt         = require('node-getopt');
const Markdown       = require('../modules/markdown');
const Log            = require('../modules/log')('note');
const My             = require('../modules/util');
const Note           = require('../modules/note');
const NoteMaker      = require('../modules/note-make');
const NoteParser     = require('../modules/note-parse');
const Options        = require('../modules/options');
const Path           = require('path');
const PrettyMs       = require('pretty-ms');
const Promise        = require('bluebird');
const Stringify      = require('stringify-object');
const Update         = require('../modules/update');
const NoteValidator  = require('../modules/note-validate');

//

const Commands = { count, create, get, parse, update, validate };
const CmdLineOptions = Getopt.create(_.concat(Clues.Options, [
    ['', 'count=NAME',      'count sources/clues/urls in a note'],
    ['', 'create=FILE',     'create note from file (default: filter result file)'],
    ['', 'text',            '  create note from any text file'],
    ['', 'point-size=SIZE', '  font point size'],
    ['', 'get=TITLE',       'get (display) a note'],
    ['', 'parse=TITLE',     'parse note into filter file format'],
//    ['', 'parse-file=FILE','parse note file into filter file format'],
    ['', 'compare',         '  parse old + dom  and show differences'],
    ['', 'old',             '  use old parse method (use with parse)'],
    ['', 'json',            '  output in json (use with parse, parse-file)'],
    ['', 'update[=NOTE]',   'update all results in worksheet, or a specific NOTE if specified'],
// TOOD: change PREFIX to REGEX
    ['', 'match=PREFIX',    '  update notes matching title PREFIX (used with --update)'],
    ['', 'force-update',    '  update single note with removed clue (used with --update)'],
    ['', 'always',          '  update notes that don\'t need updating'],
    ['', 'save',            '  save cluelist files (used with --update)'],
    ['', 'dry-run',         '  show changes only (used with --update)'],
    ['', 'from-fs',         '  update from filesystem (used with --update, --validate)'],
    ['d','download-only',   '  download only (used with --update)'],
    ['y','yes-mode',        '  yes mode (used with update)'],
    ['', 'validate[=NOTE]', 'validate sources in note file (requires --from-fs).'],
    ['', 'notebook=NAME',   'specify notebook name'],
    ['', 'production',      'use production note store'],
    ['', 'quiet',           'less noise'],
    ['', 'title=TITLE',     'specify note title (used with --create, --parse)'],
    ['v','verbose',         'more noise'],
    ['h','help', '']
])).bindHelp(
    `usage: node note --${_.keys(Commands)} [options]\n[[OPTIONS]]\n`
);

//

const DATA_DIR =  Path.normalize(`${Path.dirname(module.filename)}/../../data/`);

//

function usage (msg) {
    console.log(msg + '\n');
    CmdLineOptions.showHelp();
    process.exit(-1);
}

//

async function count (options) {
    Log.info('count');
    return getAndParse(options.count, options)
        .then(result => {
            const count = Filter.count(result.filterList);
            Log(`sources: ${count.sources}, urls: ${count.urls}, clues: ${count.clues}`);
        });
}

//

function createFromTextFile (options) {
    const title = options.title || Path.basename(options.create);
    return NoteMaker.makeFromFilterFile(options.create, { outerDiv: true })
        .then(body => {
            Debug(`body: ${body}`);
            return Note.create(title, body, options);
        }).then(note => {
            if (!options.quiet) {
                console.log(Stringify(note));
            }
        });
}

//

async function create (options) {
    if (options.text) return createFromTextFile(options);

    const title = options.title;
    if (!title) usage('--title is required');
    const list = Filter.parseFile(options.create, options);
    // NOTE: not passing cmd line options here
    const body = NoteMaker.makeFromFilterList(list, { outerDiv: true });
    return Note.create(title, body, options)
        .then(note => {
            if (!options.quiet) {
                console.log(Stringify(note));
            }
        });
}

//

async function get (options) {
    options.content = true;
    return Note.get(options.get, options)
        .then(note => {
            // get ignores --quiet
            console.log(note.content);
        });
}

//

async function getAndParse (noteName, options = {}) {
    Log.info(`getAndParse ${noteName} : ${options.guid}`);
    return Note.get(noteName, options)
        .then(note => {
            if (!note) return undefined;
            const parseOptions = Object.assign(_.clone(options), { urls: true, clues: true });
            return {
                note,
                filterList: Filter.parseLines(NoteParser.parseDom(note.content, parseOptions))
            };
        });
}

/*
//

async function saveLines (lines, path) {
    return new Promise((resolve, reject) => {
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
 */

//

async function saveLinesAndList (filename, lines, filterList) {
    const path = Path.dirname(module.filename) +`/tmp/${filename}`;
    //saveLines(lines, path + '-lines'),
    return Promise.join(Filter.save(Filter.parseLines(lines), path + '-lines'),
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
                let filterList = NoteParser.oldParse(note.content, options);
                return saveLinesAndList(note.title, lines, filterList);
            }).then(([linePath, listPath]) => {
                console.log(`lines: ${linePath}`);
                console.log(`list:  ${listPath}`);
            });
    } else if (options.old) {
        return Note.get(options.parse, options)
            .then(note => {
                options.urls = true;
                options.clues = true;
                options.content = true;
                let filterList = NoteParser.parse(note.content, options);
                if (_.isEmpty(filterList)) {
                    return console.log('no results');
                } else {
                    return Filter.dumpList(filterList, {
                        json: options.json,
                        fd: process.stdout.fd
                    });
                }
            });
    } else {
        return getAndParse(options.parse, options)
            .then(result => {
                if (_.isEmpty(result.filterList)) {
                    return console.log('no results');
                } else {
                    return Filter.dumpList(result.filterList, {
                        json: options.json,
                        fd: process.stdout.fd
                    });
                }
            });
    }
}

//

/*
function getParseSaveCommit (noteName, options) {
    return getAndParse(noteName, options)
        .then(result => {
            return Filter.saveAddCommit(noteName, result.filterList, options);
        });
}
 */

// pullOneWorksheet

function getLastUpdateTime (noteName, options) {
    return My.fstat(Filter.getUpdateFilePath(noteName, options))
        .then(stats => {
            Log.info(`file: stats: ${Stringify(stats)}`);
            return stats ? stats.mtimeMs : 0;
        });
}

//

function loadParseSaveOneWorksheet (noteName, options) {
    return Promise.resolve(options.always ? 0 : getLastUpdateTime(noteName, options))
        .then(lastUpdateTime => {
            const getOptions = Object.assign(_.clone(options), { updated_after: lastUpdateTime });
            return getAndParse(noteName, getOptions);
        }).then(result => {
            if (!result) return undefined;
            const removedClueMap = Filter.getRemovedClues(result.filterList);
            if (_.isEmpty(removedClueMap)) {
                Log.info(`no removed clues`);
            } else {
                if (!options.force_update) {
                    for (const key of removedClueMap.keys()) {
                        console.log(`removed clue: ${key} -> ${Array.from(removedClueMap.get(key).values())}`);
                    }
                    usage(`can't update single note with removed clues (${removedClueMap.size})`);
                }
                removeAllClues(removedClueMap, options);
            }
            return Filter.saveAddCommit(noteName, result.filterList, options);
        });
}

//

async function getSomeFilteredMetadata (options, filterFunc) {
    Expect(filterFunc).is.a.Function();
    Log.debug(`getSomeAndParse`);
    options.urls = true;
    options.clues = true;
    options.content = true;
    return Note.getSomeMetadata(options)
        .filter(metadata => filterFunc(metadata, options))
        .then(metadataList => {
            if (_.isEmpty(metadataList)) usage('no notes found');
            Log.info(`found ${metadataList.length} notes`);
            return metadataList;
        });
}

//

async function getSomeAndParse (options, filterFunc) {
    Expect(filterFunc).is.a.Function();
    Log.debug(`getSomeAndParse`);
    options.urls = true;
    options.clues = true;
    options.content = true;
    // NOTE: the problem with this approach is that we load all notes
    // first, then if one note craps on parsing, we just wasted all
    // of that download time/bandwidth. should download, parse and
    // *process* each note serially (eventually: in parallel)
    const result = [];
    return Note.getSomeMetadata(options)
        .filter(metadata => filterFunc(metadata, options))
        .then(metadataList => {
            if (_.isEmpty(metadataList)) usage('no notes found');
            Log.info(`found ${metadataList.length} notes`);
            return metadataList;
        }).each(metadata => { // TODO: switch to .map
            Log.info(`meta note: ${metadata.title}`);
            options.guid = metadata.guid;
            // nested chain because .each appears to continue to next
            // element upon completion of the first promise (Note.get)
            return Note.get(null, options)
                .then(note => {
                    if (!note) return undefined;
                    Log.info(`parsing note ${note.title}`);
                    result.push({
                        note,
                        filterList: Filter.parseLines(NoteParser.parseDom(note.content, options), options)
                    });
                    return undefined;
                });
        }).then(_ => result);
}

// to, from: are Map() type. must be the builtin class type, not generic objects

function addRemovedClues (to, from) {
    for (let source of from.keys()) {
        Log.debug(`${source} has ${_.size(from.get(source))} removed clue(s)`);
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
        Log.info(`removing clues in note: ${result.note.title}`);
        const removedClues = Filter.getRemovedClues(result.filterList);
        if (!_.isEmpty(removedClues)) {
            addRemovedClues(allRemovedClues, removedClues);
            Log.debug(`found ${_.size(removedClues)} source(s) with removed clues` +
                  `, total unique: ${_.size(allRemovedClues)}`);
        }
    }
    return allRemovedClues;
}

//

function removeAllClues (removedClues, options = {}) {
    Log.debug('removeAllClues');
    let total = 0;
    for (let source of removedClues.keys()) {
        let nameCsv = Markdown.hasSourcePrefix(source) ? source.slice(1, source.length) : source;
        const nameList = nameCsv.split(',').sort();
        const result = ClueManager.getCountListArrays(nameCsv, { remove: true });
        if (!result) {
            Log.debug(`no matches for source: ${source}`);
            continue;
        }
        // for each clue word in set
        for (let clue of removedClues.get(source).keys()) {
            let removed = ClueManager.addRemoveOrReject(
                { remove: clue }, nameList, result.addRemoveSet, options);
            if (removed > 0) {
                Log.debug(`removed ${removed} instance(s) of ${source} -> ${clue}`);
            }
            total += removed;
        }
    }
    Log.debug(`total removed: ${total}`);
    return total;
}

// return array of save paths

async function saveAddCommitAll (resultList, options) {
    Expect(resultList).is.an.Array();
    Expect(options).is.an.Object();
    return Promise.map(resultList, result => Filter.saveAddCommit(result.note.title, result.filterList, options), { concurrency: 2 });
}

//

function loadParseSaveAllWorksheets (options) {
    return getSomeFilteredMetadata(options, Note.chooser)
        .then(metadataList => {
            if (_.isEmpty(metadataList)) return undefined;
	    /// this should be a function: parseAndSaveSomeMetadata(metadataList, options)
            return Promise.mapSeries(metadataList, metadata => { // map from metadata to { parsed result, save path
                Log.info(`note: ${metadata.title} : ${metadata.guid}`);
                return Promise.join(metadata.guid, options.always ? 0 : getLastUpdateTime(metadata.title, options),
		    (guid, lastUpdatedTime) => {
			const getOptions = Object.assign(_.clone(options), {
			    guid: metadata.guid,
			    updated_after: lastUpdatedTime
			});
			// get, parse, then save. return {result, path }
			return getAndParse(null, getOptions)
			    .then(result => {
				if (!result) return {};
				const path = Filter.saveAddCommit(result.note.title, result.filterList, options);
				return Promise.join(result, path, (result, path) => {
				    return { result, path };
				});
			    });
		    });
	    });
        }).then(resultPathList => {
	    const resultList = resultPathList
		  .filter(resultPath =>  resultPath.result)
		  .map(resultPath => resultPath.result);
	    Log.debug(`resultList: ${_.size(resultList)}`);
            let allRemovedClues = getAllRemovedClues(resultList);
            removeAllClues(allRemovedClues, options);
            // return path list
            return resultPathList
		.filter(resultPath =>  resultPath.path)
		.map(resultPath => resultPath.path);
        });
}

//

function getAllUpdateFilePaths (options) {
    return Note.getSomeMetadata(options)
        .filter(metadata => Note.chooser(metadata, options))
        .then(metadataList => {
            if (_.isEmpty(metadataList)) usage('no notes found');
            Log.info(`found ${metadataList.length} notes`);
            return metadataList;
        }).map(metadata => {
            Log.info(`meta note: ${metadata.title}`);
            return Filter.getUpdateFilePath(metadata.title, options);
        });
}

//

async function update (options) {
    // but i could support multiple: just call updateOneClue in a for() loop
    // but not exactly that simple due to removed clues
    if (_.isArray(options.update)) usage('multiple --updates not (yet) supported');
    Log.info(`update, ${options.update || 'all'}`);
    // TODO: what if noteobok is undefined
    return Note.getNotebook(options.notebook, options)
        .then(nb => {
            if (!nb && !options.default) {
                usage(`notebook not found, ${options.notebook}`);
            }
            Log.info(`notebook, name: ${nb.name}, guid: ${nb.guid}`);
            options.notebookGuid = nb.guid;

            ClueManager.loadAllClues({ clues: Clues.getByOptions(options) });

            if (_.isEmpty(options.update)) {
                // no note name supplied, update all notes; return path list
                return options.from_fs
                    ? getAllUpdateFilePaths(options)
                    : loadParseSaveAllWorksheets(options);
            }
            // download/save a single note or load single file
            if (options.from_fs) {
                return [Filter.getUpdateFilePath(options.update, options)];
            } 
            return loadParseSaveOneWorksheet(options.update, options)
                .then(path => path ? [path] : undefined); // return a "path list" containing one path
        }).then(pathList => {
            // NOTE: updateFromPathList should do something with options.save, don't pass it to
            // updateFromFile, just call save at the end.
            if (!pathList || options.download_only) return undefined;
            return Update.updateFromPathList(pathList, options);
        });
}

//

async function validate (options) {
    if (!options.from_fs) usage('--from-fs is required for --validate');
    if (_.isArray(options.validate)) usage('multiple --validates not (yet) supported');
    Log.info(`validate, ${options.validate || 'all'}`);

    // TODO: what if noteobok is undefined
    return Note.getNotebook(options.notebook, options)
        .then(nb => {
            if (!nb && !options.default) {
                usage(`notebook not found, ${options.notebook}`);
            }
            Log.info(`notebook, title: ${nb.title}, guid: ${nb.guid}`);
            options.notebookGuid = nb.guid;

            ClueManager.loadAllClues({ clues: Clues.getByOptions(options) });

            if (_.isEmpty(options.validate)) {
                // no note name supplied: validate all note update files, return path list
                //return options.from_fs ?
                return getAllUpdateFilePaths(options);
                //: loadParseSaveAllWorksheets(options);
            }
            // note name supplied: return single note update file path as a "path list"
            return [Filter.getUpdateFilePath(options.validate, options)];
            //if (options.from_fs) {
            //} 
            //return loadParseSaveOneWorksheet(options.validate, options)
            //.then(path => [path]);
        }).then(pathList => {
            return NoteValidator.validatePathList(pathList, options);
        });
}

//

async function main () {
    const opt = CmdLineOptions.parseSystem();
    const options = opt.options; // = Options.set(opt.options);

    if (opt.argv.length > 0) {
        usage(`invalid non-option parameter(s) supplied, ${opt.argv}`);
    }
    if (options.quiet && options.verbose) {
        usage('--quiet and --verbose cannot both be specified');
    }

    if (options['from-fs']) options.from_fs = true;
    if (options['force-update']) options.force_update = true;
    if (options['point-size']) options.point_size = options['point-size'];
    if (options['yes-mode']) options.yes_mode = true;
    if (options['download-only']) options.download_only = true;
    if (options['dry-run']) {
        options.dry_run = true;
        if (!options.quiet) {
            options.verbose = true;
        }
    }
    options.notebook = options.notebook || Note.getWorksheetName(Clues.getByOptions(options));

    let cmd;
    for (const key of _.keys(Commands)) {
        if (_.has(options, key)) {
            cmd = key;
            Log.debug(`command: ${key}`);
            break;
        } else {
            Log.debug(`not: ${key}`);
        }
    }
    if (!cmd) usage(`missing command`);
    if (cmd === 'get') options.quiet = true;

    // Other modules will see snapshot of options taken here.
    Options.set(options);

    if (options.production) Log('---PRODUCTION---');
    if (options.dry_run) Log('---DRY_RUN---');
    if (options.verbose) Log('---VERBOSE---');
    
    const start = new Date();
    const result = await Commands[cmd](options).catch(err => { throw err; });
    const duration = new Duration(start, new Date()).milliseconds;
    console.log(`${cmd}: ${PrettyMs(duration)}`);
    
    // test bat,western - look at the countlists, combinations, make sure there's no duplicates
    
    return result;
}
    
//
    
main()
    .then(result => process.exit(result))
    .catch(err => {
        console.error(err, err.stack);
        console.log(err, err.stack);
    });
