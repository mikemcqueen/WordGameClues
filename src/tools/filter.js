//
// filter.js
//

'use strict';

const _            = require('lodash');
const ChildProcess = require('child_process');
const ClueManager  = require('../clue-manager');
const Clues        = require('../clue-types');
const Debug        = require('debug')('filter');
const Dir          = require('node-dir');
const Duration     = require('duration');
const Expect       = require('should/as-function');
const Fs           = require('fs');
const Getopt       = require('node-getopt');
const My           = require('../misc/util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');
const Score        = require('../modules/score');
const SearchResult = require('../modules/search-result');
const Tmp          = require('tmp');

//

const Options = new Getopt(_.concat(Clues.Options, [
	//.create(
	['a', 'article',             'filter results based on article word count'],
	['',  'copy',                'copy to clipboard as RTF'],
	['d', 'dir=NAME',            'directory name'],
	//    ['k', 'known',               'filter known results'],  // easy!, call ClueManager.filter()
	['m', 'match=EXPR',          'filename match expression' ],
	['n', 'count',               'show result/url counts only'],
	['',  'note',                'mail results to evernote'], 
	//    ['r', 'rejects',             'show only results that fail all filters'],
	['t', 'title',               'filter results based on title word count (default)'],
	['x', 'xfactor=VALUE',       'show 1) missing URL/title/summary 2) unscored URLs 3) article < summary'], 
	['v', 'verbose',             'show logging'],
	['w', 'word',                'search for additional word'],
	['h', 'help',                'this screen']
    ])).bindHelp(
	"Usage: node filter <options> [wordListFile]\n\n[[OPTIONS]]\n"
    );

//

const csvParse     = Promise.promisify(require('csv-parse'));
const fsReadFile   = Promise.promisify(Fs.readFile);

//

const XFACTOR_MISSING  = 1;
const XFACTOR_UNSCORED = 2;
const XFACTOR_BADSCORE = 3;

//

function usage (msg) {
    if (msg) {
	console.log(`\n${msg}\n`);
    }
    Options.showHelp();
    process.exit(-1);
}

//

function getUrlCount (resultList) {
    Expect(resultList).is.an.Array();
    // TODO _.reduce()
    let urlCount = 0;
    for (const result of resultList) {
	urlCount += _.size(result.urlList);
    }
    return urlCount;
}

//

function isFilteredUrl (url, filteredUrls) {
    Expect(url).is.a.String('iFU, url');
    if (_.isUndefined(filteredUrls)) return false; 
    Expect(filteredUrls).is.an.Object();
    let reject = _.isArray(filteredUrls.rejectUrls) && filteredUrls.rejectUrls.includes(url);
    let known = _.isArray(filteredUrls.knownUrls) && filteredUrls.knownUrls.includes(url);
    return known || reject;
}

// check for broken results if xfactor option specified

function isXFactor (result, options) {
    Expect(result).is.an.Object();
    Expect(options).is.an.Object();
    if (options.xfactor === XFACTOR_MISSING) { 
	const missing = _.isEmpty(result.url) || _.isEmpty(result.title) || _.isEmpty(result.summary);
	if (missing) {
	    console.log(`${JSON.stringify(result)}`);
	}
	return missing;
    }
    if (options.xfactor === XFACTOR_UNSCORED) {
	return _.isEmpty(result.score);
    }
    if (options.xfactor === XFACTOR_BADSCORE && _.isObject(result.score)) {
	return result.score.wordsInArticle < result.score.wordsInSummary;
    }
    return false;
}

//

function filterSearchResultList (resultList, wordList, filteredUrls, options, filepath) {
    Expect(resultList).is.an.Array();
    Expect(wordList).is.an.Array().and.not.empty();
    Expect(options).is.an.Object();
    // filteredUrls can be undefined or array
    let urlList = [];
    let loggedXFactor = false;
    // make any clue words with a space into multiple words.
    let wordCount = _.chain(wordList).map(word => word.split(' ')).flatten().size().value();
    
    Debug(`fSRL: filteredUrls(${_.size(filteredUrls)})\n${JSON.stringify(filteredUrls)}`);

    return Promise.map(resultList, result => {
	//Debug(`result: ${_.entries(result)}`);
	if (options.xfactor && isXFactor(result, options)) {
	    if (!loggedXFactor) {
		console.log(`x: ${filepath}`);
		loggedXFactor = true;
	    }
	    return undefined;
	}
	if (isFilteredUrl(result.url, filteredUrls) || _.isUndefined(result.score)) {
	    Debug(`filtered or unscored url, ${result.url}`);
	    return undefined;
	}
	return Score.wordCountFilter(result.score, wordCount, options) ? result : undefined;
    }).each(filterResult => {
	if (filterResult) {
	    Debug(`url: ${filterResult.url}`);
	    urlList.push(filterResult.url);
	}
	return undefined;
    }).then(() => {
	Debug(`urlList.size = ${_.size(urlList)}`);
	return {
	    src:     wordList.toString(),
	    urlList: urlList,
	    known:   ClueManager.getKnownClues(wordList)
	};
    });
    // TODO: .catch()
}

//

function loadFilteredUrls (dir, wordList, options) {
    Expect(dir).is.a.String('lFU dir');
    Expect(wordList).is.an.Array();
    Expect(options).is.an.Object();
    let filteredFilename = SearchResult.makeFilteredFilename(wordList);
    Debug(`filtered filename: ${filteredFilename}`);
    return fsReadFile(Path.format({ dir, base: filteredFilename }), 'utf8')
	.then(content => {
	    Debug(`resolving filtered urls for: ${wordList}`);
	    return JSON.parse(content);
	}).catch(err => {
	    if (err && err.code !== 'ENOENT') throw err;
	    Debug(`no filtered urls, ${wordList}, ${err}`);
	    return undefined;
	});
}

//

function hasRemaining (wordListArray, remaining) {
    Expect(wordListArray).is.an.Array();
    Expect(remaining).is.an.Array().and.not.empty();
    for (const wordList of wordListArray) {
	if (wordList.length !== remaining.length) continue;
	if (remaining.some(word => _.includes(wordList, word))) {
	    return true;
	}
    }
    return false;
}

//

function isInNameMap (nameMap, wordList) {
    Expect(nameMap).is.an.Object();       // must be object if defined
    Expect(wordList).is.an.Array().with.property('length').above(0); // at.least(1)
    let word = wordList[0];
    if (!_.has(nameMap, word)) return false;
    return hasRemaining(nameMap[word], wordList.slice(1, wordList.length));
}

// for each search result filename that matches fileMatch in dir 
//   filter out rejected word combinations
//   load the file, build filtered URL list
//   load the _filtered.json file, filter out known/reject URLs
//   if any URLs remain, add result to filteredList.
//
// NOTE that for convenience, this function currently loads all
// files that match fileMatch, even if the file's word-combo is
// a rejected combo. 
// 
function filterSearchResultDir (dir, fileMatch, options) {
    Expect(dir).is.a.String('fSRD dir');
    Expect(fileMatch).is.a.String('fSRD filematch');
    Expect(options).is.an.Object();
    return new Promise((resolve, reject) => {
	let filteredList = [];
	let rejectList = [];
	Dir.readFiles(dir , {
	    match:     new RegExp(fileMatch),
	    exclude:   /^\./,
	    recursive: false
	}, function(err, content, filepath, next) {
	    if (err) throw err; // TODO: test
	    Debug(`filename: ${filepath}`);
	    let wordList = SearchResult.makeWordlist(filepath);
	    // filter out rejected word combos
	    if (ClueManager.isRejectSource(wordList)) return next();
	    if (!_.isUndefined(options.nameMap) && !isInNameMap(options.nameMap, wordList)) return next();
	    // temp

	    /*
	    if (!_.isUndefined(options.nameMap)) {
		console.log(`filename: ${filepath}`);
		return next();
	    }
	     */
  	    loadFilteredUrls(dir, wordList, options)
		.then(filteredUrls => {
		    return filterSearchResultList(JSON.parse(content), wordList, filteredUrls, options, filepath);
		}).then(filterResult => {
		    // TODO: I question this logic at the moment
		    if (_.isEmpty(filterResult.urlList)) {
			rejectList.push(filterResult);
		    } else {
			filteredList.push(filterResult);
		    }
		    return undefined;
		}).catch(err => {
		    // report & eat all errors
		    console.log(`filterSearchResultDir, path: ${filepath}`, err, err.stack);
		});//.then(() => next()); // process files synchronously
	    return next(); // process files asynchronously
	}, function(err, files) {
            if (err) throw err;  // TODO: test
	    resolve({
		filtered: filteredList,
		rejects:  rejectList
	    });
	});
    });
}

// for each result file path in pathlist
//   load result file
//   make word list from filename
//   

function filterPathList (pathList, dir, options) {
    Expect(pathList).is.an.Array();
    Expect(options).is.an.Object();

    let filteredList = [];
    let rejectList = [];
    return Promise.map(pathList, path => {
	let filename = Path.basename(path);
	Debug(`filename: ${filename}`);
	let wordList = SearchResult.makeWordlist(filename);
  	return Promise.all([fsReadFile(path, 'utf8'), loadFilteredUrls(dir, wordList, options)])
	    .then(([content, filteredUrls]) => {
		return filterSearchResultList(JSON.parse(content), wordList, filteredUrls, options, path);
	    }).then(filterResult => {
		// TODO: this is probably wrong for rejects
		if (_.isEmpty(filterResult.urlList)) {
		    rejectList.push(filterResult);
		} else {
		    filteredList.push(filterResult);
		}
		return undefined;
	    }).catch(err => {
		// report & eat all errors
		console.log(`filterSearchResultFiles, path: ${path}`, err, err.stack);
	    });
    }).then(() => {
	return {
	    filtered: filteredList,
	    rejects:  rejectList
	};
    });
}

// TODO: move to Util.js

function loadCsv (filename) {
    return fsReadFile(filename, 'utf8')
	.then(csvContent => csvParse(csvContent, { relax_column_count: true }));
}

//

function buildNameMap (wordListArray) {
    let map = {};
    for (const wordList of wordListArray) {
	for (const word of wordList) {
	    let remaining = _.difference(wordList, [word]);
	    Expect(remaining.length).is.above(0, `${wordList}, ${word}`); // at.least(1)
	    if (!_.has(map, word)) {
		map[word] = [];
	    }
	    if (!hasRemaining(map[word], remaining)) {
		map[word].push(remaining);
	    }
	}
    }
    return map;
}

//

function getPathList (dir, fileMatch, nameMap) {
    Expect(dir).is.a.String('gPL dir');
    Expect(fileMatch).is.a.String('gPL filematch');
    return new Promise((resolve, reject) => {
	Dir.files(dir, (err, pathList) => {
	    if (err) reject(err);
	    let match = new RegExp(fileMatch);
	    let filtered = _.filter(pathList, path => {
		let filename = Path.basename(path)
		let wordList = SearchResult.makeWordlist(filename);
		// filter out rejected word combos
		if (ClueManager.isRejectSource(wordList)) return false;
		if (!_.isUndefined(nameMap) && !isInNameMap(nameMap, wordList)) return false;
		return match.test(filename);
	    });
	    console.log(`pathList(${pathList.length}), filtered(${filtered.length})`);
	    resolve(filtered);
	});
    });
}

//

function writeFilterResults (resultList, stream) {
    Expect(resultList).is.an.Array();
    for (const result of resultList) {
	if (_.isEmpty(result.urlList)) continue;
	if (ClueManager.isRejectSource(result.src)) continue;

	My.logStream(stream, `${SearchResult.SRC_PREFIX}${result.src}`);
	for (const url of result.urlList) {
	    My.logStream(stream, url);
	}
	const nameList = ClueManager.getKnownClues(result.src);
	if (!_.isEmpty(nameList)) {
	    My.logStream(stream, `\n${SearchResult.KNOWN_PREFIX}known:`);
	    for (const name of nameList) {
		My.logStream(stream, `${SearchResult.KNOWN_PREFIX}${name}`);
	    }
	}
	My.logStream(stream, '');
    }
}

// TODO: move to util.js

function createTmpFile()  {
    return new Promise((resolve, reject) => {
	Tmp.file((err, path, fd) => {
	    if (err) throw err;
	    console.log("File: ", path);
	    console.log("Filedescriptor: ", fd);
	    resolve([path, fd]);
	});
    });
}

// TODO: move to util.js

function mailTextFile(options) {
    let pipe = false;
    let fd = Fs.openSync(options.path, 'r');
    let child = ChildProcess.spawn('mail', ['-s', `${options.subject}`, `${options.to}`], {
	stdio: [pipe ? 'pipe' : fd, 1, 2]
    });
    if (pipe) {
	let s = Fs.createReadStream(null, { fd });
	s.pipe(child.stdin);
	s.on('data', (data) => {
	    console.log('s.data');
	});
	s.on('end', () => {
	    console.log('s.end');
	    child.stdin.end();
	});
    }
}

// TODO: move to util.js, comment purpose

function copyTextFile(path) {
    let fd = Fs.openSync(path, 'r');
    let textutil = ChildProcess.spawn(
	'textutil', ['-format', 'txt', '-convert', 'rtf', '-stdout', `${path}`],
	{ stdio: [fd, 'pipe', 2] });	
    let pbcopy = ChildProcess.spawn('pbcopy', ['-Prefer', 'rtf']);
    textutil.stdout.pipe(pbcopy.stdin);
    textutil.stdout.on('end', () => {
	console.log('textutil.end');
	Fs.closeSync(fd);
    });
}

//
//
//
async function main () {
    const opt = Options.parseSystem();
    const options = opt.options;

    if (opt.argv.length > 1) {
	usage('only one non-switch FILE argument allowed');
    }
    if (!options.dir) {
	options.dir = '2';
    }

    ClueManager.loadAllClues({ clues: Clues.getByOptions(options) });

    // default to title filter if no filter specified
    if (!options.article && !options.title) {
	options.title = true;
    }
    let nameMap;
    if (opt.argv.length > 0) {
	let wordListArray = await loadCsv(opt.argv[0]);
	nameMap = buildNameMap(wordListArray);
    }
    let dir = SearchResult.DIR + options.dir;
    let filterOptions = {
	filterArticle: options.article,
	filterTitle:   options.title,
	filterRejects: options.rejects,
	xfactor:       _.toNumber(options.xfactor)
    };
    let start = new Date();
    let pathList = await getPathList(dir, SearchResult.getFileMatch(options.match), nameMap);
    let getDuration = new Duration(start, new Date()).milliseconds;
    start = new Date();
    let result = await filterPathList(pathList, dir, filterOptions);
    let d = new Duration(start, new Date()).milliseconds;
    if (options.count) {
	console.log(`Results: ${_.size(result.filtered)}` +
		    `, Urls: ${getUrlCount(result.filtered)}` +
		    `, Rejects: ${_.size(result.rejects)}` +
		    `, get(${PrettyMs(getDuration)})` +
		    `, filter(${PrettyMs(d)})`);
	return undefined;
    }

    if (options.xfactor) return undefined;

    // result.filtered is a list of result objects,
    //   which contain urlLists
    // if options.word, we want to re-score all of those
    //   urls using only options.word
    // re-scoring should give us a list of "scored results"
    //   similar to what is saved in result dir. but we don't
    //   need to save those results this time.
    // then we want to call something similar to filterSearchResultList
    //   which just calls score.wordCountFilter() on those
    //   re-scored results, and without filtering "filtered" URLs
    // now we have two result.filtered lists. one contains the
    //   original URLs that matched the filename words,
    //   and one contains the URLS that match options.word
    // so the second one we can just use, right?
    //
    // other considerations:
    //
    // maybe we want to save the score data for the --word in the
    //   original results file. that way a subsequent --word filter
    //   woudln't have to do the wikipedia search.
    //
    // of course, we could just save the wikipedia download data as well.
    //   is there a database that auto-indexes all words in a document?
    //

    let useTmpFile = options.note || options.copy;
    return Promise.resolve().then(() => {
	if (useTmpFile) {
	    return createTmpFile().then(([path, fd]) => {
		return [path, Fs.createWriteStream(null, { fd })];
	    });
	}
	return [null, process.stdout];
    }).then(([path, stream]) => {
	writeFilterResults(options.rejects ? result.rejects : result.filtered, stream);
	if (useTmpFile) {
	    stream && stream.end();
	    if (options.mail) {
		mailTextFile({
		    to:      'mmcqueen112.ba110c6@m.evernote.com',
		    subject: `filtered @Worksheets.new`,
		    path:    path
		});
	    } else if (options.copy) {
		copyTextFile(path);
	    }
	}
	return undefined;
    });
}

//

main()
.catch(err => {
    console.log(err, err.stack);
});

