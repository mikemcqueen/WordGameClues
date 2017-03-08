//
// FILTER.JS
//

'use strict';

const _            = require('lodash');
const ClueManager  = require('../clue_manager.js');
const Dir          = require('node-dir');
const Fs           = require('fs');
const Duration     = require('duration');
const Expect       = require('chai').expect;
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');
const Result       = require('./result-mod');
const Score        = require('./score-mod');

const Opt          = require('node-getopt').create([
    ['a', 'article',             'filter results based on article word count'],
    ['c', 'count',               'show result/url counts only'],
    ['d', 'dir=NAME',            'directory name'],
//    ['k', 'known',               'filter known results'],  // easy!, call ClueManager.filter()
    ['m', 'match=EXPR',          'filename match expression' ],
    ['r', 'rejects',             'show only results that fail all filters'],
    ['s', 'synthesis',           'use synth clues'],
    ['t', 'title',               'filter results based on title word count (default)'],
    ['x', 'xfactor=VALUE',       'show result filenames with: 1:null URLs, 2:unscored URLs'], 
    ['v', 'verbose',             'show logging'],
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

const fsReadFile   = Promise.promisify(Fs.readFile);

//

const XFACTOR_NULL_URL = 1;
const XFACTOR_UNSCORED = 2;

//

function getUrlCount (resultList) {
    Expect(resultList).to.be.an('array');
    // TODO _.reduce()
    let urlCount = 0;
    for (const result of resultList) {
	urlCount += _.size(result.urlList);
    }
    return urlCount;
}

//

function isFilteredUrl (url, filteredUrls) {
    Expect(url).to.be.a('string');
    if (_.isUndefined(filteredUrls)) return false; 
    Expect(filteredUrls).to.be.an('object');
    let reject = _.isArray(filteredUrls.rejectUrls) && filteredUrls.rejectUrls.includes(url);
    let known = _.isArray(filteredUrls.knownUrls) && filteredUrls.knownUrls.includes(url);
    return known || reject;
}

//

function filterSearchResultList (resultList, wordList, filteredUrls, options) {
    Expect(resultList).to.be.an('array');
    Expect(wordList).to.be.an('array').that.is.not.empty;
    // filteredUrls can be undefined or array
    Expect(options).to.be.an('object');
    return new Promise((resolve, reject) => {
	let urlList = [];
	let logUnscored = false;
	// make any clue words with a space into multiple words.
	let wordCount = _.chain(wordList).map(word => word.split(' ')).flatten().size().value();
	Promise.map(resultList, result => {
	    if (options.verbose) {
		console.log(`result: ${_.entries(result)}`);
	    }
	    // shouldn't happen, but does, or did. xfactor gives a list of them.
	    if (!result.url) {
		if (options.xfactor === XFACTOR_NULL_URL) { 
		    console.log(options.filepath);
		}
		return undefined;
	    }
	    // log un-scored results
	    if (options.xfactor === XFACTOR_UNSCORED && _.isUndefined(result.score)) {
		if (!logUnscored) {
		    console.log(options.filepath);
		    logUnscored = true;
		}
	    }
	    if (isFilteredUrl(result.url, filteredUrls) || _.isUndefined(result.score)) {
		if (options.verbose) {
		    console.log(`filtered or unscored url, ${result.url}`);
		}
		return undefined;
	    }
	    return Score.wordCountFilter(result.score, wordCount, options) ? result : undefined;
	}).each(filterResult => {
	    if (!_.isUndefined(filterResult)) {
		if (options.verbose) {
		    console.log(`url: ${filterResult.url}`);
		}
		urlList.push(filterResult.url);
	    }
	    return undefined;
	}).then(() => {
	    if (options.verbose) {
		console.log(`urlList.size = ${_.size(urlList)}`);
	    }
	    resolve({
		src:     wordList.toString(),
		urlList: urlList,
		known:   ClueManager.getKnownClues(wordList)
	    });
	});
	// TODO: .catch()
    });
}

//

function loadFilteredUrls (dir, wordList, options) {
    Expect(dir).to.be.a('string');
    Expect(wordList).to.be.an('array');
    Expect(options).to.be.an('object');
    return new Promise((resolve, reject) => {
	let filteredFilename = Result.makeFilteredFilename(wordList);
	if (options.verbose) {
	    console.log(`filtered filename: ${filteredFilename}`);
	}
	fsReadFile(Path.format({ dir, base: filteredFilename }), 'utf8')
	    .then(content => {
		if (options.verbose) {
		    console.log(`resolving filtered urls, ${wordList}`);
		}
		resolve(JSON.parse(content));
	    }).catch(err => {
		if (options.verbose) {
		    console.log(`no filtered urls, ${wordList}, ${err}`);
		}
		reject(); // do NOT pass err
	    });
    });
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
    Expect(dir).to.be.a('string');
    Expect(fileMatch).to.be.an('string');
    Expect(options).to.be.an('object');
    return new Promise((resolve, reject) => {
	let filteredList = [];
	let rejectList = [];
	Dir.readFiles(dir , {
	    match:   new RegExp(fileMatch),
	    exclude: /^\./,
	    recursive: false
	}, function(err, content, filepath, next) {
	    if (err) throw err; // TODO: test
	    if (options.verbose) {
		console.log(`filename: ${filepath}`);
	    }
	    let wordList = Result.makeWordlist(filepath);
	    // filter out rejected word combos
	    if (ClueManager.isRejectSource(wordList)) return next();
	    
  	    loadFilteredUrls(dir, wordList, options)
		.catch(err => {
		    // loadFilteredUrls may reject and land here, but err will
		    // be undefined. we can continue filtering without existing
		    // filteredUrls.
		    if (err) throw err; 
		})
		.then(filteredUrls => {
		    options.filepath = filepath;
		    return filterSearchResultList(JSON.parse(content), wordList, filteredUrls, options);
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
		    console.log(`filterSearchResultFiles, path: ${filepath}, error; ${err}`);
		}).then(() => next()); // process files synchronously
	    //return next(); // process files asynchronously
	    return undefined;
	}, function(err, files) {
            if (err) throw err;  // TODO: test
	    resolve({
		filtered: filteredList,
		rejects:  rejectList
	    });
	});
    });
}

//

function displayFilterResults (resultList) {
    Expect(resultList).to.be.an('array');
    for (const result of resultList) {
	if (_.isEmpty(result.urlList)) continue;
	if (ClueManager.isRejectSource(result.src)) continue;

	console.log(`:${result.src}`);
	for (const url of result.urlList) {
	    console.log(url);
	}
	const nameList = ClueManager.getKnownClues(result.src);
	if (!_.isEmpty(nameList)) {
	    console.log('#known:');
	    for (const name of nameList) {
		console.log(`#${name}`);
	    }
	}
	console.log();
    }
}

//
//
//
function main () {
    Expect(Opt.argv, 'no non-switch arguments allowed').to.be.empty;
    Expect(Opt.options.dir, 'option -d NAME is required').to.exist;

    // default to title filter if no filter specified
    if (!Opt.options.article && !Opt.options.title) {
	Opt.options.title = true;
    }

    ClueManager.loadAllClues({
	baseDir: Opt.options.synth ? 'synth' : 'meta'
    });

    let dir = Result.DIR + Opt.options.dir;
    let filterOptions = {
	filterArticle: Opt.options.article,
	filterTitle:   Opt.options.title,
	filterRejects: Opt.options.rejects,
	verbose:       Opt.options.verbose,
	xfactor:       _.toNumber(Opt.options.xfactor)
    };
    let start = new Date();
    return filterSearchResultDir(dir, Result.getFileMatch(Opt.options.match), filterOptions)
	.then(result => {
	    let d = new Duration(start, new Date()).milliseconds;
	    if (Opt.options.count) {
		console.log(`Results: ${_.size(result.filtered)}` +
			    `, Urls: ${getUrlCount(result.filtered)}` +
			    `, Rejects: ${_.size(result.rejects)}` +
			    `, duration, ${PrettyMs(d)}`);
	    }
	    else {
		displayFilterResults(Opt.options.rejects ? result.rejects : result.filtered);
	    }
	});
}

//

try {
    main().catch(err => {
	console.log(err.stack);
    });
} catch(err) {
    console.log(err.stack);
}

