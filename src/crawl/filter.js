//
// FILTER.JS
//

'use strict';

const _            = require('lodash');
const Promise      = require('bluebird');
const Dir          = require('node-dir');
const Path         = require('path');
const Fs           = require('fs');
const Expect       = require('chai').expect;
const Duration     = require('duration');
const PrettyMs     = require('pretty-ms');
const ClueManager  = require('../clue_manager.js');
const Result       = require('./result-mod');

const Opt          = require('node-getopt').create([
    ['a', 'article',             'filter results based on article word count'],
    ['c', 'count',               'show result/url counts only'],
    ['d', 'dir=NAME',            'directory name'],
//    ['k', 'known',               'filter known results'],  // easy!, call ClueManager.filter()
    ['m', 'match=EXPR',          'filename match expression' ],
    ['r', 'rejects',             'show only results that fail all filters'],
    ['s', 'synthesis',           'use synth clues'],
    ['t', 'title',               'filter results based on title word count (default)'],
    ['v', 'verbose',             'show logging'],
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

const fsReadFile   = Promise.promisify(Fs.readFile);

//

function getUrlCount(resultList) {
    // TODO _.reduce()
    let urlCount = 0;
    for (const result of resultList) {
	urlCount += _.size(result.urlList);
    }
    return urlCount;
}

//

function isFilteredUrl(url, filteredUrls) {
    if (_.isUndefined(filteredUrls)) return false; 
    let reject = _.isArray(filteredUrls.rejectUrls) && filteredUrls.rejectUrls.includes(url);
    let known = _.isArray(filteredUrls.knownUrls) && filteredUrls.knownUrls.includes(url);
    return known || reject;
}

//

function filterSearchResultList(resultList, wordList, filteredUrls, options) {
    return new Promise((resolve, reject) => {
	let urlList = [];
	// make any clue words with a space into multiple words.
	let wordCount = _.chain(wordList).map(word => word.split(' ')).flatten().size().value();
	Promise.map(resultList, result => {
	    if (!isFilteredUrl(result.url, filteredUrls)) {
		return Result.wordCountFilter(result, wordCount, options);
	    }
	    if (options.verbose) {
		console.log(`filtered url, ${result.url}`);
	    }
	    return undefined;
	}).each(result => {
	    if (!_.isUndefined(result)) {
		//console.log('url: ' + result.url);
		urlList.push(result.url);
	    }
	}).then(filteredList => {
	    //console.log('urlList.size = ' + _.size(urlList));
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

function loadFilteredUrls(dir, wordList, options) {
    return new Promise((resolve, reject) => {
	let filteredFilename = Result.makeFilteredFilename(wordList);
	if (options.verbose) {
	    console.log(`filtered filename: ${filteredFilename}`);
	}
	return fsReadFile(`${dir}/${filteredFilename}`, 'utf8')
	    .then(content => {
		if (options.verbose) {
		    console.log(`resolving filtered urls, ${wordList}`);
		}
		// TODO: what happens on JSON.parse throw?
		resolve(JSON.parse(content));
	    }).catch(err => {
		if (options.verbose) {
		    console.log(`no filtered urls, ${wordList}, ${err}`);
		}
		// ignore file errors, filtered url file is optional
		resolve(undefined);
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
function filterSearchResultFiles(dir, fileMatch, options) {
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
	    let wordList = _.split(Path.basename(filepath, '.json'), '-');
	    // filter out rejected word combos
	    if (ClueManager.isRejectSource(wordList)) next();

  	    loadFilteredUrls(dir, wordList, options)
		.then(filteredUrls => {
		    return filterSearchResultList(JSON.parse(content), wordList, filteredUrls, options);
		}).then(filterResult => {
		    if (!_.isEmpty(filterResult.urlList)) {
			filteredList.push(filterResult);
		    }
		    else {
			rejectList.push(filterResult);
		    }
		    return undefined;
		}).catch(err => {
		    // report & eat all errors
		    console.log('filterSearchResultFiles, error: ' + err.message);
		}).then(() => next()); // process files synchronously
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

function displayFilterResults(resultList) {
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
function main() {
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
	verbose:       Opt.options.verbose
    };
    let start = new Date();
    filterSearchResultFiles(dir, Result.getFileMatch(Opt.options.match), filterOptions)
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
	}).catch(err => {
	    // TODO: test
	    console.log(err);
	});
}

//

try {
    main();
}
catch(e) {
    console.log(e.stack);
}
finally {
}

