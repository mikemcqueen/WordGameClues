//
// FILTER.JS
//

'use strict';

const _            = require('lodash');
const Promise      = require('bluebird');
const Dir          = require('node-dir');
const Path         = require('path');
const FS           = require('fs');
const fsReadFile   = Promise.promisify(FS.readFile);
const expect       = require('chai').expect;
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

//

function getUrlCount(resultList) {
    let urlCount = 0;
    resultList.forEach(result => {
	urlCount += _.size(result.urlList);
    });
    return urlCount;
}

//
// TODO: Result.wordCountFilter(result, wordCount);

function searchResultWordCountFilter(result, wordCount, options) {
    let passTitle = false;
    let passArticle = false;
    
    if (result.score) {
	if (options.filterTitle) {
	    passTitle = (result.score.wordsInTitle >= wordCount);
	}
	if (options.filterArticle) {
	    passArticle = (result.score.wordsInSummary >= wordCount ||
			   result.score.wordsInArticle >= wordCount);
	}
	if (passTitle || passArticle) {
	    return result;
	}
    }
    return undefined;
}

//

function isFilteredUrl(url, filteredUrls) {
    if (_.isUndefined(filteredUrls)) return false; 
    let reject = !_.isUndefined(filteredUrls.rejectUrls) &&
	filteredUrls.rejectUrls.includes(url);
    let known = !_.isUndefined(filteredUrls.knownUrls) &&
	filteredUrls.knownUrls.includes(url);
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
		return searchResultWordCountFilter(result, wordCount, options);
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

// for each search result filename in dir that matches fileMatch
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
	    if (options.verbose) {
		console.log(`filename: ${filepath}`);
	    }
	    if (err) throw err; // TODO:
	    let wordList = _.split(Path.basename(filepath, '.json'), '-');
	    // filter out rejected word combos
	    if (ClueManager.isRejectSource(wordList)) next();

  	    loadFilteredUrls(dir, wordList, options)
		.then(filteredUrls => {
		    // TODO: what happens on JSON.parse throw?
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
		    // TODO:
		    console.error('filterSearchResultFiles, error: ' + err.message);
		});
	    return next(); // process files async
	}, function(err, files) {
            if (err) throw err;  // TODO: 
	    resolve({
		filtered: filteredList,
		rejects:  rejectList
	    });
	});
    });
}

//

function displayFilterResults(resultList) {
    resultList.forEach(result => {
	if (_.isEmpty(result.urlList)) return;
	if (ClueManager.isRejectSource(result.src)) return;

	console.log(`:${result.src}`);
	result.urlList.forEach(url => {
	    console.log(url);
	});
	const nameList = ClueManager.getKnownClues(result.src);
	if (!_.isEmpty(nameList)) {
	    console.log('#known:');
	    for (const name of nameList) {
		console.log(`#${name}`);
	    }
	}
	console.log();
    });
}

//
//
//

function main() {
    expect(Opt.argv, 'no non-switch arguments allowed').to.be.empty;

    let base = Opt.options.synth ? 'synth' : 'meta';
    
    // default to title filter if no filter specified
    if (!Opt.options.article && !Opt.options.title) {
	Opt.options.title = true;
    }

    let dir = Result.DIR + (_.isUndefined(Opt.options.dir) ? '2' : Opt.options.dir);
    
    let filterOptions = {
	filterArticle: Opt.options.article,
	filterTitle:   Opt.options.title,
	filterRejects: Opt.options.rejects,
	verbose:       Opt.options.verbose
    };

    ClueManager.loadAllClues({
	baseDir:  base
    });

    let start = new Date();
    filterSearchResultFiles(dir, Result.getFileMatch(Opt.options.match), filterOptions)
	.then(result => {
	    let d = new Duration(start, new Date()).milliseconds;
	    if (Opt.options.count) {
		console.log('Results: ' + _.size(result.filtered) +
			    ', Urls: ' + getUrlCount(result.filtered) +
			    ', Rejects: ' + _.size(result.rejects) +
			    `, duration, ${PrettyMs(d)}`);
						   
	    }
	    else {
		displayFilterResults(Opt.options.rejects ? result.rejects : result.filtered);
	    }
	}).catch(err => {
	    console.error(err);
	});
}

//

try {
    main();
}
catch(e) {
    console.error(e.stack);
}
finally {
}

