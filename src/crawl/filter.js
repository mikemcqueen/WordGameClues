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

const ClueManager  = require('../clue_manager.js');

const Opt          = require('node-getopt').create([
    ['a', 'article',             'filter results based on article word count'],
    ['c', 'count',               'show result/url counts only'],
    ['d', 'dir=NAME',            'directory name'],
//    ['k', 'known',               'filter known results'],  // easy!, call ClueManager.filter()
    ['m', 'match=EXPR',          'filename match expression' ],
    ['r', 'rejects',             'show only results that fail all filters'],
    ['s', 'synthesis',           'use synth clues'],
    ['t', 'title',               'filter results based on title word count (default)'],
//    ['v', 'verbose',             'show logging']
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

//

const RESULT_DIR =      '../../data/results/';
const FILTERED_SUFFIX = '_filtered';

//

function getUrlCount(resultList) {
    let urlCount = 0;
    resultList.forEach(result => {
	urlCount += _.size(result.urlList);
    });
    return urlCount;
}

//

function makeFilename(wordList, suffix) {
    expect(wordList.length).to.be.at.least(2);

    let filename = '';
    wordList.forEach(word => {
	if (_.size(filename) > 0) {
	    filename += '-';
	}
	filename += word;
    });
    if (!_.isUndefined(suffix)) {
	filename += suffix;
    }
    return filename + '.json';
}

//

function searchResultWordCountFilter(result, wordCount, options) {
    return new Promise(function(resolve, reject) {
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
		resolve(result);
	    }
	}
	resolve(undefined);
    });
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
	    return undefined;
	}).each((result, index) => {
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

function loadFilteredUrls(dir, wordList) {
    return new Promise((resolve, reject) => {
	let filename = makeFilename(wordList, FILTERED_SUFFIX);
	return fsReadFile(filename, 'utf8')
	    .then(content => {
		// TODO: what happens on JSON.parse throw?
		resolve(JSON.parse(content));
	    }).catch(err => {
		// ignore file errors, this file is optional
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
	    if (err) throw err;
	    let wordList = _.split(Path.basename(filepath, '.json'), '-');
	    // filter out rejected word combos
	    if (ClueManager.isRejectSource(wordList)) next();

	    loadFilteredUrls(dir, wordList)
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
		    return next();
		}).catch(err => {
		    console.error('filterSearchResultFiles, error: ' + err.message);
		});
	}, function(err, files) {
            if (err) throw err;
	    resolve({
		filtered: filteredList,
		rejects: rejectList
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
	const known = ClueManager.getKnownClues(result.src);
	if (!_.isEmpty(known)) {
	    console.log('+known');
	    for (const clue of known) {
		const x = _.isUndefined(clue.x) ? '' : clue.x;
		console.log(`+${clue.name},${clue.src},${x}`);
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

    let dir = RESULT_DIR + (_.isUndefined(Opt.options.dir) ? '2' : Opt.options.dir);
    let fileMatch = '\.json$';
    if (!_.isUndefined(Opt.options.match)) {
	fileMatch = `.*${Opt.options.match}.*${fileMatch}`;
    }
    let filterOptions = {
	filterArticle: Opt.options.article,
	filterTitle:   Opt.options.title,
	filterRejects: Opt.options.rejects
    };

    ClueManager.loadAllClues({
	baseDir:  base
    });

    filterSearchResultFiles(dir, fileMatch, filterOptions)
	.then(result => {
	    if (Opt.options.count) {
		console.log('Results: ' + _.size(result.filtered) +
			    ', Urls: ' + getUrlCount(result.filtered) +
			    ', Rejects: ' + _.size(result.rejects));
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

