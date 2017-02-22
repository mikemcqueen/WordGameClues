//
// FILTER.JS
//

'use strict';

var _            = require('lodash');
var Promise      = require('bluebird');
var Dir          = require('node-dir');
var Path         = require('path');
var expect       = require('chai').expect;

var ClueManager  = require('../clue_manager.js');

var Opt          = require('node-getopt').create([
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

var RESULT_DIR = '../../data/results/';

//

function getUrlCount(resultList) {
    let urlCount = 0;
    resultList.forEach(result => {
	urlCount += _.size(result.urlList);
    });
    return urlCount;
}

//

function makeFilename(wordList) {
    let filename = '';
    wordList.forEach(word => {
	if (_.size(filename) > 0) {
	    filename += '-';
	}
	filename += word;
    });
    return filename + '.json';
}

//

function filterSearchResult(wordCount, result, options) {
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
	resolve(null);
    });
}

//

function filterSearchResultList(wordList, resultList, options) {
    return new Promise((resolve, reject) => {
	let urlList = [];
	let wordCount = _.chain(wordList).map(word => word.split(' ')).flatten().size().value();
	Promise.map(resultList, result => {
	    return filterSearchResult(wordCount, result, options);
	}).each((result, index) => {
	    if (result !== null) {
		urlList.push(result.url);
		//console.log('url: ' + result.url);
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

function filterSearchResultFiles(dir, fileMatch, options) {
    return new Promise((resolve, reject) => {
	let filteredList = [];
	let rejectList = [];
	Dir.readFiles(dir , {
	    match:   new RegExp(fileMatch),
	    exclude: /^\./,
	    recursive: false
	}, function(err, content, filepath, next) {
	    if (err) throw err;
	    //console.log('filename: ' + filepath);
	    // TODO: what happens on JSON.parse throw?
	    filterSearchResultList(
		_.split(Path.basename(filepath, '.json'), '-'),
		JSON.parse(content),
		options
	    ).catch(err => {
		console.error('filterResultList error: ' + err.message);
	    }).then(result => {
		if (!_.isEmpty(result.urlList)) {
		    filteredList.push(result);
		}
		else {
		    rejectList.push(result);
		}
		return next();
	    });
	}, function(err, files) {
            if (err) throw err;
	    resolve({
		filtered: filteredList,
		rejects:  rejectList
	    });
	});
    });
}

//

function displayFilterResults(resultList) {
    console.log(JSON.stringify(resultList));
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
    }).catch(err => console.error);
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

