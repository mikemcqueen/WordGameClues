//
// FILTER.JS
//

'use strict';

var _            = require('lodash');
var Promise      = require('bluebird');
var Dir          = require('node-dir');
var Path         = require('path');

var ClueManager  = require('../clue_manager.js');

var Opt          = require('node-getopt').create([
    ['c', 'count',               'show result/url counts only'],
//    ['k', 'known',               'filter known results'],  // easy!, call ClueManager.filter()
    ['r', 'rejects',             'show only results that fail all filters'],
    ['s', 'synthesis',           'use synth clues'],
    ['t', 'title',               'filter results based on title word count'],
//    ['v', 'verbose',             'show logging']
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

//

var RESULTS_DIR = '../../data/results/';

//
//
//

function main() {
    var filteredList = [];
    var base = 'meta';
    
    // default to title filter for now
    Opt.options.title = true;

    ClueManager.loadAllClues({
	baseDir:  base,
    });

    Dir.readFiles(RESULTS_DIR + 2, {
	match:   /\.json$/,
	exclude: /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err;
	//console.log('filename: ' + filepath);
	filterResultList(filepath, JSON.parse(content), {
	    filterTitle:   Opt.options.title,
	    filterRejects: Opt.options.rejects
	}).catch(err => {
	    console.error('error: ' + err.message);
	}).then(result => {
	    if (!_.isEmpty(result.urlList)) {
		filteredList.push(result);
		//console.log('list: ' + _.size(filteredList));
	    }
	    return next();
	});
    }, function(err, files) {

        if (err) throw err;
	if (Opt.options.count) {
	    console.log('Results: ' + _.size(filteredList) +
			', Urls: ' + getUrlCount(filteredList));
	}
	else {
	    console.log(JSON.stringify(filteredList));
	}
	//console.log('total: ' + _.size(filteredList));
    });
}

//

function filterResultList(filepath, resultList, options) {
    return new Promise((resolve, reject) => {
	var basename = Path.basename(filepath, '.json');
	var wordList = _.split(basename, '-');
	var urlList = [];
	
	Promise.map(resultList, result => {
	    return filterResult(result, wordList, options);
	}).each((result, index) => {
	    if (result !== null) {
		urlList.push(result.url);
		//console.log('url: ' + result.url);
	    }
	}).then(resultList => {
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

function filterResult(result, wordList, options) {
    return new Promise(function(resolve, reject) {
	var passTitle = false;
	if (result.score) {
	    if (options.filterTitle) {
		passTitle = (result.score.wordsInTitle >= _.size(wordList));
	    }
	    if (passTitle || options.filterRejects) {
		resolve(result);
	    }
	}
	resolve(null);
    });
}

//

function getUrlCount(resultList) {
    var urlCount = 0;
    resultList.forEach(result => {
	urlCount += _.size(result.urlList);
    });
    return urlCount;
}

//

function makeFilename(wordList) {
    var filename = '';

    wordList.forEach(word => {
	if (_.size(filename) > 0) {
	    filename += '-';
	}
	filename += word;
    });
    return filename + '.json';
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

