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
    ['a', 'article',             'filter results based on article word count'],
    ['c', 'count',               'show result/url counts only'],
    ['d', 'dir=NAME',            'directory name'],
//    ['k', 'known',               'filter known results'],  // easy!, call ClueManager.filter()
    ['m', 'match',               'filename match expression' ],
    ['r', 'rejects',             'show only results that fail all filters'],
    ['s', 'synthesis',           'use synth clues'],
    ['t', 'title',               'filter results based on title word count (default)'],
//    ['v', 'verbose',             'show logging']
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

//

var RESULT_DIR = '../../data/results/';

//
//
//

function main() {
    var base = 'meta';
    
    // default to title filter if no filter specified
    if (!Opt.options.article && !Opt.options.title) {
	Opt.options.title = true;
    }

    let dir = RESULT_DIR + (_.isUndefined(Opt.options.dir) ? 2 : Opt.options.dir);

    let filterOptions = {
	filterArticle: Opt.options.article,
	filterTitle:   Opt.options.title,
	filterRejects: Opt.options.rejects
    };

    ClueManager.loadAllClues({
	baseDir:  base
    });

    let filteredList = [];
    let rejectList = [];
    Dir.readFiles(dir , {
	match:   /\.json$/,
	exclude: /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err;
	//console.log('filename: ' + filepath);
	// TODO: what happens on JSON.parse throw?
	filterResultList(
	    _.split(Path.basename(filepath, '.json'), '-'),
	    JSON.parse(content),
	    filterOptions
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
	if (Opt.options.count === true) {
	    console.log('Results: ' + _.size(filteredList) +
			', Urls: ' + getUrlCount(filteredList) +
			', Rejects: ' + _.size(rejectList));
	}
	else {
	    console.log(JSON.stringify(Opt.options.rejects ? rejectList : filteredList));
	}
    });
}

//

function filterResultList(wordList, resultList, options) {
    return new Promise((resolve, reject) => {
	var urlList = [];
	
	Promise.map(resultList, result => {
	    return filterResult(wordList, result, options);
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

function filterResult(wordList, result, options) {
    return new Promise(function(resolve, reject) {
	let passTitle = false;
	let passArticle = false;
	
	if (result.score) {
	    if (options.filterTitle) {
		passTitle = (result.score.wordsInTitle >= _.size(wordList));
	    }
	    if (options.filterArticle) {
		passArticle = (result.score.wordsInSummary >= _.size(wordList) ||
			       result.score.wordsInArticle >= _.size(wordList));
	    }
	    if (passTitle || passArticle) {
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

