//
// FILTER.JS
//

'use strict';

var _            = require('lodash');
var Promise      = require('bluebird');
//var FS           = require('fs');
//var fsWriteFile  = Promise.promisify(FS.writeFile);
var Dir          = require('node-dir');
var Path         = require('path');

var ClueManager  = require('../clue_manager.js');

var Opt          = require('node-getopt')
    .create([
	['s', 'synthesis',           'use synth clues'],
	['h', 'help',                'this screen']
    ])
    .bindHelp().parseSystem();

//

var RESULTS_DIR = '../../data/results/';

//
//
//

function main() {
    var filteredList = [];
    var base = 'meta';
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
	    allWordsInTitle: true
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
	console.log(JSON.stringify(filteredList));
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
		urlList: urlList
	    });
	});
    });
}

//

function filterResult(result, wordList, options) {
    return new Promise(function(resolve, reject) {
	if (result.score) {
	    if (options.allWordsInTitle) {
		if (result.score.wordsInTitle >= _.size(wordList)) {
		    resolve(result);
		}
	    }
	}
	resolve(null);
    });
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

