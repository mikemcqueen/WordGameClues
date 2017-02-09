//
// SCORE.JS
//

'use strict';

var _            = require('lodash');
//var Q            = require('q');
var Promise      = require('bluebird');
var fs           = Promise.promisifyAll(require('fs'));
var Dir          = require('node-dir');
var Path         = require('path');

var ClueManager  = require('../clue_manager.js');
var Delay        = require('../util/delay');
var googleResult = require('./googleResult');

var RESULTS_DIR = '../../data/results/';

var Opt = require('node-getopt')
    .create([
	['s' , '--synthesis',         'use synth clues'],
	['h' , 'help',                'this screen']
    ])
    .bindHelp().parseSystem();

//
//
//

function main() {
    var base = 'meta';

    /*
    if (Opt.argv.length < 1) {
	console.log('Usage: node score [--synthesis]');
	console.log('');
	return 1;
    }

    filename = Opt.argv[0];
    console.log('filename: ' + filename);
    */

    ClueManager.loadAllClues({
	baseDir:  base,
    });

    Dir.readFiles(RESULTS_DIR + 2, {
	match:   /\.json$/,
	exclude: /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err;
	console.log('filename: ' + filepath);
	processResultFile(filepath, content).catch(
	    console.err
	).catch(err => {
	    console.log('caught error in main()');
	}).then((list) => {
	    if (list === null) {
		return next();
	    }
	    fs.writeFile(filepath, JSON.stringify(list), (err) => {
		if (err) throw err;
		return next();
	    });
	});
    }, function(err, files) {
        if (err) throw err;
    });
}

//

function processResultFile(filepath, content) {
    return new Promise((resolve, reject) => {
	var basename = Path.basename(filepath, '.json');
	var wordList = _.split(basename, '-');
	var resultList = JSON.parse(content);
	var any = false;
	
	console.log('wordList: ' + wordList);
	Promise.map(resultList, result => {
	    // don't need to get score if it's already present
	    if (result.score) return null;
	    return getScore(wordList, result);
	}).each((score, index) => {
	    if (score !== null) {
		resultList[index].score = score;
		any = true;
	    }
	}).then(scoreList => {
	    resolve(any ? resultList : null);
	});
    });
}

//

function getScore(wordList, result) {
    return new Promise(function(resolve, reject) {
	var score;
	var wordsInArticle;
	
	if (!result.title || !(typeof result.title === 'string') ||
	    !result.summary || !(typeof result.summary === 'string')) {
	    console.log('bad or missing title or summary, ' + wordList);
	    resolve(null);
	}
	
	score = {
	    wordsInTitle   : getWordCountInText(wordList, result.title),
	    wordsInSummary : getWordCountInText(wordList, result.summary),
	    disambiguation : getDisambiguation(result)
	};
	
	if (score.wordsInSummary < _.size(wordList)) {
	    getWordsInArticle(wordList, result, (err, count) => {
		if (err === null) {
		    console.log('wordsInArticle: ' + count);
		    score.wordsInArticle = count;
		}
		resolve(score);
	    });
	}
	else {
	    resolve(score);
	}
    });
}

//

function getDisambiguation(result) {
    return getWordCountInText(['disambiguation'], result.title) > 0;
}

//

function getWordCountInText(wordList, text) {
    var textWordList = _.words(text.toLowerCase());
    var countList = [];
    var wordCount = 0;

    textWordList.forEach(textWord => {
	wordList.forEach((word, index) => {
	    //if (_.startsWith(textWord, word)) {
	    if (word == textWord) {
		if (!countList[index]) countList[index] = 0;
		countList[index] += 1;
	    }
	});
    });
    countList.forEach(count => wordCount += 1);
    return wordCount;
}

//

function getWordsInArticle(wordList, result, cb) {
    cb(new Error('getWordsInArticle not implemented'));
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

