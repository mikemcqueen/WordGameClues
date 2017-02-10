//
// SCORE.JS
//

'use strict';

var _            = require('lodash');
var Promise      = require('bluebird');
var FS           = require('fs');
var bbWriteFile  = Promise.promisify(FS.writeFile);
var Dir          = require('node-dir');
var Path         = require('path');

var Delay        = require('../util/delay');

var Opt          = require('node-getopt')
    .create([
	['f', 'force',               'force re-score'],
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
    /*
    var base = 'meta';
    ClueManager.loadAllClues({
	baseDir:  base,
    });
    */

    if (Opt.options.force) {
	console.log('force: ' + Opt.options.force);
    }

    Dir.readFiles(RESULTS_DIR + 2, {
	match:   /\.json$/,
	exclude: /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err;
	console.log('filename: ' + filepath);
	processResultFile(filepath, content, {
	    force: Opt.options.force
	}).catch(err => {
	    console.error('error: ' + err.message);
	}).then(list => {
	    if (_.isEmpty(list)) {
		return next();
	    }
	    bbWriteFile(filepath, JSON.stringify(list)).then(() => {
		console.log('updated');
		return next();
	    });
	});
    }, function(err, files) {
        if (err) throw err;
    });
}

//

function processResultFile(filepath, content, options) {
    return new Promise((resolve, reject) => {
	var basename = Path.basename(filepath, '.json');
	var splitList = _.split(basename, '-');
	var resultList = JSON.parse(content);
	var wordList = [];
	var any = false;
	
	splitList.forEach(splitStr => {
	    splitStr.split(' ').forEach(word => {
		wordList.push(word);
	    });
	});
	console.log('wordList: ' + wordList);
	Promise.map(resultList, result => {
	    // don't need to get score if it's already present
	    if (result.score && !options.force) {
		return {};
	    }
	    return getScore(wordList, result);
	}).each((score, index) => {
	    if (!_.isEmpty(score)) {
		// TODO: really want to do "replace own properties" here,
		// then no empty check is necessary.
		resultList[index].score = score;
		any = true;
	    }
	}).then(scoreList => {
	    resolve(any ? resultList : []);
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

