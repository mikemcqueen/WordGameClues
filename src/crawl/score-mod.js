//
// SCORE-MOD.JS
//

'use strict';

const _       = require('lodash');
const Promise = require('bluebird');
const Wiki    = require('wikijs').default;

//

function getWordCountInText(wordList, text) {
    var textWordList = _.words(text.toLowerCase());
    var countList = [];
    var wordCount = 0;

    // still not perfect
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

function removeWikipediaSuffix(title) {
    const index = title.lastIndexOf(' - Wikipedia');
    if (index !== -1) {
	title = title.substr(0, index);
    }
    return title;
}

//

function getDisambiguation(result) {
    return getWordCountInText(['disambiguation'], result.title) > 0;
}

//

function getWikiContent(title) {
    return new Promise(function (resolve, reject) {
	Wiki().page(title).then(page => {
	    Promise.all([page.content(), page.info()]).then(result => {
		resolve({
		    text:  result[0],
		    info : result[1]
		});
	    });
	}).catch(err => {
	    console.error(`getWikiContent: ${err}`);
	    reject(err);
	});
    });
}

//

function getScore(wordList, result) {
    return new Promise(function(resolve, reject) {
	var score;
	
	if (!_.isString(result.title) || !_.isString(result.summary)) {
	    throw new Error('bad or missing title or summary, ' + wordList);
	}
	
	score = {
	    wordsInTitle   : getWordCountInText(wordList, result.title),
	    wordsInSummary : getWordCountInText(wordList, result.summary),
	    disambiguation : getDisambiguation(result)
	};
	
	if (score.wordsInSummary < _.size(wordList)) {
	    getWikiContent(removeWikipediaSuffix(result.title))
		.then(content => {
		    score.wordsInArticle = getWordCountInText(
			wordList, content.text + ' ' + _.values(content.info).join(' '));
		    resolve(score);
		}).
		catch(err => {
		    console.error(`getScore: ${err}`);
		});
		//.finally(() => console.log('hi'));
	}
	else {
	    resolve(score);
	}
    });
}


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

module.exports = {
    getScore              : getScore,
    getWikiContent        : getWikiContent,
    removeWikipediaSuffix : removeWikipediaSuffix
};
