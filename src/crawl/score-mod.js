//
// SCORE-MOD.JS
//

'use strict';

const _       = require('lodash');
const Promise = require('bluebird');
const Wiki    = require('wikijs').default;
const expect  = require('chai').expect;

//

function getWordCountInText(wordList, text) {
    expect(text).to.be.a('string');

    let textWordList = _.words(text.toLowerCase());
    let countList = new Array(wordList.length).fill(0);

    textWordList.forEach((textWord, textIndex) => {
	wordList.forEach((word, index) => {
	    // NOTE: words are all currently already split before calling this function.
	    // this code was previously used for a different multi-word lookup strategy
	    // (which may be re-employed).
	    let nextTextWord = textWord;
	    if (word.split(' ').every((subWord, subIndex, subWordList) => {
		//if (_.startsWith(nextTextWord, subWord)) {
		if (nextTextWord !== subWord) {
		    //console.log(nextTextWord + ' !=  ' + subWord);
		    return false;
		}
		//console.log(nextTextWord + ' ==  ' + subWord);
		if (subIndex < subWordList.length - 1) {
		    // we're before the last element in subword list,
		    // ty to set the next textWord.
		    let nextTextIndex = textIndex + subIndex + 1;
		    if (nextTextIndex < textWordList.length) {
			nextTextWord = textWordList[nextTextIndex];
		    } else {
			// there aren't enough words from the text remaining
			return false;
		    }
		}
	        return true;
	     })) {
	         //console.log('count[' + index + '] = ' + (countList[index] + 1));
 	         countList[index] += 1;
	     }
	});
    });
    // TODO: _.reduce()
    let wordCount = 0;
    countList.forEach(count => {
	if (count > 0) wordCount += 1;
    });
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
    return _.isString(result.title)
	? (getWordCountInText(['disambiguation'], result.title) > 0)
	: false;
}

//

function getWikiContent(title) {
    return new Promise(function (resolve, reject) {
	Wiki().page(title).then(page => {
	    Promise.all([
		Promise.resolve(page.content()).reflect(),
		Promise.resolve(page.info()).reflect()
	    ]).then(result => {
		resolve({
		    text: result[0].isFulfilled() ? result[0].value() : '',
		    info: result[1].isFulfilled() ? result[1].value() : {}
 		});
	    }).catch(err => {
		// TODO: use VError
		console.log(`getWikiContent promise.all: ${err}`);
		reject(err);
	    });
	}).catch(err => {
	    console.log(`getWikiContent Wiki.page: ${err}`);
	    reject(err);
	});
    });
}

//

function getScore(wordList, result) {
    expect(wordList).to.be.an('array');
    expect(result).to.be.an('object');

    return new Promise(function(resolve, reject) {
	let score = {
	    wordsInTitle   : _.isString(result.title) ? getWordCountInText(wordList, result.title) : 0,
	    wordsInSummary : _.isString(result.summary) ? getWordCountInText(wordList, result.summary) : 0,
	    disambiguation : getDisambiguation(result)
	};
	
	if (score.wordsInSummary < _.size(wordList)) {
	    getWikiContent(removeWikipediaSuffix(result.title))
		.then(content => {
		    score.wordsInArticle = getWordCountInText(
			wordList, `${content.text} ${_.values(content.info).join(' ')}`);
		}).
		catch(err => {
		    console.log(`getScore: ${err}`);
		}).then(() => {
		    resolve(score);
		});
	} else {
	    resolve(score);
	}
    });
}

//

function scoreResultList(wordList, resultList, options) {
    expect(wordList, 'wordList').to.be.an('array');
    expect(resultList, 'resultList').to.be.an('array');

    return new Promise((resolve, reject) => {
	let any = false;
	// convert space-separated words to separate words
	wordList = _.chain(wordList).map(word => word.split(' ')).flatten().value();
	console.log('wordList: ' + wordList);
	Promise.mapSeries(resultList, (result, index) => {
	    // only get score if it's not present, or force flag
	    if (!result.score || options.force) {
		return getScore(wordList, result)
		    .then(score => {
			// result.score = score; ?
			resultList[index].score = score;
			any = true;
		    });
		// TODO: no .catch()
	    }
	}).then(unused => {
	    resolve(any ? resultList : []);
	});
	// TODO: no .catch()
    });
}

//

module.exports = {
    getScore              : getScore,
    getWikiContent        : getWikiContent,
    scoreResultList       : scoreResultList,
    removeWikipediaSuffix : removeWikipediaSuffix
};
