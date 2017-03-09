//
// SCORE-MOD.JS
//

'use strict';

const _       = require('lodash');
const Promise = require('bluebird');
const Wiki    = require('wikijs').default;
const Expect  = require('chai').expect;

//

const WIKIPEDIA_SUFFIX         = ' - Wikipedia';
const WIKIPEDIA_DISAMBIGUATION = 'disambiguation';

//

function wordCountFilter (score, wordCount, options) {
    Expect(score).to.be.an('object');
    Expect(wordCount).to.be.a('number');
    Expect(options).to.be.an('object');
    let passTitle = false;
    let passArticle = false;
    if (options.filterTitle) {
	passTitle = (score.wordsInTitle >= wordCount);
    }
    if (options.filterArticle) {
	passArticle = (score.wordsInSummary >= wordCount ||
		       score.wordsInArticle >= wordCount);
    }
    return (passTitle || passArticle);
}

//

function getWordCount (wordList, text, options = {}) {
    if (_.isString(wordList)) { wordList = [ wordList ]; } // or .split(',')
    Expect(wordList).to.be.an('array').that.is.not.empty;
    Expect(text).to.be.a('string');

    let textWordList = _.words(text.toLowerCase());
    let countList = new Array(wordList.length).fill(0);
    textWordList.forEach((textWord, textIndex) => {
	wordList.forEach((word, index) => {
	    // NOTE: words have already been split before this function
	    // is called. this code was previously used for a different 
	    // multi-word lookup strategy (which may be re-employed).
	    let nextTextWord = textWord;
	    if (word.split(' ').every((subWord, subIndex, subWordList) => {
		//if (_.startsWith(nextTextWord, subWord)) {
		if (nextTextWord !== subWord) {
		    if (options.verbose) {
			console.log(`${nextTextWord} !== ${subWord}`);
		    }
		    return false; // every.exit
		}
		if (options.verbose) {
		    console.log(`${nextTextWord} === ${subWord}`);
		}
		if (subIndex < subWordList.length - 1) {
		    // we're before the last element in subword list,
		    // ty to set the next textWord.
		    let nextTextIndex = textIndex + subIndex + 1;
		    if (nextTextIndex < textWordList.length) {
			nextTextWord = textWordList[nextTextIndex];
		    } else {
			// there aren't enough words from the text remaining
			return false; // every.exit
		    }
		}
	        return true; // every.continue
	     })) {
 	        countList[index] += 1;
		if (options.verbose) {
	            console.log(`count[${index}] = ${countList[index]}`);
		}
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

function getDisambiguation (result, options) {
    Expect(result).to.exist;
    return _.isString(result.title) ? (getWordCount(WIKIPEDIA_DISAMBIGUATION, result.title, options) > 0) : false;
}

//

function removeWikipediaSuffix (title) {
    Expect(title).to.be.a('string');
    const index = title.lastIndexOf(WIKIPEDIA_SUFFIX);
    if (index !== -1) {
	title = title.substr(0, index);
    }
    return title;
}

//

function getWikiContent (title) {
    Expect(title).to.be.a('string');
    return Wiki().page(removeWikipediaSuffix(title)).then(page => {
	return Promise.all([
	    Promise.resolve(page.content()).reflect(),
	    Promise.resolve(page.info()).reflect()
	]).then(allResults => Object({
	    text: allResults[0].isFulfilled() ? allResults[0].value() : '',
	    info: allResults[1].isFulfilled() ? allResults[1].value() : {}
	})).catch(err => {
	    // TODO: use VError?
	    console.log(`getWikiContent promise.all error, ${err}`);
	    if (err) throw err;
	});
    }).catch(err => {
	console.log(`getWikiContent Wiki.page error, ${err}`);
	if (err) throw err;
    });
}

//

function getScore (wordList, result, options = {}) {
    Expect(wordList).to.be.an('array').that.is.not.empty;
    Expect(result).to.be.an('object');
    let score = {
	wordsInTitle   : _.isString(result.title)   ? getWordCount(wordList, result.title, options)   : 0,
	wordsInSummary : _.isString(result.summary) ? getWordCount(wordList, result.summary, options) : 0,
	disambiguation : getDisambiguation(result, options)
    };
    // if summary has all our words, we don't need to search article
    if (score.wordsInSummary >= _.size(wordList)) {
	return Promise.resolve(score);
    }
    return getWikiContent(result.title)
	.then(content => {
	    score.wordsInArticle = getWordCount(
		wordList, `${content.text} ${_.values(content.info).join(' ')}`, options);
	}).
	catch(err => {
	    // eat error
	    console.log(`getScore error, ${err}`);
	}).then(() => score);
}

//

function scoreResultList (wordList, resultList, options = {}) {
    Expect(wordList, 'wordList').to.be.an('array').that.is.not.empty;
    Expect(resultList, 'resultList').to.be.an('array');
    // convert space-separated words to separate words
    wordList = _.chain(wordList).map(word => word.split(' ')).flatten().value();
    console.log('wordList: ' + wordList);
    let any = false;
    return Promise.mapSeries(resultList, (result, index) => {
	// skip scoring if score already present, unless force flag set
	if (!_.isUndefined(result.score) && !options.force) {
	    return undefined;
	}
	return getScore(wordList, result, options)
	    .then(score => {
		// TODO: result.score = score; ?
		resultList[index].score = score;
		any = true;
	    });
	// TODO: no .catch()
    }).then(() => any ? resultList : []);
    // TODO: no .catch()
}

//

module.exports = {
    getScore,
    getWikiContent,
    getWordCount,
    scoreResultList,
    wordCountFilter
};
