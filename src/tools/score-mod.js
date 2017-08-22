//
// SCORE-MOD.JS
//

'use strict';

const _       = require('lodash');
const Promise = require('bluebird');
const Wiki    = require('wikijs').default;
const Expect  = require('should/as-function'); // ('chai').expect;

//

const WIKIPEDIA_SUFFIX         = ' - Wikipedia';
const WIKIPEDIA_DISAMBIGUATION = 'disambiguation';

//

function wordCountFilter (score, wordCount, options) {
    Expect(score).is.a.Object();
    Expect(wordCount).is.a.Number();
    Expect(options).is.a.Object();
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

/* cut&paste not ready for use
function getWordCountV2 () {
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
}
*/

//

function getWordCount (wordList, text, options = {}) {
    if (_.isString(wordList)) {
	Expect(_.includes(wordList, ',')).is.false; // easy to support, but no need yet
	wordList = [ wordList ];  // or .split(',')
    }
    Expect(wordList).is.a.Array().which.is.not.empty(); // can be 1, e.g. disambiguation
    Expect(text).is.a.String();
    let textWordList = _.words(text.toLowerCase());
    return _(wordList).map(word => _.includes(textWordList, word))
	.tap(foundList => {
	    if (options.verbose) {
		console.log(`foundList(${foundList.length}) ${_.values(foundList)}`);
	    }
	}).reduce((accum, found, index, foundList) => {
	    if (found) accum += 1;
	    if (options.verbose) {
		console.log(`found[${wordList[index]}] = ${foundList[index]}, accum(${accum})`);
	    }
	    return accum;
	}, 0);
}

//

function getDisambiguation (result, options) {
    Expect(result).is.a.Object();
    return _.isString(result.title) ? (getWordCount(WIKIPEDIA_DISAMBIGUATION, result.title, options) > 0) : false;
}

//

function removeWikipediaSuffix (title) {
    Expect(title).is.a.String();
    const index = title.lastIndexOf(WIKIPEDIA_SUFFIX);
    return (index !== -1) ? title.substr(0, index) : title;
}

//

function getWikiContent (title) {
    Expect(title).is.a.String();
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
	console.log(`getWikiContent Wiki.page, title: ${title}, error: ${err}`);
	if (err) throw err;
    });
}

//

function getScore (wordList, result, options = {}) {
    Expect(wordList).is.a.Array().not.empty();
    Expect(result).is.a.Object();
    // TODO: some conditions may warrant a re-fetch of either the
    // Google search results (null title or summary). then we could 
    // expect valid entries here.
    //Expect(result.title).is.a.String().that.is.not.empty;
    //Expect(result.summary).is.a.String().that.is.not.empty;

    let score = {
	wordsInTitle   : _.isString(result.title)   ? getWordCount(wordList, result.title, options)   : 0,
	wordsInSummary : _.isString(result.summary) ? getWordCount(wordList, result.summary, options) : 0,
	disambiguation : getDisambiguation(result, options)
    };
    Expect(score.wordsInTitle).is.a.Number();

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

// remove empty URLs

function filterBadResults (resultList) {
    return _.filter(resultList, result => {
	return !_.isEmpty(result.url);
    });
}

//

function scoreResultList (wordList, resultList, options = {}) {
    Expect(wordList).is.a.Array().which.is.not.empty();
    Expect(resultList).is.a.Array();

    let resultCount = _.size(resultList);
    resultList = filterBadResults(resultList);
    let anyChange = _.size(resultList) !== resultCount;

    // convert space-separated words to separate words
    wordList = _.flatten(_.map(wordList, word => word.split(' ')));
    console.log('wordList: ' + wordList);
    return Promise.mapSeries(resultList, (result, index) => {
	// skip scoring if score already present, unless force flag set
	if (!_.isUndefined(result.score) && !options.force) {
	    return undefined;
	}
	return getScore(wordList, result, options)
	    .then(score => {
		// TODO: result.score = score; ?
		resultList[index].score = score;
		anyChange = true;
	    });
	// TODO: no .catch()
    }).then(() => anyChange ? resultList : []);
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
