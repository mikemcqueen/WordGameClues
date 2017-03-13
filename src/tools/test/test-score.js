//
// TEST-SCORE.JS
//

'use strict';

const _      = require('lodash');
const Expect = require('chai').expect;
const Score  = require('../score-mod.js');

//

describe ('test some scoring scenarios', function () {
    this.timeout(5000);
    this.slow(2000);

    function showScores(score) {
	for (const key of _.keys(score)) {
	    console.log(`${key} : ${score[key]}`);
	}
    }

    ////////////////////////////////////////////////////////////////////////////////

    it ('should count word that only exists in infobox in wordsInArticle', function (done) {
	const wordList = [ 'betsy', 'king' ]; // betsy is only in infobox
	const searchResult = {
	    title:   'Lizzie Lloyd King - Wikipedia',
	    summary: 'Only King not the etsyBay word here'
	};
	Score.getScore(wordList, searchResult).then(score => {
	    showScores(score);
	    Expect(score.wordsInArticle).to.equal(2);
	    done();
	});
    });

    ////////////////////////////////////////////////////////////////////////////////

    // Test space-separated "compound" words
    // NOTE: this test is skipped, because it is not how getScore is currently
    // called - space-separated words get split first. But this functionality
    // is possible if I ever want to re-enable it.
    it.skip ('should count both words in title, and only single word in summary', function (done) {
	const wordList = [ 'randolph campbell', 'anne' ];
	const searchResult = {
	    // no title
	    title:   'Randolph Campbell Anne',
	    summary: 'Randolph Anne Campbell'
	};
	Score.getScore(wordList, searchResult).then(score => {
	    showScores(score);
	    Expect(score.wordsInTitle).to.equal(2);
	    Expect(score.wordsInSummary).to.equal(1);
	    done();
	});
    });

    ////////////////////////////////////////////////////////////////////////////////

    it ('should get wiki page content, report word count', function (done) {
	const title = 'Lizzie Lloyd King - Wikipedia';
	Score.getWikiContent(title)
	    .then(content => {
		//console.log(content.text + ' ' + _.values(content.info).map(value => value).join(' '));
		console.log(`content(${_(content.text).words().size()}) ` +
                            `info(${_(content.info).words().size()})`);
		done();
	    });
    });

    ////////////////////////////////////////////////////////////////////////////////
    // "title": "USS Munalbro (1916) - Wikipedia",
    // "url": "https://en.wikipedia.org/wiki/USS_Munalbro_(1916)",
    //
    // NOTE: problem here was capital SS, can reuse this test case for something else
    //
    it.skip ('should test for correct wordsInArticle count', function (done) {
	const title = 'USS Munalbro (1916) - Wikipedia';
	const wordList = [ 'ss', 'james' ];
	const options = { verbose: true };
	Score.getWikiContent(title)
	    .then(content => {
		//console.log(content.text + ' ' + _.values(content.info).map(value => value).join(' '));
		console.log(`content(${_(content.text).words().size()}) ` +
                            `info(${_(content.info).words().size()})`);
		let wordsInArticle = Score.getWordCount(
		    wordList, `${content.text} ${_.values(content.info).join(' ')}`, options);
		console.log(`wordsInArticle: ${wordsInArticle}`);
		Expect(wordsInArticle).to.equal(2);
		done();
	    }).catch(err => {
		console.log('error');
		done(err);
	    });
    });
});
