//
// TEST-SCORE.JS
//

'use strict';

const _      = require('lodash');
const Expect = require('chai').expect;
const Score  = require('../score-mod.js');

function showScores(score) {
    _.keys(score).forEach(key => {
	console.log(`${key} : ${score[key]}`);
    });
}

describe('test some scoring scenarios', function() {
    this.timeout(5000);
    this.slow(2000);

    it('should count word that only exists in infobox in wordsInArticle', function(done) {
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

    // Test space-separated "compound" words
    // NOTE: this test could be skipped, because it is not how we're
    // currently calling getScore - we split space-separated words first.
    // but this functionality works, if I ever want to re-enable it.
    it('should count both words in title, and only single word in summary', function(done) {
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

/* TODO:  test why wordsInArticle < wordsInSummary
    "title": "1000 Ways to Die (season 3, 2012) - Wikipedia",
    "url": "https://en.wikipedia.org/wiki/1000_Ways_to_Die_(season_3,_2012)",
*/

});

