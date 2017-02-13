//
// TEST-SCORE.JS
//

'use strict';

const _      = require('lodash');
const expect = require('chai').expect;
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
	    summary: 'Only King not the B word here'
	};
	Score.getScore(wordList, searchResult).then(score => {
	    showScores(score);
	    expect(score)
		.to.have.property('wordsInArticle')
		.and.to.equal(2);
	    done();
	});
    });

    it('should count compound and single word in title, and only single word in summary', function(done) {
	const wordList = [ 'randolph campbell', 'anne' ]
	const searchResult = {
	    // no title
	    title:   'Randolph Campbell Anne',
	    summary: 'Randolph Anne Campbell'
	};
	Score.getScore(wordList, searchResult).then(score => {
	    showScores(score);
	    expect(score)
		.to.have.property('wordsInTitle')
		.and.to.equal(2);
	    expect(score)
		.to.have.property('wordsInSummary')
		.and.to.equal(1);
	    done();
	});
    });

});

