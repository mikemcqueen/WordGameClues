//
// TEST-SCORE.JS
//

'use strict';

const _     = require('lodash');
const Score = require('../score-mod.js');


function testScore1() {
    const words = [ 'betsy', 'king' ];
    const searchResult = {
	title:   'Lizzie Lloyd King - Wikipedia',
	url:     'unused',
	summary: 'Only King not the B word here'
    };
    Score.getScore(words, searchResult).then(score => {
	_.keys(score).forEach(key => {
	    console.log(`${key} : ${score[key]}`);
	});
    });
}

function testScore() {
    testScore1();
}

testScore();

