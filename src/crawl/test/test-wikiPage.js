//
// TEST-WIKIPAGE.JS
//

'use strict';

const _     = require('lodash');
const Score = require('../score-mod')

function testWikiPage() {
    const title = 'Lizzie Lloyd King - Wikipedia';
    Score.getWikiContent(Score.removeWikipediaSuffix(title))
	.then(content => {
	    console.log(content.text + ' ' + _.values(content.info).map(value => value).join(' '));
	});
}

module.exports = testWikiPage;

testWikiPage();

