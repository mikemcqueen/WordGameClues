//
// test-file.js
//

'use strict';

const _      = require('lodash');
const Score  = require('../score-mod.js');
const fs     = require('fs');
const expect = require('chai').expect;

function testFile() {
    const content = fs.readFileSync('./noInfoBox-testfile.json');
    Score.processResultFile('bureau-chief.json', content, {
	force: true
    }).catch(err => {
	console.error('testScore: ' + err);
    }).then(list => {
	console.log('done: ' + _.size(list));
	let result = list[0];
	//list.forEach(result => 
	console.log('score: ' + JSON.stringify(result.score));

	expect(result.score)
	    .to.have.property('wordsInArticle')
	    .and.to.equal(2);
    });
}

try {
    testFile();
}
catch(err) {
    console.log(err);
}

