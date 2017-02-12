//
// test-file.js
//

'use strict';

const _      = require('lodash');
const Score  = require('../score-mod.js');
const fs     = require('fs');
const expect = require('chai').expect;

function testFile(filename, wordList) {
    return new Promise((resolve, reject) => {
	const content = fs.readFileSync(filename);
	
	return Score.scoreResultList(
	    wordList,
	    JSON.parse(content),
	    { force: true }
	).catch(err => {
	    console.error('testScore: ' + err);
	}).then(list => {
	    console.log('done: ' + _.size(list));
	    return list;
	});
    });
}

try {
    //
    testFile('./noInfoBox-testfile.json', [ 'bureau', 'chief' ])
	.then(list => {
	    let result = list[0];
	    console.log('score: ' + JSON.stringify(result.score));
	    expect(result.score)
		.to.have.property('wordsInArticle')
		.and.to.equal(2);
	});
	    
    //

    testFile('./file-testfile.json', [ 'betsy', 'ariana' ])
	.then(list => {
	    list.forEach(result => {
		console.log('score: ' + JSON.stringify(result.score));
	    });
	    expect(_.size(list))
		.to.equal(20);
	});
}
catch(err) {
    console.log(err);
}

