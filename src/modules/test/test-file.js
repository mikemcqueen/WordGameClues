//
// test-file.js // TODO: TEST-RESULTFILE.JS
//

'use strict';

const _       = require('lodash');
const Expect  = require('chai').expect;
const Fs      = require('fs');
const Score   = require('../score.js');

//

const TESTFILES_DIR = `${__dirname}/test-files/`;

//

function testFile (filename, wordList) {
    return new Promise((resolve, reject) => {
	const content = Fs.readFileSync(TESTFILES_DIR + filename);
	return Score.scoreResultList(
	    wordList,
	    JSON.parse(content),
	    { force: true }
	).catch(err => {
	    reject(err);
	}).then(list => {
	    console.log('done: ' + _.size(list));
	    resolve(list);
	});
    });
}

//

describe ('file tests:', function() {
    this.timeout(5000);
    this.slow(2000);
    
////////////////////////////////////////////////////////////////////////////////    

    it ('should score a page with no InfoBox', function (done) {
	testFile('noInfoBox.json', [ 'bureau', 'chief' ])
	    .then(list => {
		let result = list[0];
		console.log('score: ' + JSON.stringify(result.score));

		Expect(result.score.wordsInArticle, 'wordsInArticle').to.equal(2);
		done();
	    });
    });
	    
////////////////////////////////////////////////////////////////////////////////    

    it ('should not fail to score an invalid page title', function (done) {
	testFile('noContent.json', [ 'what', 'ever' ])
	    .then(list => {
		let result = list[0];
		console.log('score: ' + JSON.stringify(result.score));

		Expect(list, 'list length').to.have.lengthOf(1);
		done();
	    });
    });
	    
////////////////////////////////////////////////////////////////////////////////    

    it.skip ('should score a list of results', function (done) {
	this.timeout(20 * 5000);
	this.slow(20 * 2000);

	testFile('twentyResults.json', [ 'betsy', 'ariana' ])
	    .then(list => {
		list.forEach(result => {
		    console.log('score: ' + JSON.stringify(result.score));
		});

		Expect(list, 'list length').to.have.lengthOf(20);
		done();
	    });
    });
});
