//
// test-file.js
//

'use strict';

const _       = require('lodash');
//const Promise = require('bluebird');
const Score   = require('../score-mod.js');
const fs      = require('fs');
const expect  = require('chai').expect;

function testFile(filename, wordList) {
    return new Promise((resolve, reject) => {
	const content = fs.readFileSync(filename);
	
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

describe('test score-mod', function() {
    this.timeout(5000);
    this.slow(2000);
    
    //
    it('should score a page with no InfoBox', function(done) {
	testFile('./noInfoBox-testfile.json', [ 'bureau', 'chief' ])
	    .then(list => {
		let result = list[0];
		console.log('score: ' + JSON.stringify(result.score));
		expect(result.score)
		    .to.have.property('wordsInArticle')
		    .and.to.equal(2);
		done();
	    });
    });
	    
    //
    it('should not fail to score an invalid page title', function(done) {
	testFile('./noContent-testfile.json', [ 'what', 'ever' ])
	    .then(list => {
		let result = list[0];
		console.log('score: ' + JSON.stringify(result.score));
		expect(_.size(list))
		    .to.equal(1);
		done();
	    });
    });
	    
    //
    it('should score a list of results', function(done) {
	this.timeout(20 * 5000);
	this.slow(20 * 2000);

	testFile('./file-testfile.json', [ 'betsy', 'ariana' ])
	    .then(list => {
		list.forEach(result => {
		    console.log('score: ' + JSON.stringify(result.score));
		});
		expect(list)
		    .to.have.lengthOf(20);
		done();
	    });
    });
});
