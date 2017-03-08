//
// TEST-RESULT.JS
//

'use strict';

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Git          = require('simple-git');
const My           = require('../../misc/util');
const Path         = require('path');
const Promise      = require('bluebird');
const Result       = require('../result-mod');

const fsReadFile   = Promise.promisify(Fs.readFile);
const fsWriteFile  = Promise.promisify(Fs.writeFile);

//

const TESTFILES_DIR = `${__dirname}/test-files/`;

//

function removeCommitIfExists (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return My.checkIfFile(filepath)
	.then(result => result.exists ? undefined : Promise.reject())
	.then(() => {
	    // file exists, try to git remove/commit it
	    console.log(`calling gitRemoveCommit ${filepath}`);
	    return My.gitRemoveCommit(filepath, message);
	}).then(() => My.checkIfFile(filepath))
	.then(result => {
	    // TODO: add "git reset" if we had git remove error or file
	    // still exists; file may be added but not commited.
	    Expect(result.exists).to.be.false;
	    console.log('file removed');
	    return undefined;
	}).catch(err => {
	    // log real errors, eat all errors
	    if (err) {
		console.log(`removeCommitIfExists error, ${err}`);
	    }
	});
}

//

describe ('result tests:', function() {
    this.timeout(6000);
    this.slow(4000);

    let delay = { low: 500, high: 1000 };

    // NOTE: csvParse parses csv file in reverse order, so simulate that in word lists here
    let wla1234 = [ [ 'three', 'four' ], [ 'one', 'two' ] ];

    // TODO: move these purely git tests to util/test

    //
    // test gitRemoveCommit, gitAddCommit
    //
    it ('should remove/commit, then add/commit a file to git', function (done) {
	let filepath = TESTFILES_DIR + 'test-add-commit';
	removeCommitIfExists(filepath, 'removing test file')
	    .then(() => {
		console.log('writing new file');
		return fsWriteFile(filepath, JSON.stringify('[]'));
	    }).then(() => {
		console.log('committing file');
		return My.gitAddCommit(filepath, 'adding test file');
	    }).then(() => {
		console.log('done');
		done();
	    }).catch(err => {
		console.log(`error, ${err}`);
		done(err);
	    });
    });

    //
    // test Result.fileScoreSaveCommit
    //
    it ('should remove/commit, then add/commit a file to git', function (done) {
	let wordList = [ 'betsy', 'ariana' ];
	let options = { force: true };
	let srcFilepath = TESTFILES_DIR + 'oneResult.json';
	let filepath = TESTFILES_DIR + 'oneResult-copy.json';
	removeCommitIfExists(filepath, `removing ${filepath}`)
	    .then(() => {
		console.log(`reading file ${srcFilepath}`);
		return fsReadFile(srcFilepath, 'utf8');
	    }).then(data => {
		console.log(`writing ${filepath}`);
		return fsWriteFile(filepath, data);
	    }).then(() => {
		console.log('calling gitAddCommit');
		return My.gitAddCommit(filepath, 'adding test file');
	    }).then(() => {
		console.log('calling fileScoreSaveCommit');
		return Result.fileScoreSaveCommit(filepath, options, wordList);
	    }).then(() => {
		console.log('done');
		done();
	    }).catch(err => {
		console.log(`error, ${err}`);
		done(err);
	    });
    });

/*
    it('should remove/commit, then use asynchronous add/commit generator to commit a file to git', function(done) {
	let filepath = TMP_DIR + 'test-async-add-commit';
	let gen = My.gitAddCommitGenerator();
	removeCommitIfExists(filepath, 'removing test file')
	    .then(() => {
		console.log('writing new file');
		return fsWriteFile(filepath, JSON.stringify('[]'));
	    }).then(() => {
		console.log('committing file');
		gen.next(); // start generator
		return gen.next({ filepath, message: `adding test file ${filepath}` });
	    }).then(result => {
		Expect(result.done).to.be.false;
		let err = gen.next().value;
		if (err) throw err;
		console.log('done');
		done ();
	    }).catch(err => {
		console.log(`error, ${err}`);
		done(err);
	    });
    });
*/
    
});


describe('google one word pair, show result', function() {

    it.skip('test one pair', function(done) {
	const wiki = 'site:en.wikipedia.org';
	this.timeout(75000);
	Result.get('one' + 'two' + wiki, function(err, res) {
	    if (err) {
		console.log(err);
	    } else {
		console.log(res);
	    }
	    done();
	});
    });

});
