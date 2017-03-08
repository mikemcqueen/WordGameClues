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

// TODO: move to util/test. this is useful for testing git primitives.
//
function removeCommitIfExists (filepath, message = 'removing test file') {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return My.checkIfFile(filepath)
	.then(result => {
	    if (!result.exists) {
		return undefined;
	    }
	    // file exists, try to git remove/commit it
	    console.log(`git-removing ${filepath}`);
	    return My.gitRemoveCommit(filepath, message)
		.then(() => My.checkIfFile(filepath))
		.then(result => {
		    Expect(result.exists).to.be.false;
		    console.log(`git-removed ${filepath}`);
		    return undefined;
		}).catch(err => {
		    console.log(`gitRemoveCommit error, ${err}`);
		});
	    // TODO: add "git reset HEAD -- filename" if we had git remove error or file
	    // still exists; file may be added but not commited.
	}).catch(err => {
	    console.log(`removeCommitIfExists error, ${err}`);
	});
}

//

function removeCommitIfExistsV1 (filepath, message) {
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

    ////////////////////////////////////////////////////////////////////////////////
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

    ////////////////////////////////////////////////////////////////////////////////
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
});


describe('google one word pair, show result count', function() {

    this.slow(2000);
    this.timeout(5000);

    it ('should get results for one word pair', function(done) {
	const wiki = 'site:en.wikipedia.org';
	Result.get('one two ' + wiki, 1, function(err, result) {
	    if (err) {
		console.log(`error, ${err}`);
		done(err);
	    } else {
		console.log(`results (${_.size(result)})`);
		done();
	    }
	});
    });

});
