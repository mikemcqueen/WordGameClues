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

describe ('result tests:', function() {
    this.timeout(6000);
    this.slow(4000);

    ////////////////////////////////////////////////////////////////////////////////
    //
    // test Result.fileScoreSaveCommit
    //
    // TODO: all the removing/adding is unnecessary. just make sure the file is there
    // in before(), then writeAdd if it isn't
    //
    it ('should remove/commit, then add/commit a file to git', function (done) {
	let wordList = [ 'betsy', 'ariana' ];
	let options = { force: true };
	let srcFilepath = TESTFILES_DIR + 'oneResult.json';
	let filepath = TESTFILES_DIR + 'oneResult-copy.json';
	My.gitRemoveCommitIfExists(filepath, `removing ${filepath}`)
	    .then(() => {
		console.log(`reading file ${srcFilepath}`);
		return fsReadFile(srcFilepath, 'utf8');
	    }).then(data => {
		console.log(`writing ${filepath}`);
		return fsWriteFile(filepath, data);
	    }).then(() => { // TODO: actually only need to Add here, not AddCommit
		console.log('calling gitAddCommit');
		return My.gitAddCommit(filepath, 'adding test file');
	    }).then(() => {
		console.log('calling fileScoreSaveCommit');
		return Result.fileScoreSaveCommit(filepath, options, wordList);
	    }).then(() => { // TODO: removeCommit when we're done?
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
