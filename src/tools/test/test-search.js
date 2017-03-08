//
// TEST-SEARCH.JS
//

'use strict';

const _            = require('lodash');
//const Promise      = require('bluebird');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const My           = require('../../misc/util');
const Search       = require('../search-mod');
const Result       = require('../result-mod');

//

const _args = { root: `${__dirname}/`, dir:  'test-files' };

const TEST_DIR = `${__dirname}/test-files/`;

//

function removeCommitIfExists (filepath, message) {
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
	    // TODO: add "git reset" if we had git remove error or file
	    // still exists; file may be added but not commited.
	}).catch(err => {
	    console.log(`removeCommitIfExists error, ${err}`);
	});
}

//

function createFileSync (wordList) {
    let filename = Result.makeFilename(wordList); 
    console.log(`creating: ${filename}`);
    let args = _.clone(_args);
    args.base = filename;
    try {
	Fs.writeFileSync(Result.pathFormat(args), '[]');
    } catch(err) {
	throw err;
    }
}

//

function deleteFileSync (filename) {
    Expect(filename).to.be.a('string');
    console.log(`deleting: ${filename}`);
    let args = _.clone(_args);
    args.base = filename;
    try {
	Fs.unlinkSync(Result.pathFormat(args));
    } catch(err) {
	if (err.code !== 'ENOENT') {
	    throw err;
	}
	// else eat it
    }
}

// args to Search.getAllResults:
// 
// wordListArray : array of array of strings
// pages         : # of pages of results to retrieve for each wordlist
// delay         : object w/properties: high,low; ms delay between searches
// root          : root results directory (optional; default: Results.dir)
// dir           : directory within root to store results (optional; default: wordList.length)

describe ('search tests:', function() {
    this.timeout(8000);
    this.slow(4000);

    let delay = { low: 500, high: 1000 };

    let wla1234 = [ [ 'one', 'two' ], [ 'three', 'four' ] ];

////////////////////////////////////////////////////////////////////////////////

    // test skip-when-file-exists functionality
    it ('should skip [one,two] because file exists, then process [three,four]', function (done) {
	let wla = wla1234;

	let file1 = Result.makeFilename(wla[1]); 
	removeCommitIfExists(TEST_DIR + file1, 'removing test file')
	    .then(() => {
		createFileSync(wla[0]);	// create one-two.json
		deleteFileSync(file1);	// delete three-four.json
		return undefined;
	    }).then(() => {
		return Search.getAllResults({
		    wordListArray: wla,
		    pages:         1,
		    delay:         delay,
		    root:          _args.root,
		    dir:           _args.dir
		});
	    }).then(result => {
		Expect(result.skip, 'skip').to.equal(1);
		Expect(result.data, 'data').to.equal(1);
		Expect(result.error, 'error').to.equal(0);
		done();
	    }).catch(err => done(err));
    });

////////////////////////////////////////////////////////////////////////////////

    // test forced rejection in Search.getOneResult
    it.skip ('should skip [one,two] because of forced rejection, then process [three,four]', function (done) {
	let wla = wla1234;
	deleteFileSync(wla[1]);	// delete one-two.json
	deleteFileSync(wla[0]);	// delete three-four.json
	// get the results
	Search.getAllResults({
	    wordListArray:  wla,
	    pages:          1,
	    delay:          delay,
	    root:           _args.root,
	    dir:            _args.dir,
	    forceNextError: true
	}, function(err, result) {
	    if (err) throw err;
	    Expect(result.skip, 'skip').to.equal(0);
	    Expect(result.data, 'data').to.equal(1);
	    Expect(result.error, 'error').to.equal(1);
	    done();
	});
    });

////////////////////////////////////////////////////////////////////////////////

    it.skip('should process both results successfully', function(done) {
	let wla = wla1234;
	deleteFile(wla[1]);	// delete one-two.json
	deleteFile(wla[0]);	// delete three-four.json
	// get the results
	Search.getAllResults({
	    wordListArray:  wla,
	    pages:          1,
	    delay:          delay,
	    root:           _args.root,
	    dir:            _args.dir
	}, function(err, result) {
	    if (err) throw err;
	    Expect(result.skip, 'skip').to.equal(0);
	    Expect(result.data, 'data').to.equal(2);
	    Expect(result.error, 'error').to.equal(0);
	    done();
	});
    });

/* TODO:  test bird-pest search, results look funny, duplicated title and 2nd result is empty url
*/


});
