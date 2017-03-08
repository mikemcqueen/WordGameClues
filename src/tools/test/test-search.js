//
// TEST-SEARCH.JS
//

'use strict';

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const My           = require('../../misc/util');
const Search       = require('../search-mod');
const Result       = require('../result-mod');

//

const TEST_ROOT    = `${__dirname}/`
const TMP_DIRNAME  = 'tmp';
const TMP_DIR      = `${TEST_ROOT}${TMP_DIRNAME}/`;

//

function createTmpFileSync (filename) {
    Expect(filename).to.be.a('string');
    console.log(`creating: ${filename}`);
    let args = {
	root: TEST_ROOT,
	dir:  TMP_DIRNAME,
	base: filename
    };
    try {
	Fs.writeFileSync(Result.pathFormat(args), '[]');
    } catch(err) {
	throw err;
    }
}

//

function deleteTmpFileSync (filename) {
    Expect(filename).to.be.a('string');
    console.log(`deleting: ${filename}`);
    let args = {
	root: TEST_ROOT,
	dir:  TMP_DIRNAME,
	base: filename
    };
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
    const wordList = [ 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten' ];

    ////////////////////////////////////////////////////////////////////////////////

    // build a 2-element array of unique 2-element word lists
    // e.g. [ [ 'one', 'two' ], [ 'three', 'four' ] ]
    function getWordListArray () {
	function getPair(other = []) {
	    const pair = [ _.sample(wordList), _.sample(wordList) ];
	    if (!_.isEqual(pair, other)) return pair;
	    return getPair(other);
	}
	let firstPair = getPair();
	return [ firstPair, getPair(firstPair) ];
    }

    ////////////////////////////////////////////////////////////////////////////////

    // test skip-when-file-exists functionality
    it ('should skip [one,two] because file exists, then process [three,four]', function (done) {
	let wla = getWordListArray();
	createTmpFileSync(Result.makeFilename(wla[0]));	// create one-two.json
	deleteTmpFileSync(Result.makeFilename(wla[1]));	// create three-four.json

	Search.getAllResults({
	    wordListArray: wla,
	    pages:         1,
	    delay:         delay,
	    root:          TEST_ROOT,
	    dir:           TMP_DIRNAME,
	}).then(result => {
	    Expect(result.skip, 'skip').to.equal(1);
	    Expect(result.data, 'data').to.equal(1);
	    Expect(result.error, 'error').to.equal(0);
	    done();
	}).catch(err => done(err));
    });
    
    ////////////////////////////////////////////////////////////////////////////////

    // test forced rejection in Search.getOneResult
    it ('should skip [one,two] because of forced rejection, then process [three,four]', function (done) {
	let wla = getWordListArray();
	let options = {
	    force: true,          // search even if file exists
	    forceNextError: true  // reject in getOneResult
	};
	Search.getAllResults({
	    wordListArray:  wla,
	    pages:          1,
	    delay:          delay,
	    root:           TEST_ROOT,
	    dir:            TMP_DIRNAME
	}, options).then(result => {
	    Expect(result.skip, 'skip').to.equal(0);
	    Expect(result.data, 'data').to.equal(1);
	    Expect(result.error, 'error').to.equal(1);
	    done();
	}).catch(err => done(err));
    });

    ////////////////////////////////////////////////////////////////////////////////

    it ('should process both results successfully', function(done) {
	let wla = getWordListArray();
	// for testing wihtout --force, need to delete both files
	deleteTmpFileSync(Result.makeFilename(wla[0]));	  // delete one-two.json
	deleteTmpFileSync(Result.makeFilename(wla[1]));	  // delete three-four.json

	Search.getAllResults({
	    wordListArray:  wla,
	    pages:          1,
	    delay:          delay,
	    root:           TEST_ROOT,
	    dir:            TMP_DIRNAME
	}).then(result => {
	    Expect(result.skip, 'skip').to.equal(0);
	    Expect(result.data, 'data').to.equal(2);
	    Expect(result.error, 'error').to.equal(0);
	    done();
	}).catch(err => done(err));
    });

    ////////////////////////////////////////////////////////////////////////////////
    // TODO:

    it ('should test bird-pest search, results look funny, duplicated title and 2nd result is empty url');

});
