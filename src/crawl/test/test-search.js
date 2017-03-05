//
// TEST-SEARCH.JS
//

'use strict';

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Search       = require('../search-mod');
const Result       = require('../result-mod');
const GoogleResult = require('../googleResult');

//

let _args = { root: './', dir:  'tmp' };

//

function createFile(wordList) {
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

function deleteFile(wordList) {
    let filename = Result.makeFilename(wordList); 
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


// args to getAllResults:
// 
// wordListArray : array of array of strings
// pages         : # of pages of results to retrieve for each wordlist
// delay         : object w/properties: high,low; ms delay between searches
// root          : root results directory (optional; default: Results.dir)
// dir           : directory within root to store results (optional; default: wordList.length)

describe('search tests:', function() {
    this.timeout(6000);
    this.slow(4000);

    let delay = { low: 500, high: 1000 };
    let wla1234 = [ [ 'three', 'four' ], [ 'one', 'two' ] ];

    // NOTE: csvParse parses csv file in reverse order, so simulate that in word lists here

    // test "skip" throwing an error for control flow
    it('should skip [one,two] because file exists, then process [three,four]', function(done) {
	let wla = _.clone(wla1234);
	// create one-two.json
	createFile(wla[1]);
	// delete three-four.json
	deleteFile(wla[0]);
	// get the results
	Search.getAllResults({
	    wordListArray: wla,
	    pages:         1,
	    delay:         delay,
	    root:          _args.root,
	    dir:           _args.dir
	}, function(err, result) {
	    if (err) throw err;
	    Expect(result.skip, 'skip').to.equal(1);
	    Expect(result.data, 'data').to.equal(1);
	    Expect(result.error, 'error').to.equal(1);
	    done();
	});
    });

    it('should skip [one,two] because of forced rejection, then process [three,four]', function(done) {
	let wla = _.clone(wla1234);
	// delete one-two.json
	deleteFile(wla[1]);
	// delete three-four.json
	deleteFile(wla[0]);
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
	    Expect(result.skip, 'skip').to.be.undefined;
	    Expect(result.data, 'data').to.equal(1);
	    Expect(result.error, 'error').to.equal(1);
	    done();
	});
    });

    it('should process both results successfully', function(done) {
	let wla = _.clone(wla1234);
	// delete one-two.json
	deleteFile(wla1234[1]);
	// delete three-four.json
	deleteFile(wla1234[0]);
	// get the results
	Search.getAllResults({
	    wordListArray:  wla1234,
	    pages:          1,
	    delay:          delay,
	    root:           _args.root,
	    dir:            _args.dir,
	}, function(err, result) {
	    if (err) throw err;
	    Expect(result.skip, 'skip').to.be.undefined;
	    Expect(result.data, 'data').to.equal(2);
	    Expect(result.error, 'error').to.be.undefined;
	    done();
	});
    });

});
