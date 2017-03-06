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

const _args = { root: './', dir:  'tmp' };

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

// args to Search.getAllResults:
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
    // NOTE: csvParse parses csv file in reverse order, so simulate that in word lists here
    let wla1234 = [ [ 'three', 'four' ], [ 'one', 'two' ] ];

    // test skip-when-file-exists functionality
    it('should skip [one,two] because file exists, then process [three,four]', function(done) {
	let wla = wla1234;
	createFile(wla[1]);	// create one-two.json
	deleteFile(wla[0]);	// delete three-four.json
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
	    Expect(result.error, 'error').to.be.undefined;
	    done();
	});
    });

    // test forced rejection in Search.getOneResult
    it('should skip [one,two] because of forced rejection, then process [three,four]', function(done) {
	let wla = wla1234;
	deleteFile(wla[1]);	// delete one-two.json
	deleteFile(wla[0]);	// delete three-four.json
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
	let wla = wla1234;
	deleteFile(wla[1]);	// delete one-two.json
	deleteFile(wla[0]);	// delete three-four.json
	// get the results
	Search.getAllResults({
	    wordListArray:  wla,
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
