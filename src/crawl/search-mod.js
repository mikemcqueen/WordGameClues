//
// SEARCH-MOD.JS
//

'use strict';

const _            = require('lodash');
const Promise      = require('bluebird');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Ms           = require('ms');
const Between      = require('../util/between');
const GoogleResult = require('./googleResult');
const Result       = require('./result-mod');

const fsWriteFile  = Promise.promisify(Fs.writeFile);

// check if path exists, and whether it is a file.
//
function checkIfFile(path) {
    return new Promise((resolve, reject) => {
	Fs.stat(path, (err, stats) => {
	    if (err) {
		if (err.code === 'ENOENT') {
		    resolve({       // does not exist, therefore not a file
			exists: false,
			isFile: false
		    });
		} else {
		    reject(err);
		}
	    } else {
		resolve ({          // exists, maybe is a file
		    exists: true,
		    isFile: stats.isFile()
		});
	    }
	});
    });
}

// make a search term from a list of words and the supplied options
//
function makeSearchTerm(wordList, options) {
    let term = wordList.join(' ');
    if (_.isObject(options) && options.wikipedia) {
	term += ' site:en.wikipedia.org';
    }
    return term;
}

// 
//
function getOneResult(wordList, pages, options) {
    return new Promise((resolve, reject) => {
	let term = makeSearchTerm(wordList, { wikipedia: true });
	console.log(`term: ${term}, pages: ${pages}`);
	GoogleResult.get(term, pages, (err, data) => {
	    if (!err && options.reject) {
		err = new Error('getOneResult: forced rejection');
	    }
	    if (err) {
		console.log('getOneResult: rejecting');
		reject(err);
	    }
	    else {
		resolve(data);
	    }
	});
    });
}

//args:
// wordListArray : array of array of strings
// pages         : # of pages of results to retrieve for each wordlist
// delay         : object w/properties: high,low; ms delay between searches
// root          : root results directory (optional; default: Results.dir)
// dir           : directory within root to store results (optional; default: wordList.length)
// 
// asynchronously "recursive". mutates args.wordListArray by
// calling pop() each time.
//
function mutatingGetAllResults(args, cb) { // optional callback with stats, for testing
    Expect(cb, 'cb').to.be.a('function');
    Expect(args, 'args').to.exist;
    Expect(args.wordListArray, 'wordListArray')
	.to.be.an('array').that.is.not.empty;

    let wordList = args.wordListArray.pop();
    let filename = Result.makeFilename(wordList);
    console.log(`list: ${wordList}`);
    console.log(`file: ${filename}`);
    
    let path = Result.pathFormat({
	root:  args.root,
	dir:  _.isUndefined(args.dir) ? _.toString(wordList.length) : args.dir,
	base: filename
    });
    let nextDelay = 0;
    checkIfFile(path)
	.then(result => {
	    if (result.exists) { 
		args.skip = (args.skip || 0) + 1; // test support
		console.log(`Skip: file exists, ${path}`); 
		return undefined;
	    }
	    // file does not already exist; do the search
	    let options = { reject : args.forceNextError }; // test support
	    args.forceNextError = false;
	    return getOneResult(wordList, args.pages, options)
		.then(data => {
		    if (_.size(data) > 0) {
			args.data = (args.data || 0) + 1; // test support
			return fsWriteFile(path, JSON.stringify(data))
			    .then(() => console.log(`Saved: ${path}`));
		    }
		    args.empty = (args.empty || 0) + 1; // test support
		    return undefined
		}).then(() => {
		    if (args.wordListArray.length > 0) {
			nextDelay = Between(args.delay.low, args.delay.high);
		    }
		});
	    // TODO: .catch()
	})
    	.catch(err => {
	    // eat all errors. continue  with any remaining wordlists.
	    console.log(`getAllResults, ${err.message}`);
	    args.error = (args.error || 0) + 1; // test support
	}).then(() => { // finally
	    //  queue up next call
	    if (args.wordListArray.length > 0) {
		if (nextDelay > 0) {
		    console.log(`Delaying ${PrettyMs(nextDelay)} for next search...`);
		}
		setTimeout(() => getAllResults(args, cb), nextDelay);
	    }
	    else {
		cb(null, args);
	    }
	});
}

// shim to protect args.wordListArray from mutating
//
function getAllResults(args, cb) {
    Expect(args.wordListArray, 'wordListArray')
	.to.be.an('array').that.is.not.empty;
    args.wordListArray = _.clone(args.wordListArray);
    return mutatingGetAllResults(args, cb);
}

//

module.exports = {
    getAllResults  : getAllResults
}
