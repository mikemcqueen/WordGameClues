//
// SEARCH-MOD.JS
//

'use strict';

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Ms           = require('ms');
const My           = require('../misc/util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');
const Result       = require('./result-mod');

const fsWriteFile  = Promise.promisify(Fs.writeFile);

// make a search term from a list of words and the supplied options
//
function makeSearchTerm (wordList, options = {}) {
    Expect(options).is.an('object');
    let term = wordList.join(' ');
    if (options.wikipedia) {
	term += ' site:en.wikipedia.org';
    }
    return term;
}

// 
//
function getOneResult (wordList, pages, options = {}) {
    Expect(options).is.an('object');
    return new Promise((resolve, reject) => {
	let term = makeSearchTerm(wordList, { wikipedia: true });
	console.log(`term: ${term}, pages: ${pages}`);
	Result.get(term, pages, (err, data) => {
	    if (!err && options.reject) {
		err = new Error('getOneResult: forced rejection');
	    }
	    if (err) {
		console.log('getOneResult: rejecting');
		reject(err);
	    } else {
		resolve(data);
	    }
	});
    });
}

//args:
// wordListArray : array of array of strings
// pages         : # of pages of results to retrieve for each wordlist
// delay         : object w/properties: high,low - ms delay between searches
// root          : root results directory (optional; default: Results.dir)
// dir           : directory within root to store results (optional, default: wordList.length)
//
// optiosn:
//   force       : search even if results file already exists (overwrites. TODO: append new results, instead)
//   forceNextError: test support, sets getOnePromise.options.reject one time
//
async function getAllResultsLoop (args, options = {}) {
    Expect(args).to.exist;
    Expect(args.wordListArray).to.be.an('array').that.is.not.empty;
    Expect(options).to.be.an('object');
    let result = { skip: 0, empty: 0, data: 0, error: 0 }; // test support
    for (const [index, wordList] of args.wordListArray.entries()) {
	let filename = Result.makeFilename(wordList);
	console.log(`list: ${wordList}`);
	console.log(`file: ${filename}`);
	let path = Result.pathFormat({
	    root:  args.root,
	    dir:  _.isUndefined(args.dir) ? _.toString(wordList.length) : args.dir,
	    base: filename
	});
	let nextDelay = 0;
	let saved = await My.checkIfFile(path)
	    .then(checkResult => {
		// skip this file if it already exists, unless force flag set
		if (checkResult.exists && !options.force) { 
		    result.skip += 1;
		    console.log(`Skip: file exists, ${path}`); 
		    return Promise.reject();
		}
		let oneResultOptions = { reject : options.forceNextError };
		options.forceNextError = false;
		// file does not already exist; do the search
		return getOneResult(wordList, args.pages, oneResultOptions);
	    }).then(oneResult => {
		// we successfully searched. set delay for next search
		nextDelay = My.between(args.delay.low, args.delay.high);
		if (_.isEmpty(oneResult)) {
		    result.empty += 1;
		    return Promise.reject();
		}
		result.data += 1;
		return fsWriteFile(path, JSON.stringify(oneResult)).then(() => {
		    console.log(`Saved: ${path}`);
		    return true; // saved
		});
	    }).catch(err => {
		// log & eat all errors (err might be undefined from Promise.reject())
		if (err) {
		    console.log(`getAllResultsLoop error, ${err}`);
		    result.error += 1;
		}
	    });

	if (saved === true) {
	    // NOTE: we intentionally do NOT await completion of the following
	    // add-commit-score-commit operations. they can be executed asynchronously
	    // with the execution of this loop. makes logs a little messier though.
	    My.gitAddCommit(path, 'new result')
		.then(() => {
		    console.log(`Committed: ${path}`);
		    return Result.fileScoreSaveCommit(path);
		}).catch(err => {
		    // log & eat all errors
		    console.log(`getAllResultsLoop commit error, ${err}`);
		});
	}

	// if there are more wordlists to process
	if (index < args.wordListArray.length - 1) {
	    // if nextDelay is specified, delay before next search
	    if (nextDelay > 0) {
		console.log(`Delaying ${PrettyMs(nextDelay)} for next search...`);
		await My.waitFor(nextDelay);
	    }
	}
    }
    return result;
}

//
//
function getAllResults (args, options = {}) {
    Expect(args).to.be.an('object');
    Expect(args.wordListArray).to.be.an('array').that.is.not.empty;
    return new Promise((resolve, reject) => {
	getAllResultsLoop(args, options)
	    .then(data => resolve(data))
	    .catch(err => reject(err));
    });
}

//

module.exports = {
    getAllResults
}
