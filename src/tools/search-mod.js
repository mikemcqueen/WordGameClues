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

const FsWriteFile  = Promise.promisify(Fs.writeFile);

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

// check if file exists; if not, get, then save, a search result

function checkGetSaveResult(args, options) {
    let nextDelay = 0;

    return My.checkIfFile(args.path)
	.then(checkResult => {
	    // skip this file if it already exists, unless force flag set
	    if (checkResult.exists && !options.force) { 
		args.count.skip += 1;
		console.log(`Skip: file exists, ${args.path}`); 
		return Promise.reject();
	    }

	    // we are going to do a search; set delay for next search.
	    // NOTE: need to set this here, rather than on successful
	    // search, because we may get robot warning
	    // TODO: add retry to getOneResult; check for robot result
	    nextDelay = args.delay; // My.between(args.delay.low, args.delay.high);

	    let oneResultOptions = { reject: options.forceNextError };
	    options.forceNextError = false;
	    // file does not already exist; do the search
	    return getOneResult(args.wordList, args.pages, oneResultOptions);
	}).then(oneResult => {
	    if (_.isEmpty(oneResult)) {
		// empty search result. lower delay for next search, and
		// save an empty result so we don't do this search again
		args.count.empty += 1;
		oneResult = [];
	    } else {
		args.count.data += 1;
	    }
	    return FsWriteFile(args.path, JSON.stringify(oneResult))
		.then(() => {
		    console.log(`Saved: ${args.path}`);
		    return [oneResult, nextDelay];
		});
	}).catch(err => {
	    // log & eat all errors (err might be undefined from Promise.reject())
	    if (err) {
		console.log(`checkGetSaveResult error, ${err}`);
		args.count.error += 1;
	    }
	    return [null, nextDelay];
	});
}

//args:
// wordListArray : array of array of strings
// pages         : # of pages of results to retrieve for each wordlist
// delay         : object w/properties: high,low - ms delay between searches
// root          : root results directory (optional; default: Results.dir)
// dir           : directory within root to store results (optional, default: wordList.length)
//
// options:
//   force       : search even if results file already exists (overwrites. TODO: append new results, instead)
//   forceNextError: test support, sets getOnePromise.options.reject one time
//
async function getAllResultsLoop (args, options = {}) {
    Expect(args).to.exist;
    Expect(args.wordListArray).to.be.an('array').that.is.not.empty;
    Expect(options).to.be.an('object');
    let count = { skip: 0, empty: 0, data: 0, error: 0 }; // test support
    for (const [index, wordList] of args.wordListArray.entries()) {
	let filename = Result.makeFilename(wordList);
	console.log(`list: ${wordList}`);
	console.log(`file: ${filename}`);

	let path = Result.pathFormat({
	    root: args.root,
	    dir:  args.dir || _.toString(wordList.length),
	    base: filename
	});

	let [result, nextDelay] = await checkGetSaveResult({
	    wordList,
	    path,
	    count,
	    pages: args.pages,
	    delay: My.between(args.delay.low, args.delay.high)
	}, options);

	if (result) {
	    // NOTE: we intentionally do NOT await completion of the following
	    // add-commit-score-commit operations. they can be executed asynchronously
	    // with the execution of this loop. makes logs a little messier though.
	    My.gitAddCommit(path, 'new result')
		.then(() => {
		    console.log(`Committed: ${path}`);
		    return !_.isEmpty(result) && Result.fileScoreSaveCommit(path);
		}).catch(err => {
		    // log & eat all errors
		    console.log(`getAllResultsLoop commit error`, err, err.stack);
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
    return count;
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
