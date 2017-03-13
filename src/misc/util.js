//
// UTIL.JS
//

'use strict';

//

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Path         = require('path');
const Git          = require('simple-git');
const Ms           = require('ms');
const PrettyMs     = require('pretty-ms');

//

function waitFor (delay) {
    return new Promise(resolve => {
	setTimeout(resolve, delay);
    });
}

//

function between (lo, hi) {
    Expect(lo, 'lo').to.be.a('number');
    Expect(hi, 'hi').to.be.a('number');
    Expect(hi, 'hi < lo').to.be.at.least(lo);
    return lo + Math.random() * (hi - lo);
}

// check if path exists, and whether it is a file.
//
function checkIfFile (path) {
    Expect(path).to.be.a('string');
    return new Promise((resolve, reject) => {
	Fs.stat(path, (err, stats) => {
	    let exists = false;
	    if (err) {
		if (err.code !== 'ENOENT') return reject(err); // some unknown error
	    } else {
		exists = true;
	    }
	    resolve ({
		exists,
		isFile: exists ? stats.isFile() : false
	    });
	});
    });
}

//

function gitAdd (filepath) {
    Expect(filepath).to.be.a('string');
    return new Promise((resolve, reject) => {
	Git(Path.dirname(filepath))
	    .add(Path.basename(filepath), err => err ? reject(err) : resolve());
    });
}

//

function gitRemove (filepath) {
    Expect(filepath).to.be.a('string');
    return new Promise((resolve, reject) => {
	Git(Path.dirname(filepath))
	    .rm(Path.basename(filepath), err => err ? reject(err) : resolve());
    });
}

//

function gitCommitOnce (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return new Promise((resolve, reject) => {
	Git(Path.dirname(filepath))
	    .commit(message, Path.basename(filepath), {}, err => err ? reject(err) : resolve(true));
    });
}

//
// git commit error on mac:
//
// Unable to create '/..path../.git/index.lock': File exists.
//
// If no other git process is currently running, this probably means a
// git process crashed in this repository earlier. Make sure no other git
// pprocess is running and remove the file manually to continue.
//

function isGitLockError (err) {
    return _.isString(err) && _.includes(err, 'index.lock');
}

//

const RETRY_DELAY_LOW     = 1000;
const RETRY_DELAY_HIGH    = 5000;
const DEFAULT_RETRY_COUNT = 5;

async function gitRetryCommit (filepath, message, retryCount = DEFAULT_RETRY_COUNT) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    Expect(retryCount).to.be.a('number');
    let lastError;
    while (true) {
	let result = await gitCommitOnce(filepath, message)
		.catch(err => {
		    console.log(`gitRetryCommit error, (${typeof err}), ${err}`);
		    if (!isGitLockError(err)) {
			// no retries for unknown errors
			retryCount = 0;
		    }
		    lastError = err;
		    return false;
		});
	console.log(`commitResult: ${result}`);
	if (result !== false) return Promise.resolve(result);
	if (retryCount < 1) {
	    console.log('rejecting');
	    return Promise.reject(lastError);
	}
	console.log(`retryCount: ${retryCount}`);
	retryCount -= 1;
	const delay = between(RETRY_DELAY_LOW, RETRY_DELAY_HIGH);
	console.log(`Retrying in ${PrettyMs(delay)}`);
	await waitFor(delay);
    }
}

//

function gitCommit (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return gitRetryCommit(filepath, message);
}

//
    
function gitAddCommit (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return gitAdd(filepath)
	.then(() => gitCommit(filepath, message))
	.catch(err => console.log(`gitAddCommit error, ${err}`));// TODO: bad
}

//

function gitRemoveCommit (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return gitRemove(filepath)
	.then(() => gitCommit(filepath, message))
	.catch(err => console.log(`gitRemoveCommit error, ${err}`)); // TODO: bad
}

//

function gitForceAddCommit (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    let basename = Path.basename(filepath);
    return new Promise((resolve, reject) => {
	return Git(Path.dirname(filepath))
	    .raw([ 'add', '-f', basename ], err => {
		if (err) reject(err);
	    }).commit(message, basename, {}, err => err ? reject(err) : resolve());
    });
}

//

function gitRemoveCommitIfExists (filepath, message = 'removing test file') {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return checkIfFile(filepath)
	.then(result => {
	    if (!result.exists) Promise.reject();
	    // file exists, try to git remove/commit it
	    console.log(`git-removing ${filepath}`);
	    return gitRemoveCommit(filepath, message);
	}).then(() => checkIfFile(filepath))
	.then(result => {
	    Expect(result.exists).to.be.false;
	    console.log(`git-removed ${filepath}`);
	    return undefined;
	    // TODO: add "git reset HEAD -- filename" if we had git remove error or file
	    // still exists; file may be added but not committed.
	}).catch(err => {
	    // log real errors, eat all errors  // TODO: bad
	    if (err) {
		console.log(`gitRemoveCommitIfExists error, ${err}`);
	    };
	    //TODO: if (err) Promise.reject(err);
	    return undefined; 
	});
}

//

module.exports = {
    between,
    checkIfFile,
    gitAdd,
    gitAddCommit,
    gitCommit,
    gitForceAddCommit,
    gitRemove,
    gitRemoveCommit,
    gitRemoveCommitIfExists,
    waitFor
};
