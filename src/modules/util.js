//
// util.js
//

'use strict';

//

const _              = require('lodash');
const Debug          = require('debug')('util');
const Expect         = require('should/as-function');
const Fs             = require('fs-extra');
const Git            = require('simple-git');
const Path           = require('path');
const PrettyMs       = require('pretty-ms');
const Retry          = require('retry');
const Tmp            = require('tmp');

//

//

const RETRY_DELAY_LOW     = 1000;
const RETRY_DELAY_HIGH    = 5000;

const DEFAULT_RETRY_COUNT = 5;

const RETRY_EXPO_START_MS      = 1000;
const RETRY_EXPO_COUNT         = 7;  // 64s
const RETRY_LINEAR_DELAY_MS    = 5000;
const RETRY_LINEAR_DELAY_LO_MS = 1000;
const RETRY_LINEAR_DELAY_HI_MS = 5000;
const RETRY_LINEAR_COUNT       = 10; // 50s


const RETRY_TIMEOUTS_LINEAR        = linearTimeouts(RETRY_LINEAR_DELAY_MS,
                                                    RETRY_LINEAR_DELAY_MS,
                                                    RETRY_LINEAR_COUNT);
const RETRY_TIMEOUTS_RANDOM_LINEAR = linearTimeouts(RETRY_LINEAR_DELAY_LO_MS,
                                                    RETRY_LINEAR_DELAY_HI_MS,
                                                    RETRY_LINEAR_COUNT);
const RETRY_TIMEOUTS_EXPO          = expoTimeouts(RETRY_EXPO_START_MS, RETRY_EXPO_COUNT);

//

function linearTimeouts (low, high, count) {
    let timeouts = [];
    // gotta bet a lodash function for this, like fillWith()
    let to = between(low, high);
    while (count > 0) {
        timeouts.push(to);
        to += between(low, high);
        count -= 1;
    }
    return timeouts;
};

//

function expoTimeouts (first, count) {
    let timeouts = [];
    // gotta bet a lodash function for this, like fillWith()
    let to = first;
    while (count > 0) {
        timeouts.push(to);
        to *= 2;
        count -= 1;
    }
    return timeouts;
};

//

function waitFor (delay) {
    return new Promise(resolve => {
        setTimeout(resolve, delay);
    });
}

//

function between (lo, hi) {
    Expect(lo).is.a.Number();
    Expect(hi).is.a.Number();
    Expect(hi).is.above(lo - 1); // at.least(lo);
    return lo + Math.random() * (hi - lo);
}

function fstat (path) {
    return new Promise((resolve, reject) => {
	Expect(path).is.a.String();
        Fs.stat(path, (err, stats) => {
            if (err) {
                if (err.code !== 'ENOENT') return reject(err); // some unknown error
	    }
	    return resolve(err ? undefined : stats);
	});
    });
}

// check if path exists, and whether it is a file.
//
async function checkIfFile (path) {
    Expect(path).is.a.String();
    return new Promise((resolve, reject) => {
        Fs.stat(path, (err, stats) => {
            let exists = false;
            if (err) {
                if (err.code !== 'ENOENT') return reject(err); // some unknown error
            } else {
                exists = true;
            }
            return resolve ({
                exists,
                isFile: exists ? stats.isFile() : false
            });
        });
    });
}

//

function gitRetryAdd (filepath, callback) {
    let op = Retry.operation(
        linearTimeouts(RETRY_LINEAR_DELAY_LO_MS,
                       RETRY_LINEAR_DELAY_HI_MS,
                       RETRY_LINEAR_COUNT));
    op.attempt(num => {
        if (num > 1) {
            console.log(`gitRetryAdd #${num} @ ${PrettyMs(Date.now())}`);
        }
        Git(Path.dirname(filepath))
            .add(Path.basename(filepath), err => {
                if (op.retry(err)) {
                    console.log('gitRetryAdd: error, retrying...');
                    return;
                }
                callback(err ? op.mainError() : null);
            });
    });
}

//

function gitAdd (filepath) {
    Expect(filepath).is.a.String();
    return new Promise((resolve, reject) => {
        gitRetryAdd(filepath, err =>  {
            if (err) {
                console.error('gitAdd failed');
                return reject(err);
            }
            return resolve();
        });
    });
}

//

function gitRemove (filepath) {
    Expect(filepath).is.a.String();
    return new Promise((resolve, reject) => {
        Git(Path.dirname(filepath))
            .rm(Path.basename(filepath), err => err ? reject(err) : resolve());
    });
}

//

function gitCommitOnce (filepath, message) {
    Expect(filepath).is.a.String();
    Expect(message).is.a.String();
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


async function gitRetryCommit (filepath, message, retryCount = DEFAULT_RETRY_COUNT) {
    Expect(filepath).is.a.String();
    Expect(message).is.a.String();
    Expect(retryCount).is.a.Number();
    let lastError;
    while (true) {
        let result = await gitCommitOnce(filepath, message)
                .catch(err => {
                    if (!isGitLockError(err)) {
                        console.error(`gitRetryCommit error, (${typeof err}), ${err}`);
                        // no retries for unknown errors
                        retryCount = 0;
                    }
                    lastError = err;
                    return false;
                });
        console.log(`${Path.basename(filepath)} commitResult: ${result}`);
        // success: resolve
        if (result !== false) return result; // Promise.resolve(result);
        // failure: retry logic
        if (retryCount < 1) {
            console.error(`gitRetryCommit failure, ${filepath}`);
            return Promise.reject(lastError);
        }
        const delay = between(RETRY_DELAY_LOW, RETRY_DELAY_HIGH);
        Debug(`${Path.basename(filepath)} retryCount: ${retryCount}` +
              `, retrying in ${PrettyMs(delay)}`);
        retryCount -= 1;
        await waitFor(delay);
    }
}

//

function gitCommit (filepath, message) {
    Expect(filepath).is.a.String();
    Expect(message).is.a.String();
    return gitRetryCommit(filepath, message);
}

//
    
function gitAddCommit (filepath, message) {
    Expect(filepath).is.a.String();
    Expect(message).is.a.String();
    return gitAdd(filepath)
        .then(_ => gitCommit(filepath, message));
    //  .catch(err => console.log(`gitAddCommit error, ${err}`));// TODO: bad
}

//

function gitRemoveCommit (filepath, message) {
    Expect(filepath).is.a.String();
    Expect(message).is.a.String();
    return gitRemove(filepath)
        .then(_ => gitCommit(filepath, message));
    //  .catch(err =>  console.log(`gitRemoveCommit error, ${err}`)); // TODO: bad
}

//

function gitForceAddCommit (filepath, message) {
    Expect(filepath).is.a.String();
    Expect(message).is.a.String();
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
    Expect(filepath).is.a.String();
    Expect(message).is.a.String();
    return checkIfFile(filepath)
        .then(result => {
            if (!result.exists) return Promise.reject();
            // file exists, try to git remove/commit it
            console.log(`git-removing ${filepath}`);
            return gitRemoveCommit(filepath, message);
        }).then(_ => checkIfFile(filepath))
        .then(result => {
            Expect(result.exists).is.false();
            console.log(`git-removed ${filepath}`);
            return undefined;
            // TODO: add "git reset HEAD -- filename" if we had git remove error or file
            // still exists; file may be added but not committed.
        }).catch(err => {
            // log real errors, eat all errors  // TODO: bad
            if (err) {
                console.error(`gitRemoveCommitIfExists error, ${err}`);
            };
            //TODO: if (err) Promise.reject(err);
            return undefined; 
        });
}

//

function logStream (stream, string) {
    // is this async ? promise? return?
    // answer: has a callback
    stream.write(string + '\n');
    // TODO: inner function 'write()' that uses once('drain', write);
}

// 

function createTmpFile (keep)  {
    return new Promise((resolve, reject) => {
        Tmp.file({ keep }, (err, path, fd) => {
            if (err) reject(err);
            Debug(`File: ${path}, fd: ${fd}`);
            resolve([path, fd]);
        });
    });
}

//

async function write (dest, text) {
    const output = text;
    if (_.isString(dest)) {
        return `${dest}${output}`;
    }
    if (_.isNumber(dest)) {
        return Fs.write(dest, output);
    }
    throw new Error(`bad dest, ${dest}`);
}

//

function writeln (dest, text) {
    return write(dest, `${text}\n`);
}

//

module.exports = {
    between,
    checkIfFile,
    createTmpFile,
    fstat,
    gitAdd,
    gitAddCommit,
    gitCommit,
    gitForceAddCommit,
    gitRemove,
    gitRemoveCommit,
    gitRemoveCommitIfExists,
    logStream,
    waitFor,
    write,
    writeln
};
