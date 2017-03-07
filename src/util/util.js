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
	    if (err) {
		if (err.code === 'ENOENT') {
		    resolve({       // does not exist, therefore not a file
			exists: false,
			isFile: false
		    });
		} else {
		    // some other error
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

function gitCommit (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return new Promise((resolve, reject) => {
	Git(Path.dirname(filepath)).commit(
	    message, Path.basename(filepath), {}, err => err ? reject(err) : resolve());
    });
}

//
    
function gitAddCommit (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return gitAdd(filepath)
	.then(() => gitCommit(filepath, message))
	.catch(err => console.log(`gitAddCommit error, ${err}`));
}

//

function gitRemoveCommit (filepath, message) {
    Expect(filepath).to.be.a('string');
    Expect(message).to.be.a('string');
    return gitRemove(filepath)
	.then(() => gitCommit(filepath, message))
	.catch(err => console.log(`gitRemoveCommit error, ${err}`));
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

module.exports = {
    between,
    checkIfFile,
    gitAdd,
    gitAddCommit,
    gitCommit,
    gitForceAddCommit,
    gitRemove,
    gitRemoveCommit,
    waitFor
};
