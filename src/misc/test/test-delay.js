//
// TEST-UTIL.JS
//

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Ms           = require('ms');
const My           = require('../util');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');

//const fsReadFile   = Promise.promisify(Fs.readFile);
const fsWriteFile  = Promise.promisify(Fs.writeFile);

//

const TESTFILES_DIR = `${__dirname}/test-files/`;

//

function randomObject() {
    return { a : _.random(), b: _.random() };
}

//

function writeAddCommit(filepath, obj) {
    console.log(`writeAddCommit ${filepath}`);
    console.log('committing file');
    return My.gitCommit(filepath, 'adding test file');
}

//

describe ('git tests:', function () {

    ////////////////////////////////////////////////////////////////////////////////
    //
    // test gitCommit retrying under the covers
    //
    it.skip ('should attempt multiple simultaneous commits to same file', function (done) {
	let filepath = TESTFILES_DIR + 'test-retry-commit';
	let p = [];
	let n = 15;

	this.timeout(10000 + (n * 500)); // 10s per commit
	this.slow(5000 * (n * 500)); // 5s per commit is slow

	fsWriteFile(filepath, '[]')
	    .then(() => {
		for (let index = 0; index < n; index += 1) {
		    p.push(My.gitCommit(filepath, `simultaneous commit ${index + 1} of ${n}`));
		}
		console.log(`${n} asynch gitCommits are running...`);
		return Promise.all(p);
	    }).then(() => {
		console.log(`done`);
		done();
	    }).catch(err => {
		console.log(`error, ${err}`);
		done(err);
	    });
	console.log('eof');
    });

    ////////////////////////////////////////////////////////////////////////////////
    //
    // test gitRemoveCommit, gitAddCommit
    //
    it ('should remove/commit, then add/commit a file to git', function (done) {
	this.timeout(30000);
	this.slow(10000);

	let filepath = TESTFILES_DIR + 'test-add-commit';
	My.gitRemoveCommitIfExists(filepath, 'removing test file')
	    .then(() => {
		console.log('writing new file');
		return fsWriteFile(filepath, '[]');
	    }).then(() => {
		console.log('calling addCommit');
		return My.gitAddCommit(filepath, 'adding test file');
	    }).then(() => {
		console.log('done');
		done();
	    }).catch(err => {
		console.log(`error, ${err}`);
		done(err);
	    });
    });
});

//
	  
describe ('delay tests:', function () {

    it.skip ('between: verify 20x 2-3 minute values', function () {
	let lo = Ms('2m');
	let hi = Ms('3m');
	for (let count = 0; count < 20; ++count) {
	    let delay = My.between(lo, hi);
	    Expect(delay).to.be.at.least(lo).and.at.most(hi);
	    console.log(PrettyMs(delay));
	}
    });

});
