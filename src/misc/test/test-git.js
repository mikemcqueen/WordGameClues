//
// TEST-GIT.JS
//

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Ms           = require('ms');
const My           = require('../util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');
const Test         = require('./test');

//const fsReadFile   = Promise.promisify(Fs.readFile);
const fsWriteFile  = Promise.promisify(Fs.writeFile);

//

function writeAll (filepathList) {
    console.log('++writeAll');
    let p = [];
    for (const filepath of filepathList) {
	p.push(fsWriteFile(filepath, '[]'));
    }
    return Promise.all(p);
}

//

async function removeAll (filepathList) {	
    console.log('++removeAll');
    for (const filepath of filepathList) {
	await My.gitRemoveCommitIfExists(filepath, 'removing test file');
    }
}

//

describe ('git tests:', function () {

    ////////////////////////////////////////////////////////////////////////////////
    //
    // test gitCommit retrying under the covers
    //
    it.skip ('should attempt multiple simultaneous commits to same file', function (done) {
	let filepath = Path.format({
	    dir:  Test.DIR,
	    base: 'test-retry-commit'
	});

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
    // test gitAddCommit retrying under the covers
    //
    it ('should attempt multiple simultaneous AddCommits to differents files', function (done) {
	let filepath = Path.format({
	    dir:  Test.DIR,
	    base: 'test-retry-commit'
	});

	let filecount = 9;

	this.timeout(10000 * filecount); // 10s per commit
	this.slow(5000 * filecount); // 5s per commit is slow

	// TODO: don't need to create N files, just 1 file is fine.
	// don't need to removeAll/writeAll, just do it the same as below

	let filepathList = [];
	let time = Date.now();
	for (let index = 0; index < filecount; index += 1) {
	    filepathList.push(Path.format({
		dir:  Test.DIR,
		base: 'test-add-commit.' + new Date(time + 1000)
	    }));
	}

	removeAll(filepathList)
	    .then(_ => {
		return writeAll(filepathList);
	    }).then(() => {
		let p = [];
		for (const [index, filepath] of filepathList.entries()) {
		    console.log(`${index}. addCommit(${filepath})`);
		    p.push(My.gitAddCommit(filepath, `simultaneous addCommit ${index + 1} of ${filepathList.length}`));
		}
		console.log(`${filepathList.length} asynch gitCommits are running...`);
		return Promise.all(p);
	    }).then(_ => {
		removeAll(filepathList);
	    }).then(_ => {
		console.log(`success`);
		done();
	    }).catch(err => {
		console.log(`error, ${err}`);
		done(err);
	    });
    });

    ////////////////////////////////////////////////////////////////////////////////
    //
    // test gitRemoveCommit, gitAddCommit
    //
    // steps:
    //   gitRemoveCommit file
    //   create file
    //   gitAddCommit file
    //
    it.skip ('should remove/commit, then add/commit a file to git', function (done) {
	this.timeout(30000);
	this.slow(10000);

	let filepath = Path.format({
	    dir:  Test.DIR,
	    base: 'test-add-commit'
	});
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

