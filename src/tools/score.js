//
// SCORE.JS
//

'use strict';

const _            = require('lodash');
const Dir          = require('node-dir');
const Expect       = require('chai').expect;
const fs           = require('fs');
const Linebyline   = require('linebyline');
const Path         = require('path');
const Promise      = require('bluebird');
const Readlines    = require('n-readlines');
const Score        = require('../modules/score');
const SearchResult = require('../modules/search-result');

const fsReadFile   = Promise.promisify(fs.readFile);
const fsWriteFile  = Promise.promisify(fs.writeFile);

const Opt          = require('node-getopt')
      .create([
	  ['d', 'dir=NAME',            'directory name'],
	  ['',  'force',               'force re-score'],
	  ['m', 'match=EXPR',          'filename match expression' ],
	  ['v', 'verbose',             'extra logging'],
	  ['h', 'help',                'this screen']
      ]).bindHelp().parseSystem();

//

function scoreSearchResultDir (dir, fileMatch, options = {}) {
    Expect(dir).to.be.a('string');
    Expect(fileMatch).to.be.a('string');
    Expect(options).to.be.an('object');

    let path = SearchResult.DIR + dir;
    console.log('dir: ' + path);
    Dir.readFiles(path, {
	match:     new RegExp(fileMatch),
	exclude:   /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err; // TODO: test this
	Promise.resolve(JSON.parse(content))
	    .then(resultList => {
		if (_.isEmpty(resultList)) return undefined;
		return SearchResult.scoreSaveCommit(resultList, filepath, options);
	    })
	    .catch(err => {
		if (err) {
		    console.error(`scoreSearchResultsDir error, ${err}`);
		}
	    }).then(() => {
		return next();
	    });
    }, function(err, files) {
        if (err) throw err; // TODO: test this
    });
}

//

function scoreSaveFile (filepath, options) {
    Expect(filepath).to.be.a('string');
    Expect(options).to.be.an('object');

    // TODO: Path.format()
    return fsReadFile(filepath, 'utf8')
	.then(content => Score.scoreResultList(
	    SearchResult.makeWordlist(filepath),
	    JSON.parse(content),
	    options
	).then(scoreResult => {
	    if (_.isEmpty(scoreResult)) return false;
	    return fsWriteFile(filepath, JSON.stringify(scoreResult));
	}));
}

//

/*
async function scoreSearchResultFiles (dir, inputFilename, options = {}) {
    //Expect(dir).to.be.a('string');
    Expect(inputFilename).to.be.a('string');
    Expect(options).to.be.an('object');
    
    // ignore dir for now, add it to options later
    console.log(`dir: ${options.dir}`);
    let readLines = new Readlines(inputFilename);
    while (true) {
	let nextLine = readLines.next(); 
	if (nextLine  === false) return;
	let wordList = nextLine.toString().trim().split(',');
	let filepath = Path.format({
	    dir:  SearchResult.DIR + options.dir || _.toString(wordList.length),
	    base: SearchResult.makeFilename(wordList)
	});
	console.log(`filepath: ${filepath}`);
	await scoreSaveFile(filepath, options)
	    .then(() => {
		console.log('updated');
	    }).catch(err => {
		console.log(`scoreSaveFile error, ${err}`);
	    });
    }
}
*/

//

async function scoreSearchResultFiles2 (dir, inputFilename, options = {}) {
    //Expect(dir).to.be.a('string');
    Expect(inputFilename).to.be.a('string');
    Expect(options).to.be.an('object');
    
    // ignore dir for now, add it to options later
    console.log(`dir: ${options.dir}`);
    let readLines = new Readlines(inputFilename);
    while (true) {
	let nextLine = readLines.next(); 
	if (nextLine  === false) return;
	let wordList = nextLine.toString().trim().split(',');
	let filepath = Path.format({
	    dir:  SearchResult.DIR + (options.dir || _.toString(wordList.length)),
	    base: SearchResult.makeFilename(wordList)
	});
	console.log(`filepath: ${filepath}`);
	await scoreSaveFile(filepath, options)
	    .then(scoreResult => {
		if (scoreResult === false) {
		    console.log('empty, not updated');
		} else {
		    console.log('updated');
		}
	    }).catch(err => {
		console.log(`scoreSaveFile error, ${err}`);
	    });
    }
}

//
//
//

async function main () {
    Expect(Opt.argv.length, 'only one optional FILE parameter is allowed')
	.to.be.at.most(1);

    let inputFile;
    if (Opt.argv.length === 1) {
	Expect(Opt.options.match, 'option -m EXPR not allowed with FILE').to.be.undefined;
	inputFile = Opt.argv[0];
	console.log(`file: ${inputFile}`);
    }
    if (Opt.options.force === true) {
	console.log('force: ' + Opt.options.force);
    }
    let scoreOptions = {
	force:   Opt.options.force,
	verbose: Opt.options.verbose
    };
    if (inputFile) {
	scoreSearchResultFiles2(Opt.options.dir, inputFile, scoreOptions);
    } else {
	Expect(Opt.options.dir, 'option -d NAME is required').to.exist;
	let fileMatch = SearchResult.getFileMatch(Opt.options.match);
	console.log(`fileMatch: ${fileMatch}`);
	scoreSearchResultDir(Opt.options.dir, fileMatch, scoreOptions);
    }
}

//

main().catch(err => {
    console.log(err.stack);
});
