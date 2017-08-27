//
// SCORE.JS
//

'use strict';

const _            = require('lodash');
const Dir          = require('node-dir');
const Expect       = require('should/as-number');
const Fs           = require('fs');
const Linebyline   = require('linebyline');
const Path         = require('path');
const Promise      = require('bluebird');
const Readlines    = require('n-readlines');
const Score        = require('../modules/score');
const SearchResult = require('../modules/search-result');

const FsReadFile   = Promise.promisify(Fs.readFile);
const FsWriteFile  = Promise.promisify(Fs.writeFile);

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
    Expect(dir).is.a('string');
    Expect(fileMatch).is.a('string');
    Expect(options).is.an('object');

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
    Expect(filepath).is.a('string');
    Expect(options).is.an('object');

    // TODO: Path.format()
    return FsReadFile(filepath, 'utf8')
	.then(content => Score.scoreResultList(
	    SearchResult.makeWordlist(filepath),
	    JSON.parse(content),
	    options
	).then(scoreResult => {
	    if (_.isEmpty(scoreResult)) return false;
	    return FsWriteFile(filepath, JSON.stringify(scoreResult));
	}));
}

//

/*
async function scoreSearchResultFiles (dir, inputFilename, options = {}) {
    //Expect(dir).is.a('string');
    Expect(inputFilename).is.a('string');
    Expect(options).is.an('object');
    
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
    //Expect(dir).is.a('string');
    Expect(inputFilename).is.a('string');
    Expect(options).is.an('object');
    
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
    if (Opt.argv.length > 1) {
	console.log('only one optional FILE parameter is allowed');
	process.exit(-1);
    }

    let inputFile;
    if (Opt.argv.length === 1) {
	if (Opt.options.match) {
	    console.log('option -m EXPR not allowed with FILE');
	    process.exit(-1);
	}
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
	if (!Opt.options.dir) {
	    console.log('option -d NAME is required');
	    process.exit(-1);
	}
	let fileMatch = SearchResult.getFileMatch(Opt.options.match);
	console.log(`fileMatch: ${fileMatch}`);
	scoreSearchResultDir(Opt.options.dir, fileMatch, scoreOptions);
    }
}

//

main().catch(err => {
    console.log(err.stack);
});
