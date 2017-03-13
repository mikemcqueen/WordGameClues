//
// SCORE.JS
//

'use strict';

const _            = require('lodash');
const Promise      = require('bluebird');
const fs           = require('fs');
const Readlines    = require('n-readlines');
const Dir          = require('node-dir');
const Path         = require('path');
const Expect       = require('chai').expect;
const Score        = require('./score-mod');
const Result       = require('./result-mod');
const Opt          = require('node-getopt')
      .create([
	  ['d', 'dir=NAME',            'directory name'],
	  ['f', 'force',               'force re-score'],
	  ['m', 'match=EXPR',          'filename match expression' ],
	  ['s', 'synth',               'use synth clues'],
	  ['v', 'verbose',             'extra logging'],
	  ['h', 'help',                'this screen']
      ]).bindHelp().parseSystem();

const fsReadFile   = Promise.promisify(fs.readFile);
const fsWriteFile  = Promise.promisify(fs.writeFile);

//

function scoreSearchResultDir (dir, fileMatch, options = {}) {
    Expect(dir).to.be.a('string');
    Expect(fileMatch).to.be.a('string');
    Expect(options).to.be.an('object');

    let path = Result.DIR + dir;
    console.log('dir: ' + path);
    Dir.readFiles(path, {
	match:   new RegExp(fileMatch),
	exclude: /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err; // TODO: test this
	Result.scoreSaveCommit(JSON.parse(content), filepath, options)
	    .catch(err => {
		if (err) {
		    console.error(`scoreSaveCommit error, ${err}`);
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
    fsReadFile(filepath, 'utf8')
	.then(content => Score.scoreResultList(
	    Result.makeWordList(filepath),
	    JSON.parse(content),
	    options
	).then(scoreResult => {
	    if (_.isEmpty(scoreResult)) {
		console.log('empty score result');
		return undefined;
	    }
	    return fsWriteFile(filepath, JSON.stringify(scoreResult));
	}));
}

//

async function scoreSearchResultFiles (dir, inputFilename, options = {}) {
    Expect(dir).to.be.a('string');
    Expect(inputFilename).to.be.a('string');
    Expect(options).to.be.an('object');

    dir = Result.DIR + dir;
    console.log('dir: ' + dir);
    let readLines = new Readlines(inputFilename);
    while (true) {
	let filename = readLines.next();
	if (filename === false) return;
	filename = filename.toString().trim();
	let filepath = Path.format({ dir, base: filename });
	console.log('filepath: ' + filepath);
	await scoreSaveFile(filepath, options)
	    .then(() => console.log('updated'))
	    .catch(err => {
		console.log(`scoreSaveFile error, ${err.message}`);
		if (err) throw err;
	    });
    }
}

//
//
//

function main () {
    Expect(Opt.argv.length, 'only one optional FILE parameter is allowed')
	.to.be.at.most(1);
    Expect(Opt.options.dir, 'option -d NAME is required').to.exist;

    let inputFile;
    if (Opt.argv.length === 1) {
	inputFile = Opt.argv[0];
	console.log(`file: ${inputFile}`);
	Expect(Opt.options.match, 'option -m EXPR not allowed with FILE').to.be.undefined;
    }
    if (Opt.options.force === true) {
	console.log('force: ' + Opt.options.force);
    }
    let scoreOptions = {
	force:   Opt.options.force,
	verbose: Opt.options.verbose
    };
    if (!_.isUndefined(inputFile)) {
	scoreSearchResultFiles(Opt.options.dir, inputFile, scoreOptions);
    }
    else {
	let fileMatch = Result.getFileMatch(Opt.options.match);
	console.log(`fileMatch: ${fileMatch}`);
	scoreSearchResultDir(Opt.options.dir, fileMatch, scoreOptions);
    }
}

//

try {
    main();
}
catch(e) {
    console.log(e.stack);
}
finally {
}
