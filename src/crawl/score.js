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
	  ['h', 'help',                'this screen']
      ]).bindHelp().parseSystem();

const fsReadFile   = Promise.promisify(fs.readFile);
const fsWriteFile  = Promise.promisify(fs.writeFile);

//

function scoreSearchResultDir(dir, fileMatch, options) {
    Expect(dir).to.be.a('string');
    Expect(fileMatch).to.be.a('string');

    let path = Result.DIR + dir;
    console.log('dir: ' + path);
    Dir.readFiles(path, {
	match:   new RegExp(fileMatch),
	exclude: /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err; // TODO: why throw? test this
	console.log('filename: ' + filepath);
	Score.scoreResultList(
	    _.split(Path.basename(filepath, '.json'), '-'),
	    JSON.parse(content),
	    options
	).then(list => {
	    return _.isEmpty(list) ? true : fsWriteFile(filepath, JSON.stringify(list));
	}).then(isEmpty => {
	    if (!isEmpty) {
		console.log('updated');
	    }
	}).catch(err => {
	    console.error(`scoreSearchResultDir error, ${err.message}`);
	}).then(() => {
	    return next();
	});
    }, function(err, files) {
        if (err) throw err; // TODO: test this
    });
}

//

function scoreNextFile(dir, readLines, options) {
    let filename = readLines.next();
    if (filename === false) return;
    filename = filename.toString().trim();

    // TODO: Path.makePath() or something similar
    let filepath = dir + '/' + filename;
    console.log('filepath: ' + filepath);
    fsReadFile(filepath, 'utf8')
	.then(content => JSON.parse(content))
	.then(inputList => {
	    return Score.scoreResultList(
		_.split(Path.basename(filepath, '.json'), '-'),
		inputList,
		options);
	}).then(outputList => {
	    return _.isEmpty(outputList) ? true : fsWriteFile(filepath, JSON.stringify(outputList));
	}).then(isEmpty  => {
	    console.log(isEmpty ? 'empty output list, not updating' : 'updated');
	}).catch(err => {
	    // report & eat all errors
	    console.log(`scoreNextFile error, ${err.message}`);
	}).then(() => {
	    return setTimeout(scoreNextFile(dir, readLines, options), 0);
	});
}

//

function scoreSearchResultFiles(dir, inputFilename, options) {
    Expect(dir).to.be.a('string');
    Expect(inputFilename).to.be.a('string');

    dir = Result.DIR + dir;
    console.log('dir: ' + dir);
    let readLines = new Readlines(inputFilename);
    scoreNextFile(dir, readLines, options);
}

//
//
//

function main() {
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
	force: Opt.options.force
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
