//
// SCORE.JS
//

'use strict';

const _            = require('lodash');
const Promise      = require('bluebird');
const fs           = require('fs');
const fsReadFile   = Promise.promisify(fs.readFile);
const fsWriteFile  = Promise.promisify(fs.writeFile);
const Readlines    = require('n-readlines');
const Dir          = require('node-dir');
const Path         = require('path');
const expect       = require('chai').expect;
const Opt          = require('node-getopt')
      .create([
	  ['d', 'dir=NAME',            'directory name'],
	  ['f', 'force',               'force re-score'],
	  ['m', 'match=EXPR',          'filename match expression' ],
	  ['s', 'synth',               'use synth clues'],
	  ['h', 'help',                'this screen']
      ]).bindHelp().parseSystem();

const Score        = require('./score-mod');

//

const RESULT_DIR   = '../../data/results/';

//

function scoreSearchResultDir(dir, fileMatch, options) {
    expect(dir).to.be.a('string');
    expect(fileMatch).to.be.a('string');

    let path = RESULT_DIR + dir;
    console.log('dir: ' + path);
    Dir.readFiles(path, {
	match:   new RegExp(fileMatch),
	exclude: /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err;
	console.log('filename: ' + filepath);
	Score.scoreResultList(
	    _.split(Path.basename(filepath, '.json'), '-'),
	    JSON.parse(content),
	    options
	).catch(err => {
	    console.error('scoreResultList error: ' + err);
	}).then(list => {
	    if (_.isEmpty(list)) {
		return next();
	    }
	    fsWriteFile(filepath, JSON.stringify(list)).then(() => {
		console.log('updated');
		return next();
	    }).catch(err => {
		console.error('fsWriteFile error: ' + err);
	    });
	});
    }, function(err, files) {
        if (err) throw err;
    });
}

//

function scoreNextFile(dir, readLines, options) {
    let filename = readLines.next();
    if (filename === false) return;
    filename = filename.toString();
    if (filename.trim() === '') return scoreNextFile(readLines, options);

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
	    if (_.isEmpty(outputList)) {
		console.log('empty output list, not updating');
		return true;
	    }
	    return fsWriteFile(filepath, JSON.stringify(outputList));
	}).then((empty) => {
	    if (empty !== true) {
		console.log('updated');
	    }
	}).catch(err => {
	    throw err;
	}).then(() => scoreNextFile(dir, readLines, options));
}

//

function scoreSearchResultFiles(dir, inputFilename, options) {
    expect(dir).to.be.a('string');
    expect(inputFilename).to.be.a('string');

    dir = RESULT_DIR + dir;
    console.log('dir: ' + dir);

    let readLines = new Readlines(inputFilename);
    scoreNextFile(dir, readLines, options);
}

//
//
//

function main() {
    expect(Opt.argv.length, 'only one optional FILE parameter is allowed')
	.to.be.at.most(1);
    expect(Opt.options.dir, 'option -d NAME is required').to.exist;

    let inputFile;
    if (Opt.argv.length === 1) {
	inputFile = Opt.argv[0];
	console.log(`file: ${inputFile}`);
	expect(Opt.options.match, 'option -m EXPR not allowed with FILE').to.be.undefined;
    }
    if (Opt.options.force === true) {
	console.log('force: ' + Opt.options.force);
    }
    let fileMatch = '\.json$';
    if (!_.isUndefined(Opt.options.match)) {
	fileMatch = `.*${Opt.options.match}.*` + fileMatch;
    }
    let scoreOptions = {
	force: Opt.options.force
    };
    if (!_.isUndefined(inputFile)) {
	scoreSearchResultFiles(Opt.options.dir, inputFile, scoreOptions);
    }
    else {
	scoreSearchResultDir(Opt.options.dir, fileMatch, scoreOptions);
    }
    /*
    else {
	expect('not supported').to.equal('specify -d NAME');
	for (let dir = 2; dir < 8; ++dir) {
	    scoreSearchResultFiles(dir, fileMatch, scoreOptions);
	}
    }
    */
}

//

try {
    main();
}
catch(e) {
    console.error(e.stack);
}
finally {
}
