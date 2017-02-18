//
// SCORE.JS
//

'use strict';

const _            = require('lodash');
const Promise      = require('bluebird');
const FS           = require('fs');
const fsWriteFile  = Promise.promisify(FS.writeFile);
const Dir          = require('node-dir');
const Path         = require('path');
const expect       = require('chai').expect;
const Opt          = require('node-getopt').create([
    ['d', 'dir=NAME',            'directory name'],
    ['f', 'force',               'force re-score'],
    ['m', 'match=EXPR',          'filename match expression' ],
    ['s', 'synth',               'use synth clues'],
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

const Score        = require('./score-mod');

//

const RESULT_DIR = '../../data/results/';
const DATA_DIR = '../../data/';

function scoreSearchResultFiles(dir, fileMatch, options) {
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
//
//

function main() {
    if (!_.isUndefined(Opt.options.force)) {
	console.log('force: ' + Opt.options.force);
    }

    expect(Opt.options.dir, 'option -d NAME is required').to.exist;

    let scoreOptions = {
	force: Opt.options.force
    };

    let fileMatch = '\.json$';
    if (!_.isUndefined(Opt.options.match)) {
	fileMatch = `.*${Opt.options.match}.*` + fileMatch;
    }

    if (!_.isUndefined(Opt.options.dir)) {
	scoreSearchResultFiles(Opt.options.dir, fileMatch, scoreOptions);
    }
    else {
	expect('not supported').to.equal('specify -d NAME');
	for (let dir = 2; dir < 8; ++dir) {
	    scoreSearchResultFiles(dir, fileMatch, scoreOptions);
	}
    }
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
