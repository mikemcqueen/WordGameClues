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
const Opt          = require('node-getopt')
    .create([
	['d', 'dir=NAME',                 'directory name'],
	['f', 'force',               'force re-score'],
	//['m', 'match=REGEX',         'match file regex'],
	['s', 'synth',               'use synth clues'],
	['h', 'help',                'this screen']
    ])
    .bindHelp().parseSystem();

const Score        = require('./score-mod');

//

// NOTE: TODO: This shoudl be in data/meta/results (or synth, depending on flag)

const RESULT_DIR = '../../data/results/';
const DATA_DIR = '../../data/';
const RESULTS_DIR = 'results/';

//
//
//

function main() {
    if (!_.isUndefined(Opt.options.force)) {
	console.log('force: ' + Opt.options.force);
    }

    let base = _.isUndefined(Opt.options.synth) ? 'meta' : 'synth';

    if (!_.isUndefined(Opt.options.dir)) {
	scoreDir(base, Opt.options.dir);
    }
    else {
	expect('not supported').to.equal('specify -d NAME');
	for (let dir = 2; dir < 8; ++dir) {
	    scoreDir(base, dir);	    
	}
    }
}

function scoreDir(base, dir) {
    // DATA_DIR + base + RESULTS_DIR + dir
    let path = RESULT_DIR + dir;
    console.log('dir: ' + path);
    Dir.readFiles(path, {
	match:   /\.json$/,
	exclude: /^\./,
	recursive: false
    }, function(err, content, filepath, next) {
	if (err) throw err;
	console.log('filename: ' + filepath);
	Score.scoreResultList(
	    _.split(Path.basename(filepath, '.json'), '-'),
	    JSON.parse(content),
	    { force: Opt.options.force }
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

try {
    main();
}
catch(e) {
    console.error(e.stack);
}
finally {
}
