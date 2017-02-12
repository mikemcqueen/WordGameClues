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
const Opt          = require('node-getopt')
    .create([
	['f', 'force',               'force re-score'],
	['s', 'synthesis',           'use synth clues'],
	['h', 'help',                'this screen']
    ])
    .bindHelp().parseSystem();

const Score        = require('./score-mod');

//

const RESULTS_DIR = '../../data/results/';

//
//
//

function main() {
    /*
    var base = 'meta';
    ClueManager.loadAllClues({
	baseDir:  base,
    });
    */

    if (Opt.options.force) {
	console.log('force: ' + Opt.options.force);
    }

    Dir.readFiles(RESULTS_DIR + 2, {
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
	    console.error('error: ' + err);
	}).then(list => {
	    if (_.isEmpty(list)) {
		return next();
	    }
	    fsWriteFile(filepath, JSON.stringify(list)).then(() => {
		console.log('updated');
		return next();
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
