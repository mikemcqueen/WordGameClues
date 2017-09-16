/*
 * note-merge.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('note-merge');
const Note             = require('../modules/note');
const NoteMerge        = require('../modules/note-merge');
const Path             = require('path');
 
const Options     = require('node-getopt')
    .create([
	['', 'note=NAME',       'specify note name'],
	['', 'notebook=NAME',   'specify notebook name'],
	['', 'production',      'create note in production']
    ]).bindHelp(
	"Usage: node note-merge [options] FILE\n\n[[OPTIONS]]\n"
    );

//

function usage (msg) {
    console.log(msg + '\n');
    Options.showHelp();
    process.exit(-1);
}

//

async function main () {
    const opt = Options.parseSystem();
    const options = opt.options;

    if (opt.argv.length !== 1) {
	usage('exactly one FILE argument required');
    }
    const filename = opt.argv[0];
    Debug(`filename: ${filename}`);

    const noteName = options.note || Path.basename(filename);

    // if production && !default (notebook)
    //   if --notebook specified, call getNotebook to get GUID
    //   if --notebook not specified, get notebook name from filename
    
    if (options.production && !options.default) {
	const nbName = options.notebook || Note.getNotebookName(noteName);
	const nb = await Note.getNotebook(nbName, options);
	if (!nb) {
	    usage(`Can't find notebook ${nbName}`);
	}
	options.notebookGuid = nb.guid;
    }

    // TODO: streams = better here
    //let fd = process.stdout.fd;
    return NoteMerge.mergeFilterFile(filename, noteName, options)
	.then(resultList => {
	    if (_.isEmpty(resultList)) {
		console.log('no results');
		return;
	    }
	    // rather than dumping results, validate that the total sources + urls
	    // added is equal to result total 
	    for (const result of resultList) {
		console.log(result);
	    }
	});
}

//

main().catch(err => {
    console.log(err, err.stack);
});
