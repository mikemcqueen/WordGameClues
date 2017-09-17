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
	['',  'note=NAME',       'specify note name'],
	['',  'notebook=NAME',   'specify notebook name'],
	['',  'production',      'create note in production'],
	['v', 'verbose',         'more logging']
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

    // if production && !default (notebook)
    //   if --notebook specified, use specified notebook name
    //   else get notebook name from base filename
    const noteName = options.note || Path.basename(filename);
    if (options.production && !options.default) {
	const nbName = options.notebook || Note.getNotebookName(noteName);
	const nb = await Note.getNotebook(nbName, options).catch(err => { throw err; });
	if (!nb) {
	    usage(`notebook not found, ${nbName}`);
	}
	options.notebookGuid = nb.guid;
    }
    return NoteMerge.mergeFilterFile(filename, noteName, options);
}

//

main().catch(err => {
    console.log(err, err.stack);
});
