/*
 * create.js
 */

'use strict';

//

//const _           = require('lodash');
//const Debug       = require('debug')('make');
const Expect      = require('should/as-function');
const Fs          = require('fs-extra');
const My          = require('../modules/util');
const Note        = require('../modules/note');
const NoteMake    = require('../modules/note-make');
const Path        = require('path');
const Tmp         = require('tmp');
 
const Options = require('node-getopt')
    .create([
	['', 'notebook=NAME',   'specify notebook name']
	['', 'production',      'create note in production']
    ]).bindHelp(
	"Usage: node create [options] FILE\n\n[[OPTIONS]]\n"
    );

//

function usage () {
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
    if (!filename) {
	usage();
    }
    const [path, fd] = await My.createTmpFile(true).catch(err => { throw err; });

    // todo: if --notebook specified, and production, do getNotebook to get GUID
    // figure out how that fits into the async'ness below
    // if not specified, and --production, get notebook name from filename
    //   if notebook name from filename does exist, require --default


    // TODO: streams = better here
    return NoteMake.make(filename, fd)
	.then(_ => {
	    Fs.closeSync(fd);
	    return Fs.readFile(path);
	}).then(content => {
	    // TODO: DEP0013 ??
	    Expect(content).is.ok();
	    // TODO: check if note exists!
	    return Note.create(Path.basename(filename), content.toString(), options);
	});
}

//

main().catch(err => {
    console.log(err, err.stack);
});
