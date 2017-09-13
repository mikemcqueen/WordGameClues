/*
 * create.js
 */

'use strict';

//

//const _                = require('lodash');
//const Debug            = require('debug')('make');
const Expect      = require('should/as-function');
const Fs          = require('fs');
const My          = require('../modules/util');
const Note        = require('../modules/note');
const NoteMake    = require('../modules/note-make');
const Path        = require('path');
const Promise     = require('bluebird');
const Tmp         = require('tmp');
 
const Options = require('node-getopt')
    .create([
	['', 'production',    'create note in production']
    ]).bindHelp(
	"Usage: node filter <options> [wordListFile]\n\n[[OPTIONS]]\n"
    );

//

const FsReadFile = Promise.promisify(Fs.readFile);

//

function usage () {
    Options.showHelp();
    process.exit(-1);
}

//

async function main () {
    const opt = Options.parseSystem();
    const options = opt.options;

    const filename = opt.argv[0];
    if (!filename) {
	usage();
    }
    const [path, fd] = await My.createTmpFile(true).catch(err => { throw err; });

    // TODO: streams = better here
    return NoteMake.make(filename, fd)
	.then(_ => {
	    Fs.closeSync(fd);
	    return FsReadFile(path);
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
