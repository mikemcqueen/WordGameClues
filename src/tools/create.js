/*
 * create.js
 */

'use strict';

//

//const _           = require('lodash');
const Debug       = require('debug')('create');
const Expect      = require('should/as-function');
const Fs          = require('fs-extra');
const My          = require('../modules/util');
const Note        = require('../modules/note');
const NoteMake    = require('../modules/note-make');
const Path        = require('path');
const Tmp         = require('tmp');
 
const Options     = require('node-getopt')
    .create([
	['', 'notebook=NAME',   'specify notebook name'],
	['', 'production',      'create note in production']
    ]).bindHelp(
	"Usage: node create [options] FILE\n\n[[OPTIONS]]\n"
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
    //   if --notebook specified, call getNotebook to get GUID
    //   if --notebook not specified, get notebook name from filename
    
    if (options.production && !options.default) {
	const nbName = options.notebook || Note.getNotebookName(Path.basename(filename));
	const nb = await Note.getNotebook(nbName, options);
	if (!nb) {
	    usage(`Can't find notebook ${nbName}`);
	}
	options.notebookGuid = nb.guid;
    }
    const keep = true; // <-- NOTE
    const [path, fd] = await My.createTmpFile(keep).catch(err => { throw err; });

    // TODO: streams = better here
    return NoteMake.make(filename, fd)
	.then(_ => {
	    // a little weird. fseek(0) then read (then close)?
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
