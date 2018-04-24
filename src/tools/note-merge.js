/*
 * note-merge.js
 */

'use strict';

//

const _                = require('lodash');
const ClueManager      = require('../modules/clue-manager');
const Clues            = require('../modules/clue-types');
const Debug            = require('debug')('note-merge');
const Note             = require('../modules/note');
const NoteMerge        = require('../modules/note-merge');
const GetOpt           = require('node-getopt');
const Path             = require('path');
 
//

const Options = GetOpt.create(_.concat(Clues.Options, [
    ['',  'force-create',    `create note if it doesn't exist`],
    ['',  'no-filter-urls'],
    ['',  'no-filter-sources'],
    ['',  'note=NAME',       'specify note name'],
    ['',  'notebook=NAME',   'specify notebook name'],
    ['',  'production',      'create note in production'],
    ['v', 'verbose',         'more logging']
])).bindHelp(
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

    ClueManager.loadAllClues({ clues: Clues.getByOptions(options) });

    if (options['no-filter-urls'])    options.noFilterUrls    = true;
    if (options['no-filter-sources']) options.noFilterSources = true;
    if (options['force-create'])      options.force_create = true;

    // if production && !default (notebook)
    //   if --notebook specified, use specified notebook name
    //   else get notebook name from base filename
    const noteName = options.note || Path.basename(filename);
    if (!options.default) {
        const nbName = options.notebook || Note.getWorksheetName(noteName);
        const nb = await Note.getNotebook(nbName, options).catch(err => { throw err; });
        if (!nb) {
            usage(`notebook not found, ${nbName}`);
        }
        options.notebookGuid = nb.guid;
        Debug(`notebookGuid: ${options.notebookGuid}`);
    }
    return NoteMerge.mergeFilterFile(filename, noteName, options);
}

//

main().catch(err => {
    console.error(err, err.stack);
    console.log(err, err.stack);
});
