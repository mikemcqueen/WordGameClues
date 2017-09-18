//
// update.js
//

'use strict';

const _            = require('lodash');
const Update       = require('../modules/update');

const Options      = require('node-getopt')
    .create(_.concat(Clues.Options, [
//	['d', 'dir=NAME',            'directory name'],
	['',  'save',                'save clues'],
	['v', 'verbose',             'show logging'],
	['h', 'help',                'this screen']
    ])).bindHelp(
	"Usage: node update [options] <wordListFile>\n\n[[OPTIONS]]\n"
    );

//

function usage (msg) {
    console.log(msg + '\n');
    Options.showHelp();
    process.exit(-1);
}

//

async function main() {
    const opt = Options.parseSystem();
    const options = opt.options;

    if (opt.argv.length !== 1) {
	usage('exactly one FILE argument is required');
    }
    let filename = opt.argv[0];
    console.log(`filename: ${filename}`);
    if (options.verbose) {
	console.log('verbose: true');
    }

    return Update.updateFromFile(filename, options)
	.then(result => {
	    console.log(`updated knownClues(${result.count.knownClues})` +
			`, maybeClues(${result.count.maybeClues})` +
			`, rejectClues(${result.count.rejectClues})` +
			`, knownUrls(${result.count.knownUrls})` +
			`, maybeUrls(${result.count.maybeUrls})` +
			`, rejectUrls(${result.count.rejectUrls})`);
	});
}

//

main().catch(err => {
    console.error(err, err.stack);
});

