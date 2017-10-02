/*
 * note-parse.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('test-note-parse');
const Filter           = require('../modules/filter');
const NoteParse        = require('../modules/note-parse');
 
//

function usage () {
    console.log('usage: note-parse <filename>');
    process.exit(-1);
}

//

async function main () {
    const filename = process.argv[2];
    if (!filename) {
	usage();
    }
    Debug(`filename: ${filename}`);
    // TODO: streams = better here
    let fd = process.stdout.fd;
    return NoteParse.parseFile(filename)
	.then(filterList => {
	    if (_.isEmpty(filterList)) {
		console.log('no results');
		return;
	    }
	    Filter.dumpList(filterList, { fd: process.stdout.fd });
	});
}

//

main().catch(err => {
    console.log(err, err.stack);
});
