/*
 * note-parse.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('test-note-parse');
const NoteParse        = require('../modules/note-parse');
 
//

function usage () {
    console.log('usage: make <filename>');
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
    const resultList = await NoteParse.parseFile(filename, { urls: true });
    if (_.isEmpty(resultList)) {
	console.log('no results');
	return;
    }
    for (const result of resultList) {
	console.log(result);
    }
}

//

main().catch(err => {
    console.log(err, err.stack);
});
