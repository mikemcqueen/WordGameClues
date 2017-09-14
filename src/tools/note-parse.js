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
    const wordPairList = await NoteParse.parseFile(filename);
    if (_.isEmpty(wordPairList)) {
	console.log('no results');
	return;
    }
    for (const wordPair of wordPairList) {
	console.log(wordPair);
    }
}

//

main().catch(err => {
    console.log(err, err.stack);
});
