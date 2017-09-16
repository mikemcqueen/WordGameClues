/*
 * note-merge.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('note-merge');
const NoteMerge        = require('../modules/note-merge');
 
//

function usage () {
    console.log('usage: note-merge <filename>');
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
    //let fd = process.stdout.fd;
    const resultList = await NoteMerge.merge(filename);
    if (_.isEmpty(resultList)) {
	console.log('no results');
	return;
    }
    // rather than dumping results, validate that the total sources + urls
    // added is equal to result total 
    for (const result of resultList) {
	console.log(result);
    }
}

//

main().catch(err => {
    console.log(err, err.stack);
});
