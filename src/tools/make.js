/*
 * make.js
 */

'use strict';

//

const Debug            = require('debug')('make');
const NoteMaker        = require('../modules/note-make');
 
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
    NoteMaker.makeFromFile(filename, { fd: process.stdout.fd, outerDiv: true });
}

//

main().catch(err => {
    console.log(err, err.stack);
});
