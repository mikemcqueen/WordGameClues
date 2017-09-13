/*
 * make.js
 */

'use strict';

//

//const _                = require('lodash');
const Debug            = require('debug')('make');
//const Expect           = require('should/as-function');
const NoteMake         = require('../modules/note-make');
 
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
    let stream = process.stdout.fd;
    NoteMake.make(filename, stream);
}

//

main().catch(err => {
    console.log(err, err.stack);
});
