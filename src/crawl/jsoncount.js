//
// JSONCOUNT.JS
//

'use strict';

var _            = require('lodash');
//var Promise      = require('bluebird');
var FS           = require('fs');
//var fsReadFile   = Promise.promisify(FS.readFile);

var Opt          = require('node-getopt')
    .create([
	['h', 'help',                'this screen']
    ])
    .bindHelp().parseSystem();

//
//
//

function main() {
    if (Opt.argv.length < 1) {
	console.log('Usage: node count json-file');
	return 1;
    }

    //buffer = 	fsReadFileSync(Opt.argv[0], 'utf8');
    console.log(_.size(JSON.parse(FS.readFileSync(Opt.argv[0], 'utf8'))));
}

//

try {
    main();
}
catch(e) {
    console.error(e.stack);
}
finally {
}
