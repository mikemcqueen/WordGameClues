//
// JSONCOUNT.JS
//

'use strict';

const Fs           = require('fs');
const Opt          = require('node-getopt')
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
    console.log(JSON.parse(Fs.readFileSync(Opt.argv[0], 'utf8')).length);
}

//

try {
    main();
}
catch(e) {
    console.error(e, e.stack);
}
