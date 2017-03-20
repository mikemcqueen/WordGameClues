//
// SEARCH.JS
//

'use strict';

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Ms           = require('ms');
const Promise      = require('bluebird');
const Search       = require('./search-mod');
const Opt          = require('node-getopt')
      .create([
	  ['d', 'dir=NAME',            'directory name'],
	  ['h', 'help',                'this screen' ]
      ])
      .bindHelp().parseSystem();

//

const csvParse     = Promise.promisify(require('csv-parse'));
const fsReadFile   = Promise.promisify(Fs.readFile);

//

const DEFAULT_PAGE_COUNT = 2;
const DEFAULT_DELAY_LOW  = 8;
const DEFAULT_DELAY_HIGH = 12;

//
//
//

function main() {
    if (Opt.argv.length < 1) {
	console.log('Usage: node search pairs-file [delay-minutes-low delay-minutes-high]');
	console.log(' ex: node search file 4    ; delay 4 minutes between searches');
	console.log(' ex: node search file 4 5  ; delay 4 to 5 minutes between searches');
	console.log(` defaults, low: ${DEFAULT_DELAY_LOW}, high: ${DEFAULT_DELAY_HIGH}`);
	console.log(Opt);
	return 1;
    }

    // arg0
    let filename = Opt.argv[0];
    console.log(`filename: ${filename}`);
    // arg1
    let delayLow = DEFAULT_DELAY_LOW;
    let delayHigh = DEFAULT_DELAY_HIGH;
    if (Opt.argv.length > 1) {
	delayLow = _.toNumber(Opt.argv[1]);
	Expect(delayLow, 'delayLow').to.be.at.least(1);
	delayHigh = delayLow;
    }
    // arg2
    if (Opt.argv.length > 2) {
	delayHigh = _.toNumber(Opt.argv[2]);
	Expect(delayHigh, 'delayHigh').to.be.at.least(delayLow);
    }
    console.log(`Delaying ${delayLow} to ${delayHigh} minutes between searches`);
    
    let delay = {
	low:   Ms(`${delayLow}m`),
	high:  Ms(`${delayHigh}m`)
    };
    fsReadFile(filename, 'utf8')
	.then(csvContent => csvParse(csvContent, null))
	.then(wordListArray => Search.getAllResults({
	    // NOTE: use default dir
	    wordListArray: wordListArray,
	    pages:         DEFAULT_PAGE_COUNT,
	    delay:         delay
	})).catch(err => {
	    console.log(`error caught in main, ${err}`);
	    console.log(err.stack);
	});
}

//

try {
    main();
}
catch(err) {
    console.log(`error caught in try/catch, ${err}`);
    console.log(err.stack);
}
finally {
}
