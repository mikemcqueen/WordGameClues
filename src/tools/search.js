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

const CsvParse     = Promise.promisify(require('csv-parse'));
const FsReadFile   = Promise.promisify(Fs.readFile);

//

const DEFAULT_PAGE_COUNT = 2;
const DEFAULT_DELAY_LOW  = 8;
const DEFAULT_DELAY_HIGH = 12;

//

const Opt          = require('node-getopt')
      .create([
/*	  ['d', 'dir=NAME',            'directory name'],*/
	  ['',  'force',               'force search even if results file exists'],
	  ['h', 'help',                'this screen' ]
      ])
      .bindHelp(
	  'Usage: node search pairs-file [delay-minutes-low delay-minutes-high]' +
	      ' ex: node search file 4    ; delay 4 minutes between searches' +
	      ' ex: node search file 4 5  ; delay 4 to 5 minutes between searches' +
	  ` defaults, low: ${DEFAULT_DELAY_LOW}, high: ${DEFAULT_DELAY_HIGH}`
      ).parseSystem();

//
//
//

async function main() {
    if (Opt.argv.length < 1) {
	Opt.showHelp();
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
	Expect(delayLow, 'delayLow').to.be.at.least(0);
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
    return FsReadFile(filename, 'utf8')
	.then(csvContent => CsvParse(csvContent, { relax_column_count: true } ))
    /* testing CsvParse option.relax_column_count
	.then(wordListArray => {
	    wordListArray.forEach(wordList => {
		console.log(wordList.join(','));
	    });
	    return Promise.reject();
	})
     */
	.then(wordListArray => Search.getAllResults({
	    // NOTE: use default dir
	    wordListArray: wordListArray,
	    pages:         DEFAULT_PAGE_COUNT,
	    delay:         delay
	}, { force: Opt.options.force }));
        /*
	.catch(err => {
	    console.log(`error caught in main, ${err}`);
	    if (err) console.log(err.stack);
	});
	*/
}

//

main().catch(err =>  {
    console.log(`error, ${err}`);
    console.log(err.stack);
});
