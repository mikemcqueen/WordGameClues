//
// search.js
//

'use strict';

const _            = require('lodash');
const Expect       = require('should/as-function');
const Fs           = require('fs-extra');
const Ms           = require('ms');
const Promise      = require('bluebird');
const Search       = require('../modules/search');

const CsvParse     = Promise.promisify(require('csv-parse'));

//

const DEFAULT_PAGE_COUNT = 2;
const DEFAULT_DELAY_LOW  = 8;
const DEFAULT_DELAY_HIGH = 12;

//

const CmdLineOptions = require('node-getopt')
      .create([
          ['f', 'force',               'force search even if results file exists'],
	  ['n', 'nowiki',              'don\'t restrict to en.wikipedia.org'],
	  ['p', 'pages=COUNT',         'retrieve COUNT pages of result'],
	  ['q', 'quoted',              'search using quoted words'],
          ['h', 'help',                'this screen' ]
      ])
      .bindHelp(
          'Usage: node search pairs-file [delay-minutes-low delay-minutes-high]' +
              ' ex: node search file 4    ; delay 4 minutes between searches' +
              ' ex: node search file 4 5  ; delay 4 to 5 minutes between searches' +
          ` defaults, low: ${DEFAULT_DELAY_LOW}, high: ${DEFAULT_DELAY_HIGH}`
      );

//
//
//

async function main() {
    let Opt = CmdLineOptions.parseSystem();
    if (Opt.argv.length < 1) {
        CmdLineOptions.showHelp();
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
        Expect(delayLow).is.above(-1);
        delayHigh = delayLow;
    }
    // arg2
    if (Opt.argv.length > 2) {
        delayHigh = _.toNumber(Opt.argv[2]);
        Expect(delayHigh >= delayLow).is.true();
    }
    console.log(`Delaying ${delayLow} to ${delayHigh} minutes between searches`);
    
    let delay = {
        low:   Ms(`${delayLow}m`),
        high:  Ms(`${delayHigh}m`)
    };
    let pages = Opt.options.pages ? _.toNumber(Opt.options.pages) : DEFAULT_PAGE_COUNT;
    console.log(`pages: ${pages}`);
    return Fs.readFile(filename)
        .then(csvContent => CsvParse(csvContent, { relax_column_count: true } ))
        .then(wordListArray => Search.getAllResults({
	    // NOTE: use default dir
            wordListArray, pages, delay,
        }, Opt.options));
}

//

main().catch(err =>  {
    console.log(`error, ${err}`);
    console.log(err.stack);
});
