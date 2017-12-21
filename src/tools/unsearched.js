//
// unsearched.js
//

'use strict';

const _            = require('lodash');
const Debug        = require('debug')('unsearched');
const Expect       = require('should/as-function');
const Fs           = require('fs-extra');
const Getopt       = require('node-getopt');
const My           = require('../modules/util');
const Path         = require('path');
const Promise      = require('bluebird');
const Readlines    = require('n-readlines');
const Search       = require('../modules/search');
const SearchResult = require('../modules/search-result');

const CsvParse     = Promise.promisify(require('csv-parse'));

const Options = Getopt.create([
    ['v', 'verbose',             'extra logging'],
    ['h', 'help',                'this screen']
]).bindHelp();

//

async function main () {
    const opt = Options.parseSystem();
    const options = opt.options;
    if (opt.argv.length !== 1) {
        console.log('a FILE parameter is required');
        process.exit(-1);
    }
    const filename = opt.argv[0];

    const wordListArray = await Fs.readFile(filename)
              .then(csvContent => CsvParse(csvContent, { relax_column_count: true }))
              .then(wordListArray => wordListArray)
              .catch(err => { throw err; });
    
    for (const [index, wordList] of wordListArray.entries()) {
        let filename = SearchResult.makeFilename(wordList);
        let path = SearchResult.pathFormat({
            //root: args.root,
            dir:  _.toString(wordList.length),
            base: filename
        }, options);
        Debug(path);
        Debug(`list: ${wordList}`);
        Debug(`file: ${filename}`);
        let result = await My.checkIfFile(path).catch(err => { throw err; });
        Debug(`path: ${path}, exists: ${result.exists}`);
        if (result.exists) continue;
        console.log(wordList.join(','));
    }
}

//

main().catch(err => {
    console.error(err, err.stack);
    process.exit(-1);
});
