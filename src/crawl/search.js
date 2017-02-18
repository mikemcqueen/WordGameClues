//
// SEARCH.JS
//

'use strict';

var _            = require('lodash');
var Promise      = require('bluebird');
var expect       = require('chai').expect;
var prettyMs     = require('pretty-ms');
var expect       = require('chai').expect;

var fs           = Promise.promisifyAll(require('fs'));
var fsReadFile   = Promise.promisify(fs.readFile);
var csvParse     = Promise.promisify(require('csv-parse'));

var Delay        = require('../util/delay');
var googleResult = require('./googleResult');

var RESULTS_DIR = '../../data/results/';

var Opt = require('node-getopt')
    .create([
	['d', 'dir=NAME',            'directory name'],
	['h', 'help',                'this screen' ]
    ])
    .bindHelp().parseSystem();

const DEFAULT_DELAY_LOW = 8;
const DEFAULT_DELAY_HIGH = 12;

//
//
//

function main() {
    if (Opt.argv.length < 1) {
	console.log('Usage: node search pairs-file [delay-minutes-low delay-minutes-high]');
	console.log(' ex: node search file 4    ; delay 4 minutes between searches');
	console.log(' ex: node search file 4 5  ; delay 4 to 5 minutes between searches');
	console.log(' defaults, low: ' + DEFAULT_DELAY_LOW + ', high: ' + DEFAULT_DELAY_HIGH);
	console.log(Opt);
	return 1;
    }

    // arg0
    let filename = Opt.argv[0];
    console.log('filename: ' + filename);

    // arg1
    let delayLow = DEFAULT_DELAY_LOW;
    let delayHigh = DEFAULT_DELAY_HIGH;
    if (Opt.argv.length > 1) {
	delayLow = _.toNumber(Opt.argv[1])
	expect(delayLow, 'delayLow').to.be.at.least(1);
	delayHigh = delayLow;
    }

    // arg2
    if (Opt.argv.length > 2) {
	delayHigh = _.toNumber(Opt.argv[2]);
	expect(delayHigh, 'delayHigh').to.be.at.least(delayLow);
    }

    console.log(`Delaying ${delayLow} to ${delayHigh} minutes between searches`);

    fsReadFile(filename, 'utf8')
	.then(csvData => csvParse(csvData, null))
	.then(wordListArray => {
	    getAllResults(wordListArray, {
		low:  delayLow,
		high: delayHigh
	    });
	}).catch(err => {
	    console.log('error, ' + err);
	});
    /*
    fsReadFile(filename, 'utf8').then(csvData => {
	csvParse(csvData, null).then(wordListArray => {
	    getAllResults(wordListArray, {
		low:  delayLow,
		high: delayHigh
	    });
	}).catch(err => {
	    console.log('csvParse error, ' + err);
	});
    }).catch(err => {
	console.log('fs.readFile error, ' + err);
    });
    */
}

//

function getAllResults(wordListArray, delay) {
    expect(wordListArray).to.be.an('array');
    expect(delay).to.have.property('low')
    expect(delay).to.have.property('high')

    let wordList = wordListArray.pop();
    if (_.isUndefined(wordList)) return;

    let filename = makeFilename(wordList);
    let path = RESULTS_DIR + _.size(wordList) + '/' + filename;
    
    console.log('list: ' + wordList);
    console.log('file: ' + filename);
    
    checkIfFile(path, (err, isFile) => {
	if (err) throw err;
	if (isFile) {
	    console.log(`Skip: file exists, ${filename}`);
	    return getAllResults(wordListArray, delay);
	}
	getOneResult(wordList, (err, data) => {
	    if (err) throw err;
	    if (_.size(data) > 0) {
		fs.writeFile(path, JSON.stringify(data), (err) => {
		    if (err) throw err;
		    console.log(`Saved: ${filename}`);
		});
	    }
	    let msDelay = Delay.between(delay.low, delay.high, Delay.Minutes);
	    console.log('Delaying ' + prettyMs(msDelay) + ' for next search...');
	    setTimeout(() => getAllResults(wordListArray, delay), msDelay);
	});
    });
}

//

function getOneResult(wordList, cb) {
    let term = makeSearchTerm(wordList, { wikipedia: true });

    console.log('term: ' + term);
    googleResult.get(term, cb);
}

//

function checkIfFile(file, cb) {
    fs.stat(file, function fsStat(err, stats) {
	if (err) {
	    if (err.code === 'ENOENT') {
		return cb(null, false);
	    } else {
		return cb(err);
	    }
	}
	return cb(null, stats.isFile());
    });
}

//

function makeFilename(wordList) {
    let filename = '';
    
    wordList.forEach(word => {
	if (_.size(filename) > 0) {
	    filename += '-';
	}
	filename += word;
    });
    return filename + '.json';
}

//

function makeSearchTerm(wordList, options) {
    let term = '';

    wordList.forEach(word => {
	if (_.size(term) > 0) {
	    term += ' ';
	}
	term += word;
    });
    
    if (options && options.wikipedia) {
	term += ' site:en.wikipedia.org';
    }
    return term;
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
