//
// CRAWL.JS
//

'use strict';

var _            = require('lodash');
var Q            = require('q');
var fs           = require('fs');
var parse        = require('csv-parse');

var Delay        = require('../util/delay');
var googleResult = require('./googleResult');

var RESULTS_DIR = '../../data/results/';


var Opt = require('node-getopt')
    .create([
	['h' , 'help'               ,         'this screen']
    ])
    .bindHelp().parseSystem();

function main() {
    var fileLoadDefer;
    var filename;

    if (Opt.argv.length < 1) {
	console.log('Usage: node crawl pairs-file');
	console.log('');
	console.log(Opt);
	return 1;
    }

    filename = Opt.argv[0];
    console.log('filename: ' + filename);

    fileLoadDefer = Q.defer();
    fs.readFile(filename, 'utf8', function(err, data) {
	if (err) throw err;
	fileLoadDefer.resolve(data);
    });
    
    fileLoadDefer.promise.then(data => {
	parse(data, null, (err, output) => {
	    if (err) throw err;
	    getResults(output);
	});
    });
}

//

function getResults(wordListArray) {
    var wordList = wordListArray.pop();
    var filename;
    var path;

    if (wordList === undefined) {
	return;
    }
    filename = makeFilename(wordList);
    path = RESULTS_DIR + _.size(wordList) + '/' + filename;
    
    if (1) {
	console.log('list: ' + wordList);
	console.log('file: ' + filename);
    }
    
    checkIfFile(path, (err, isFile) => {
	if (err) throw err;
	if (isFile) {
	    console.log('Skip: file exists, ' + filename);
	    getResults(wordListArray);
	}
	else {
	    getOneResult(wordList, (err, data) => {
		if (err) throw err;
		if (_.size(data) > 0) {
		    fs.writeFile(path, JSON.stringify(data), (err) => {
			if (err) throw err;
		    });
		}
		setTimeout(() => {
		    getResults(wordListArray);
		}, Delay.between(8, 12, Delay.Minutes));
	    });
	}
    });
}

//

function getOneResult(wordList, cb) {
    var term = makeSearchTerm(wordList, { wikipedia: true });

    console.log('term: ' + term);
    googleResult.get(term, function(err, res) {
	cb(err, res);
    });
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
    var filename = '';

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
    var term = '';

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
