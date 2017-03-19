//
// FILTER.JS
//

'use strict';

const _            = require('lodash');
const ClueManager  = require('../clue_manager.js');
const Clues        = require('../clue-types.js');
const Dir          = require('node-dir');
const Fs           = require('fs');
const Duration     = require('duration');
const Expect       = require('chai').expect;
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');
const Result       = require('./result-mod');
const Score        = require('./score-mod');

const Opt          = require('node-getopt').create([
    ['a', 'article',             'filter results based on article word count'],
    ['d', 'dir=NAME',            'directory name'],
    //['f', 'final',             'use final clues'],
//    ['k', 'known',               'filter known results'],  // easy!, call ClueManager.filter()
    ['m', 'match=EXPR',          'filename match expression' ],
    ['n', 'count',               'show result/url counts only'],
//    ['r', 'rejects',             'show only results that fail all filters'],
    //['r', 'harmony',             'use harmony clues'],
    ['t', 'title',               'filter results based on title word count (default)'],
    ['x', 'xfactor=VALUE',       'show borked results, with 1) missing URL/title/summary 2) unscored URLs' +
                                   ' 3) article > summary'], 
    ['y', 'synthesis',           'use synthesis clues'],

    ['v', 'verbose',             'show logging'],
    ['h', 'help',                'this screen']
]).bindHelp().parseSystem();

//

const csvParse     = Promise.promisify(require('csv-parse'));
const fsReadFile   = Promise.promisify(Fs.readFile);

//

const XFACTOR_MISSING  = 1;
const XFACTOR_UNSCORED = 2;
const XFACTOR_BADSCORE = 3;

//

function getUrlCount (resultList) {
    Expect(resultList).to.be.an('array');
    // TODO _.reduce()
    let urlCount = 0;
    for (const result of resultList) {
	urlCount += _.size(result.urlList);
    }
    return urlCount;
}

//

function isFilteredUrl (url, filteredUrls) {
    Expect(url).to.be.a('string');
    if (_.isUndefined(filteredUrls)) return false; 
    Expect(filteredUrls).to.be.an('object');
    let reject = _.isArray(filteredUrls.rejectUrls) && filteredUrls.rejectUrls.includes(url);
    let known = _.isArray(filteredUrls.knownUrls) && filteredUrls.knownUrls.includes(url);
    return known || reject;
}

// check for broken results if xfactor option specified

function isXFactor (result, options) {
    Expect(result).to.be.an('object');
    Expect(options).to.be.an('object');
    if (options.xfactor === XFACTOR_MISSING) { 
	return _.isEmpty(result.url) || _.isEmpty(result.title) || _.isEmpty(result.summary);
    }
    if (options.xfactor === XFACTOR_UNSCORED) {
	return _.isEmpty(result.score);
    }
    if (options.xfactor === XFACTOR_BADSCORE && _.isObject(result.score)) {
	return result.score.wordsInArticle < result.score.wordsInSummary;
    }
    return false;
}

//

function filterSearchResultList (resultList, wordList, filteredUrls, options) {
    Expect(resultList).to.be.an('array');
    Expect(wordList).to.be.an('array').that.is.not.empty;
    // filteredUrls can be undefined or array
    Expect(options).to.be.an('object');
    let urlList = [];
    let loggedXFactor = false;
    // make any clue words with a space into multiple words.
    let wordCount = _.chain(wordList).map(word => word.split(' ')).flatten().size().value();
    return Promise.map(resultList, result => {
	if (options.verbose) {
	    console.log(`result: ${_.entries(result)}`);
	}
	if (options.xfactor && isXFactor(result, options)) {
	    if (!loggedXFactor) {
		console.log(options.filepath);
		loggedXFactor = true;
	    }
	    return undefined;
	}
	if (isFilteredUrl(result.url, filteredUrls) || _.isUndefined(result.score)) {
	    if (options.verbose) {
		console.log(`filtered or unscored url, ${result.url}`);
	    }
	    return undefined;
	}
	return Score.wordCountFilter(result.score, wordCount, options) ? result : undefined;
    }).each(filterResult => {
	if (!_.isUndefined(filterResult)) {
	    if (options.verbose) {
		console.log(`url: ${filterResult.url}`);
	    }
	    urlList.push(filterResult.url);
	}
	return undefined;
    }).then(() => {
	if (options.verbose) {
	    console.log(`urlList.size = ${_.size(urlList)}`);
	}
	return {
	    src:     wordList.toString(),
	    urlList: urlList,
	    known:   ClueManager.getKnownClues(wordList)
	};
    });
    // TODO: .catch()
}

//

function loadFilteredUrls (dir, wordList, options) {
    Expect(dir).to.be.a('string');
    Expect(wordList).to.be.an('array');
    Expect(options).to.be.an('object');
    let filteredFilename = Result.makeFilteredFilename(wordList);
    if (options.verbose) {
	console.log(`filtered filename: ${filteredFilename}`);
    }
    return fsReadFile(Path.format({ dir, base: filteredFilename }), 'utf8')
	.then(content => {
	    if (options.verbose) {
		console.log(`resolving filtered urls, ${wordList}`);
	    }
	    return JSON.parse(content);
	}).catch(err => {
	    if (err && err.code !== 'ENOENT') throw err;
	    if (options.verbose) {
		console.log(`no filtered urls, ${wordList}, ${err}`);
	    }
	    return undefined;
	});
}

//

function hasRemaining (wordListArray, remaining) {
    Expect(wordListArray).is.an('array');
    Expect(remaining).is.an('array').that.is.not.empty;
    for (const wordList of wordListArray) {
	if (wordList.length !== remaining.length) continue;
	if (remaining.some(word => _.includes(wordList, word))) {
	    return true;
	}
    }
    return false;
}

//

function isInNameMap (nameMap, wordList) {
    Expect(nameMap).to.be.an('object');
    Expect(wordList).to.be.an('array').with.length.of.at.least(1);
    let word = wordList[0];
    if (!_.has(nameMap, word)) return false;
    return hasRemaining(nameMap[word], wordList.slice(1, wordList.length));
}

// for each search result filename that matches fileMatch in dir 
//   filter out rejected word combinations
//   load the file, build filtered URL list
//   load the _filtered.json file, filter out known/reject URLs
//   if any URLs remain, add result to filteredList.
//
// NOTE that for convenience, this function currently loads all
// files that match fileMatch, even if the file's word-combo is
// a rejected combo. 
// 
function filterSearchResultDir (dir, fileMatch, options) {
    Expect(dir).to.be.a('string');
    Expect(fileMatch).to.be.an('string');
    Expect(options).to.be.an('object');
    return new Promise((resolve, reject) => {
	let filteredList = [];
	let rejectList = [];
	Dir.readFiles(dir , {
	    match:     new RegExp(fileMatch),
	    exclude:   /^\./,
	    recursive: false
	}, function(err, content, filepath, next) {
	    if (err) throw err; // TODO: test
	    if (options.verbose) {
		console.log(`filename: ${filepath}`);
	    }
	    let wordList = Result.makeWordlist(filepath);
	    // filter out rejected word combos
	    if (ClueManager.isRejectSource(wordList)) return next();
	    if (!isInNameMap(options.nameMap, wordList)) return next();
	    // temp
	    if (!_.isUndefined(options.nameMap)) {
		console.log(`filename: ${filepath}`);
		return next();
	    }
  	    loadFilteredUrls(dir, wordList, options)
		.then(filteredUrls => {
		    options.filepath = filepath;
		    return filterSearchResultList(JSON.parse(content), wordList, filteredUrls, options);
		}).then(filterResult => {
		    // TODO: I question this logic at the moment
		    if (_.isEmpty(filterResult.urlList)) {
			rejectList.push(filterResult);
		    } else {
			filteredList.push(filterResult);
		    }
		    return undefined;
		}).catch(err => {
		    // report & eat all errors
		    console.log(`filterSearchResultFiles, path: ${filepath}, error; ${err}`);
		});//.then(() => next()); // process files synchronously
	    return next(); // process files asynchronously
	}, function(err, files) {
            if (err) throw err;  // TODO: test
	    resolve({
		filtered: filteredList,
		rejects:  rejectList
	    });
	});
    });
}

//

function displayFilterResults (resultList) {
    Expect(resultList).to.be.an('array');
    for (const result of resultList) {
	if (_.isEmpty(result.urlList)) continue;
	if (ClueManager.isRejectSource(result.src)) continue;

	console.log(`${Result.SRC_PREFIX}${result.src}`);
	for (const url of result.urlList) {
	    console.log(url);
	}
	const nameList = ClueManager.getKnownClues(result.src);
	if (!_.isEmpty(nameList)) {
	    console.log('${Result.KNOWN_PREFIX}known:');
	    for (const name of nameList) {
		console.log(`${Result.KNOWN_PREFIX}${name}`);
	    }
	}
	console.log();
    }
}

// TODO: move to Util.js

function loadCsv (filename) {
    return fsReadFile(filename, 'utf8')
	.then(csvContent => csvParse(csvContent, null));
}

//

function buildNameMap (wordListArray) {
    let map = {};
    for (const wordList of wordListArray) {
	for (const word of wordList) {
	    let remaining = _.difference(wordList, [word]);
	    Expect(remaining.length, 'remaining').is.at.least(1);
	    if (!_.has(map, word)) {
		map[word] = [];
	    }
	    if (!hasRemaining(map[word], remaining)) {
		map[word].push(remaining);
	    }
	}
    }
    return map;
}

//
//
//
async function main () {
    Expect(Opt.argv.length, 'only one non-switch FILE argument allowed').is.at.most(1);
    Expect(Opt.options.dir, 'option -d NAME is required').to.exist;

    ClueManager.loadAllClues({
	baseDir: Opt.options.synth ? Clues.SYNTH.name : Clues.META.name
    });

    // default to title filter if no filter specified
    if (!Opt.options.article && !Opt.options.title) {
	Opt.options.title = true;
    }
    let nameMap;
    if (Opt.argv.length > 0) {
	let wordListArray = await loadCsv(Opt.argv[0]);
	nameMap = buildNameMap(wordListArray);
    }
    let dir = Result.DIR + Opt.options.dir;
    let filterOptions = {
	filterArticle: Opt.options.article,
	filterTitle:   Opt.options.title,
	filterRejects: Opt.options.rejects,
	verbose:       Opt.options.verbose,
	xfactor:       _.toNumber(Opt.options.xfactor),
	nameMap
    };
    let start = new Date();
    let result = await filterSearchResultDir(dir, Result.getFileMatch(Opt.options.match), filterOptions);
    let d = new Duration(start, new Date()).milliseconds;
    if (Opt.options.count) {
	console.log(`Results: ${_.size(result.filtered)}` +
		    `, Urls: ${getUrlCount(result.filtered)}` +
		    `, Rejects: ${_.size(result.rejects)}` +
		    `, duration, ${PrettyMs(d)}`);
    }
    else {
	displayFilterResults(Opt.options.rejects ? result.rejects : result.filtered);
    }
}

//

main()
.catch(err => {
    console.log(err.stack);
});

