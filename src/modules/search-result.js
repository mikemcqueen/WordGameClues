//
// RESULT-MOD.JS
//

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Google       = require('google');
const Ms           = require('ms');
const My           = require('../misc/util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');
const Score        = require('./score-mod');

const fsReadFile   = Promise.promisify(Fs.readFile);
const fsWriteFile  = Promise.promisify(Fs.writeFile);

//

const RESULT_DIR      =  Path.normalize(`${Path.dirname(module.filename)}/../../data/results/`);
//const RESULT_DIR    = '../../data/results/';
const FILTERED_SUFFIX = '_filtered';
const EXT_JSON        = '.json';

const SRC_PREFIX      = '@';
const MAYBE_PREFIX    = ':';
const KNOWN_PREFIX    = '#';

// '../../file-path.json' => [ 'file', 'path' ]
//
function makeWordlist (filepath) {
    Expect(filepath).to.be.a('string');
    return _.split(Path.basename(filepath, EXT_JSON), '-');
}

// match '*.json', or '*match*.json', but not
// '*_filtered.json', or '*match*_filtered.json'
//
function getFileMatch (match = undefined) {
    let prefix = `(?=^((?!${FILTERED_SUFFIX}).)*$)`;
    let suffix = `.*\\${EXT_JSON}$`;
    return _.isUndefined(match) ? `${prefix}${suffix}` : `${prefix}.*${match}${suffix}`;
}

// ([ 'word', 'list' ], '_suffix' ) => 'word-list_suffix.json'
//
function makeFilename (wordList, suffix = undefined) {
    Expect(wordList.length, `wordList: ${wordList}`).to.be.at.least(2);
    let filename = wordList.join('-');
    if (suffix) { // necessary?
	filename += suffix;
    }
    return Path.format({
	name: filename,
	ext:  EXT_JSON
    });
}

//

function makeFilteredFilename (wordList) {
    Expect(wordList).to.be.an('array').that.is.not.empty;
    return makeFilename(wordList, FILTERED_SUFFIX);
}

//

function pathFormat (args) {
    Expect(args, 'args').to.exist;
    Expect(args.dir, 'args.dir').to.be.a('string');
    let root = _.isUndefined(args.root) ? RESULT_DIR : args.root;
    args.dir = root + args.dir; 
    // Path.format ignores args.root if args.dir is set
    return Path.format(args);
}

//

function get (text, pages, cb) {
    Expect(text, 'text').to.be.a('string');
    Expect(pages, 'pages').to.be.a('number').that.is.at.least(1);
    Expect(cb, 'callback').to.be.a('function');
    let resultList = [];
    let count = 0;
    Google(text, function (err, result) {
	if (err) return cb(err);
	resultList.push(...result.links.map(link => {
	    return {
		title:   link.title,
		url:     link.href,
		summary: link.description
	    };
	}));
	count += 1;
	// done?
	if (count === pages || !result.next) {
	    return cb(null, resultList);
	}
	Expect(count).to.be.below(pages);
	Expect(result.next).to.exist;
	let msDelay = My.between(Ms('30s'), Ms('60s'));
	console.log(`Delaying ${PrettyMs(msDelay)} for next page of results...`);
	// TODO async function, use await My.delay(msDelay)
	setTimeout(result.next, msDelay);
	return undefined;
    });
}

//

function scoreSaveCommit (resultList, filepath, options = {}, wordList = undefined) {
    Expect(resultList).to.be.an('array').that.is.not.empty;
    Expect(filepath).to.be.a('string');
    Expect(options).to.be.an('object');
    if (!_.isUndefined(wordList)) {
	Expect(wordList).to.be.an('array').with.length.of.at.least(2);
    }
    console.log(`scoreSaveCommit, ${filepath} (${_.size(resultList)})`);
    return Score.scoreResultList(
	_.isUndefined(wordList) ? makeWordlist(filepath) : wordList,
	resultList,
	options
    ).then(list => {
	if (_.isEmpty(list)) {
	    // if list is empty, all the results in this file were already scored
	    console.log('empty list, already scored?');
	    return Promise.reject(); // skip save/commit
	}
	return fsWriteFile(filepath, JSON.stringify(list))
	    .then(() => console.log(' updated'));
    }).then(() => {
	return My.gitCommit(filepath, 'updated score')
	    .then(() => console.log(' committed'));
    });
}

//

function fileScoreSaveCommit (filepath, options = {}, wordList = undefined) {
    Expect(filepath).to.be.a('string');
    Expect(options).to.be.an('object');
    if (!_.isUndefined(wordList)) {
	Expect(wordList).to.be.an('array').with.length.of.at.least(2);
    }
    return fsReadFile(filepath, 'utf8')
	.then(data => scoreSaveCommit(JSON.parse(data), filepath, options, wordList));
}

//

module.exports = {
    fileScoreSaveCommit,
    get,
    getFileMatch,
    makeFilename,
    makeFilteredFilename,
    makeWordlist,
    pathFormat,
    scoreSaveCommit,

    EXT_JSON,
    SRC_PREFIX,
    MAYBE_PREFIX,
    KNOWN_PREFIX,
    DIR: RESULT_DIR
};
