//
// RESULT-MOD
//

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Path         = require('path');
const Promise      = require('bluebird');
const Score        = require('./score-mod.js');

const fsWriteFile  = Promise.promisify(Fs.writeFile);

//

const RESULT_DIR      = '../../data/results/';
const FILTERED_SUFFIX = '_filtered';
const EXT_JSON        = '.json';

// '../../results/file-path.json' => [ 'file', 'path' ]
//
function makeWordlist (filepath) {
    return _.split(Path.basename(filepath, EXT_JSON), '-');
}

// match *.json, or *match*.json, but not
// *_filtered.json, or *match*_filtered.json
//
function getFileMatch (match = undefined) {
    let prefix = `(?=^((?!${FILTERED_SUFFIX}).)*$)`;
    let suffix = `.*\\${EXT_JSON}$`;
    return _.isUndefined(match) ? `${prefix}${suffix}` : `${prefix}.*${match}${suffix}`;
}

// ([ 'word', 'list' ], '_suffix' ) => 'word-list_suffix.json'
//
function makeFilename (wordList, suffix = undefined) {
    Expect(wordList.length).to.be.at.least(2);
    let filename = wordList.join('-');
    if (!_.isUndefined(suffix)) {
	filename += suffix;
    }
    return Path.format({
	name: filename,
	ext:  EXT_JSON
    });
}

//

function makeFilteredFilename (wordList) {
    return makeFilename(wordList, FILTERED_SUFFIX);
}

//

function pathFormat (args) {
    Expect(args, 'args').to.exist;
    Expect(args.dir, 'args.dir').to.be.a('string');
    let root = _.isUndefined(args.root) ? RESULT_DIR : args.root;
    args.dir = root + args.dir;
    // Path.format ignore args.root if args.dir is set
    return Path.format(args);
}

//

function wordCountFilter (result, wordCount, options) {
    let passTitle = false;
    let passArticle = false;
    if (result.score) {
	if (options.filterTitle) {
	    passTitle = (result.score.wordsInTitle >= wordCount);
	}
	if (options.filterArticle) {
	    passArticle = (result.score.wordsInSummary >= wordCount ||
			   result.score.wordsInArticle >= wordCount);
	}
    }
    return (passTitle || passArticle) ? result : undefined;
}

//

function scoreSaveCommit (resultList, filepath, options) {
    Expect(resultList).to.be.an('array');
    Expect(filepath).to.be.a('string');

    console.log('filename: ' + filepath);
    return Score.scoreResultList(
	makeWordlist(filepath),
	resultList,
	options
    ).then(list => {
	if (!_.isEmpty(list)) {
	    return fsWriteFile(filepath, JSON.stringify(list))
		.then(() => console.log('updated'));
	}
	return undefined;
    });
    // TODO: .catch()
}

//

module.exports = {
    getFileMatch:         getFileMatch,
    makeFilename:         makeFilename,
    makeFilteredFilename: makeFilteredFilename,
    pathFormat:           pathFormat,
    scoreSaveCommit:      scoreSaveCommit,
    wordCountFilter:      wordCountFilter,
    DIR:                  RESULT_DIR
};
