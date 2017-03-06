//
// RESULT-MOD
//

const _            = require('lodash');
const Expect       = require('chai').expect;
const Path         = require('path');

//

const RESULT_DIR      = '../../data/results/';
const FILTERED_SUFFIX = '_filtered';
const EXT_JSON        = '.json';

// match *.json, or *match*.json, but not
// *_filtered.json, or *match*_filtered.json
//
function getFileMatch (match = undefined) {
    let prefix = `(?=^((?!${FILTERED_SUFFIX}).)*$)`;
    let suffix = `.*\\${EXT_JSON}$`;
    return _.isUndefined(match) ? `${prefix}${suffix}` : `${prefix}.*${match}${suffix}`;
}

// ([ "word", "list" ], "_suffix" ) => word-list[_suffix].json
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

module.exports = {
    getFileMatch:         getFileMatch,
    makeFilename:         makeFilename,
    makeFilteredFilename: makeFilteredFilename,
    pathFormat:           pathFormat,
    wordCountFilter:      wordCountFilter,
    DIR:                  RESULT_DIR
};
