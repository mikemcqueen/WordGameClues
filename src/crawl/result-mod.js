//
// RESULT-MOD
//

const _            = require('lodash');
const expect       = require('chai').expect;
const Path         = require('path');

//

const RESULT_DIR   =    '../../data/results/';
const FILTERED_SUFFIX = '_filtered';
const EXT_JSON =        '.json';

// match *.json, or *match*.json, but not
// *_filtered.json, or *match*_filtered.json
//
function getFileMatch(match = undefined) {
    let prefix = `(?=^((?!${FILTERED_SUFFIX}).)*$)`;
    let suffix = `.*\${EXT_JSON}$`;
    return _.isUndefined(match) ? `${prefix}${suffix}` : `${prefix}.*${match}${suffix}`;
}

// ([ "word", "list" ], "_suffix" ) => word-list[_suffix].json
//
function makeFilename(wordList, suffix) {
    expect(wordList.length).to.be.at.least(2);

    let filename = '';
    wordList.forEach(word => {
	if (_.size(filename) > 0) {
	    filename += '-';
	}
	filename += word;
    });
    if (!_.isUndefined(suffix)) {
	filename += suffix;
    }
    return Path.format({
	name: filename,
	ext:  EXT_JSON
    });
}

//

function makeFilteredFilename(wordList) {
    return makeFilename(wordList, FILTERED_SUFFIX);
}

//

function pathFormat(args) {
    expect(args.dir, 'args.dir').to.be.a('string');
    let root = _.isUndefined(args.root) ? RESULT_DIR : args.root;
    args.dir = root + args.dir;
    // Path.format ignore args.root if args.dir is set
    return Path.format(args);
}

//

module.exports = {
    getFileMatch:         getFileMatch,
    makeFilename:         makeFilename,
    makeFilteredFilename: makeFilteredFilename,
    pathFormat:           pathFormat,
    DIR:                  RESULT_DIR
};
