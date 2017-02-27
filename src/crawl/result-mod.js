//
// RESULT-MOD
//

const _            = require('lodash');
const expect       = require('chai').expect;

//

const RESULT_DIR   =    '../../data/results/';
const FILTERED_SUFFIX = '_filtered';
const EXT_JSON =        'json';

//

function getFileMatch(match = undefined) {
    let prefix = `(?=^((?!${FILTERED_SUFFIX}).)*$)`
    let suffix = `.*\.${EXT_JSON}$`;
    return _.isUndefined(match) ? `${prefix}${suffix}` : `${prefix}.*${match}${suffix}`;
}

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
    return `${filename}.${EXT_JSON}`;
}

//

function makeFilteredFilename(wordList) {
    return makeFilename(wordList, FILTERED_SUFFIX);
}

//

module.exports = {
    getFileMatch:         getFileMatch,
    makeFilename:         makeFilename,
    makeFilteredFilename: makeFilteredFilename,
    DIR:          RESULT_DIR
};
