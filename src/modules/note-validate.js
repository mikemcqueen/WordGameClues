//
// validate.js
//

'use strict';

const ClueManager  = require('../dist/modules/clue-manager');

const _            = require('lodash');
const Clues        = require('./clue-types');
const Dir          = require('node-dir');
const Duration     = require('duration');
const Expect       = require('should/as-function');
const Filter       = require('./filter');
const Fs           = require('fs-extra');
const Log          = require('./log')('note-validate');
const Markdown     = require('./markdown');
const My           = require('./util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Readlines    = require('n-readlines');
const SearchResult = require('./search-result');

//

function initResult () {
    return {
        count : {
            knownUrls      : 0,
            rejectUrls     : 0,
            maybeUrls      : 0,
            knownClues     : 0,
            rejectClues    : 0,
            maybeClues     : 0,
            knownCountSet  : new Set(),
            rejectCountSet : new Set(),
            maybeCountSet  : new Set()
        },
        timing : {
            getCountList : 0
        }
    };
}

//

async function _validateFile(filename, options) {
    Expect(filename).is.a.String();
    Log.info(`file: ${Path.basename(filename)}`);

    if (!ClueManager.loaded) {
        Log.info('_validateFile: calling ClueManager.loadAllClues()');
        ClueManager.loadAllClues({ clues: Clues.getByOptions(options) });
    }
    // get trailing word of filename
    const lastDot = _.lastIndexOf(filename, '.');
    if (lastDot === -1) throw new Error(`filename has no dot!?, ${filename}`);
    const word = filename.slice(lastDot + 1, filename.length);

    // for each clue source in file
    const filterList = Filter.parseFile(filename, options);
    for (const clueObj of filterList) {
        // remove prefix from clueObj.source
        const [source, prefix]  = Markdown.getPrefix(clueObj.source, Markdown.Prefix.source);
        Expect(prefix).is.ok();
        const countListResult = ClueManager.getCountListArrays(source, options);
        if (!countListResult || !countListResult.valid) {
            const message = !countListResult ? 'doesnt exist??' : countListResult.invalid ? 'invalid' : 'rejected';
            console.log(`${source} ${message}`);
        }
        // Not sure if i should do this, or just a straight grep-style match.
        const regex = new RegExp(`(^${word}|[ ,]${word})`);
        const matchResult = source.match(regex);
        if (!matchResult) {
            console.log(`${source}: no sources match ${word}`);
        }
    }
}

//

async function validateFile(filename, options) {
    const start = new Date();
    return _validateFile(filename, options)
        .then(result => {
            if (options.perf) {
                const duration = new Duration(start, new Date()).milliseconds;
                Log.info(`validateFile duration(${PrettyMs(duration)})` +
                         `, getCountList(${PrettyMs(result.timing.getCountList)})`);
            }
        });
}
    
//

function aggregate (result, allResults) {
    allResults.timing.getCountList += result.timing.getCountList;

    allResults.count.knownClues += result.count.knownClues;
    allResults.count.rejectClues += result.count.rejectClues;
    allResults.count.maybeClues += result.count.maybeClues;
    allResults.count.knownCountSet = new Set([...result.count.knownCountSet, ...allResults.count.knownCountSet]);
    allResults.count.maybeCountSet = new Set([...result.count.maybeCountSet, ...allResults.count.maybeCountSet]);
    allResults.count.rejectCountSet = new Set([...result.count.rejectCountSet, ...allResults.count.rejectCountSet]);
    return allResults;
}

//

async function validatePathList(pathList, options) {
    Expect(pathList).is.an.Array();
//    if (options.dry_run) return;
    let allResults = initResult();
    let start = new Date();
    for (let path of pathList) {
        let result = await _validateFile(path, options)
            .then(result => {
                //allResults = aggregate(result, allResults);
                return result;
            }).catch(err => {
                if (err.code === 'ENOENT') {
                    Log.info('file not found');
                } else {
                    throw err;
                }
            });
    }
    if (options.perf) {
        let duration = new Duration(start, new Date()).milliseconds;
        Log.info(`validateFromPathList duration(${PrettyMs(duration)})` +
                 `, getCountList(${PrettyMs(allResults.timing.getCountList)})`);
    }
    const count = allResults.count;
//    Log.message(`new: known(${count.knownClues}), maybe(${count.maybeClues}), rejects(${count.rejectClues})`);
//    return saveClues(allResults, options);
}

//

module.exports = {
    validateFile,
    validatePathList
}
