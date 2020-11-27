//
// search-result.js
//

const _            = require('lodash');
const Debug        = require('debug')('search-result');
const Expect       = require('should/as-function');
const Fs           = require('fs-extra');
const GoogleIt     = require('google-it');
const Ms           = require('ms');
const My           = require('./util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');
const Score        = require('./score');

//

const RESULT_DIR      =  Path.normalize(`${Path.dirname(module.filename)}/../../data/results/`);
const FILTERED_SUFFIX = '_filtered';
const EXT_JSON        = '.json';

// '../../file-path.json' => [ 'file', 'path' ]
//
function makeWordlist (filepath) {
    Expect(filepath).is.a.String();
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
    Debug(wordList);
    Expect(wordList.length).is.aboveOrEqual(2);
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
    Expect(wordList).is.an.Array().and.not.empty();
    return makeFilename(wordList, FILTERED_SUFFIX);
}

//

function old_pathFormat (args) {
    Expect(args).is.an.Object()
    Expect(args.dir).is.a.String();
    let root = _.isUndefined(args.root) ? RESULT_DIR : args.root;
    args.dir = root + args.dir; 
    // Path.format ignores args.root if args.dir is set
    return Path.format(args);
}

//

function makeResultDir (args, options) {
    const root = args.root || RESULT_DIR;
    let resultDir = `${root}${args.dir}`;
    if (!options.flat) {
        const baseDir = args.base.slice(0, 1);
        resultDir += `/${baseDir}`;
    }
    Debug(`flat: ${options.flat}, resultDir: ${resultDir}`);
    return resultDir;
}

//

function pathFormat (args, options = {}) {
    Expect(args).is.an.Object();
    Expect(args.dir).is.a.String();
    let formatArgs = _.clone(args);
    formatArgs.dir = makeResultDir(args, options);
    // Path.format ignores args.root if args.dir is set
    return Path.format(formatArgs);
}

//

function get (text, pages, cb) {
    Expect(text).is.a.String();
    Expect(pages).is.a.Number().above(0);
    Expect(cb).is.a.Function();
    let resultList = [];
    let count = 0;
    Google(text, function (err, result) {
        if (err) return cb(err);
        Debug(`result count: ${_.size(result.links)}`);
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
        Expect(count).is.below(pages);
        Expect(result.next).is.ok();
        let msDelay = My.between(Ms('30s'), Ms('60s'));
        console.log(`Delaying ${PrettyMs(msDelay)} for next page of results...`);
        // TODO async function, use await My.delay(msDelay)
        setTimeout(result.next, msDelay);
        return undefined;
    });
}

function get_it (text, limit = 20) {
    Expect(text).is.a.String();
    Expect(limit).is.a.Number().above(0);
    let resultList = [];

    let options = {};

    return GoogleIt({'query': text, 'limit': limit})
        .then(results => {
            Debug(`result count: ${_.size(results)}`);



	    Debug(`results: ${results}, results[0].keys: ${_.keys(results[0])}`);
            resultList.push(...results.map(entry => {
                return {
                    title:   entry.title,
                    url:     entry.link
                };
            }));
            return resultList;
        }).catch(err => {
            console.log('get-it ' + err);
        });
}

//

function scoreSaveCommit (resultList, filepath, options = {}, wordList = undefined) {
    Expect(resultList).is.an.Array().and.not.empty();
    Expect(filepath).is.a.String();
    Expect(options).is.a.Object();
    if (!_.isUndefined(wordList)) {
        Expect(wordList).is.an.Array().with.property('length').above(1); // at.least(2);
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
        return Fs.writeFile(filepath, JSON.stringify(list))
            .then(() => console.log(' updated'));
    }).then(() => {
        return My.gitCommit(filepath, 'updated score')
            .then(() => console.log(' committed'));
    });
}

//

function fileScoreSaveCommit (filepath, options = {}, wordList = undefined) {
    Expect(filepath).is.a.String();
    Expect(options).is.a.Object();
    if (!_.isUndefined(wordList)) {
        Expect(wordList).is.an.Array().with.property('length').above(1); // at.least(2)
    }
    return Fs.readFile(filepath, 'utf8')
        .then(data => scoreSaveCommit(JSON.parse(data), filepath, options, wordList));
}

//

module.exports = {
    fileScoreSaveCommit,
    get,
    get_it,
    getFileMatch,
    makeFilename,
    makeFilteredFilename,
    makeWordlist,
    pathFormat,
    scoreSaveCommit,

    EXT_JSON,
    DIR: RESULT_DIR
};
