//
// search.js
//

'use strict';

const _            = require('lodash');
const Debug        = require('debug')('search');
const Expect       = require('should/as-function');
const Fs           = require('fs-extra');
const Ms           = require('ms');
const My           = require('./util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const SearchResult = require('./search-result');

// make a search term from a list of words and the supplied options
//
function makeSearchTerm (wordList, options = {}) {
    Expect(options).is.an.Object();
    let term = wordList.join(' ');
    if (options.wikipedia) {
        term += ' site:en.wikipedia.org';
    }
    return term;
}

// 
//
function getOneResult (wordList, pages, options = {}) {
    Expect(options).is.an.Object();
    return new Promise((resolve, reject) => {
        let term = makeSearchTerm(wordList, { wikipedia: true });
        console.log(`term: ${term}, pages: ${pages}`);
        SearchResult.get(term, pages, (err, data) => {
            if (!err && options.reject) {
                err = new Error('getOneResult: forced rejection');
            }
            if (err) {
                console.log('getOneResult: rejecting');
                reject(err);
            } else {
                resolve(data);
            }
        });
    });
}

//

async function retryGetOneResult (wordList, pages, options = {}) {
    Expect(options).is.an.Object();
    const retryDelay = options.retryDelay || 5000;
    while (true) {
        const result = await getOneResult(wordList, pages, options)
                  .catch(err => {
                      if (_.includes(err.message, 'CAPTCHA')) {
                          console.log('CAPTCHA error');
                      } else {
                          console.log(err);
                      }
                      return false;
                  });
        if (result !== false) return result;
        console.log(`Retrying in ${PrettyMs(retryDelay)}...`);
        await My.waitFor(retryDelay);
    }
}

// check if file exists; if not, get, then save, a search result

function checkGetSaveResult(args, options) {
    let nextDelay = 0;
    let mode = options.force ? 'w' : 'wx';
    return Fs.open(args.path, mode)
        .then(fd => {
            Expect(fd).is.above(-1);
            return Fs.close(fd);
        }).then(_ => {
            // we are going to do a search; set delay for next search.
            // NOTE: need to set this here, rather than on successful
            // search, because we may get robot warning
            // TODO: add retry to getOneResult; check for robot result
            nextDelay = args.delay;
            let oneResultOptions = {
                reject: options.forceNextError,
                retryDelay:  args.delay * 2
            };
            options.forceNextError = false;
            // file did not exist prior to creation; do the search
            return retryGetOneResult(args.wordList, args.pages, oneResultOptions);
        }).then(oneResult => {
            if (_.isEmpty(oneResult)) {
                // empty search result. 
                // save an empty result so we don't do this search again
                args.count.empty += 1;
                oneResult = [];
            } else {
                args.count.data += 1;
            }
            return Fs.writeFile(args.path, JSON.stringify(oneResult))
                .then(_ => {
                    console.log(`Saved: ${args.path}`);
                    return [oneResult, nextDelay];
                });
        }).catch(err => {
            if (err) {
                if (err.code === 'EEXIST') {
                    args.count.skip += 1;
                    console.log(`Skip: file exists, ${args.path}`);
                } else {
                    console.log(`checkGetSaveResult error, ${err}`);
                    args.count.error += 1;
                }
            }
            return [null, nextDelay];
        });
}

//args:
// wordListArray : array of array of strings
// pages         : # of pages of results to retrieve for each wordlist
// delay         : object w/properties: high,low - ms delay between searches
// root          : root results directory (optional; default: Results.dir)
// dir           : directory within root to store results (optional, default: wordList.length)
//
// options:
//   force       : search even if results file already exists (overwrites. TODO: append new results, instead)
//   forceNextError: test support, sets getOnePromise.options.reject one time
//
async function getAllResultsLoop (args, options) {
    Expect(args).is.an.Object();
    Expect(args.wordListArray).is.an.Array().and.not.empty();
    Expect(options).is.an.Object();
    let count = { skip: 0, empty: 0, data: 0, error: 0 }; // test support
    for (const [index, wordList] of args.wordListArray.entries()) {
        let filename = SearchResult.makeFilename(wordList);
        console.log(`list: ${wordList}`);
        console.log(`file: ${filename}`);

        let path = SearchResult.pathFormat({
            root: args.root,
            dir:  args.dir || _.toString(wordList.length),
            base: filename
        }, options);
        Debug(path);

        let [result, nextDelay] = await checkGetSaveResult({
            wordList,
            path,
            count,
            pages: args.pages,
            delay: My.between(args.delay.low, args.delay.high)
        }, options);

        if (result) {
            // NOTE: we intentionally do NOT await completion of the following
            // add-commit-score-commit operations. they can be executed asynchronously
            // with the execution of this loop. makes logs a little messier though.
            My.gitAddCommit(path, 'new result')
                .then(() => {
                    console.log(`Committed: ${path}`);
                    return !_.isEmpty(result) && SearchResult.fileScoreSaveCommit(path);
                }).catch(err => {
                    // log & eat all errors
                    console.log(`getAllResultsLoop commit error`, err, err.stack);
                });
        }

        // if there are more wordlists to process
        const remaining = args.wordListArray.length - index - 1;
        if (remaining > 0) {
            // if nextDelay is specified, delay before next search
            if (nextDelay > 0) {
                console.log(`Delaying ${PrettyMs(nextDelay)}, ${remaining} remaining...`);
                await My.waitFor(nextDelay);
            }
        }
    }
    return count;
}

//
//
function getAllResults (args, options = {}) {
    Expect(args).is.an.Object();
    Expect(args.wordListArray).is.an.Array().and.not.empty();
    return new Promise((resolve, reject) => {
        getAllResultsLoop(args, options)
            .then(data => resolve(data))
            .catch(err => reject(err));
    });
}

//

module.exports = {
    getAllResults
}
