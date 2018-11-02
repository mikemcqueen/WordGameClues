//
// filter.js
//
// TODO:
//   move some methods to modules/filter
//

'use strict';

const _            = require('lodash');
const ChildProcess = require('child_process');
const ClueManager  = require('../modules/clue-manager');
const Clues        = require('../modules/clue-types');
const Debug        = require('debug')('filter');
const Dir          = require('node-dir');
const Duration     = require('duration');
const Expect       = require('should/as-function');
const Filter       = require('../modules/filter');
const Fs           = require('fs-extra');
const Getopt       = require('node-getopt');
const Markdown     = require('../modules/markdown');
const My           = require('../modules/util');
const Path         = require('path');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');
const Score        = require('../modules/score');
const SearchResult = require('../modules/search-result');
const Stringify    = require('stringify-object');

//

const csvParse     = Promise.promisify(require('csv-parse'));

const Options = new Getopt(_.concat(Clues.Options, [
    ['',  'add-known',           'add known clues'],
    ['a', 'article',             'filter results based on article word count'],
    ['',  'copy',                'copy to clipboard as RTF'],
    ['d', 'dir=NAME',            'directory name'],
    ['',  'known-urls',           'filter known URLs'],
    ['',  'keep', 'keep tmp file'],
    ['m', 'match=EXPR',          'filename match expression' ],
    ['n', 'count',               'show result/url counts only'],
    ['',  'parse',               'parse a filter output file'],
    ['',  'note',                'mail results to evernote'], 
    //['',  'merge',               ''], 
    // ['r', 'rejects',             'show only results that fail all filters'],
    ['t', 'title',               'filter results based on title word count (default)'],
    ['x', 'xfactor=VALUE',       'show 1) missing URL/title/summary 2) unscored URLs 3) article < summary'], 
    ['w', 'word',                'search for additional word'],
    ['h', 'help',                'this screen']
])).bindHelp(
    "Usage: node filter <options> [wordListFile]\n\n[[OPTIONS]]\n"
);

//

const XFACTOR_MISSING  = 1;
const XFACTOR_UNSCORED = 2;
const XFACTOR_BADSCORE = 3;

//

function usage (msg) {
    if (msg) {
        console.log(`\n${msg}\n`);
    }
    Options.showHelp();
    process.exit(-1);
}

//

function getUrlCount (resultList) {
    Expect(resultList).is.an.Array();
    // TODO _.reduce()
    let urlCount = 0;
    for (const result of resultList) {
        urlCount += _.size(result.urlList);
    }
    return urlCount;
}

//

function isUrlInList (url, list) {
    Expect(url).is.a.String();
    if (_.isUndefined(list)) return false; 
    Expect(list).is.an.Array();
    return list.includes(url);
}

//

function isRejectUrl (url, filteredUrls) {
    return isUrlInList(url, filteredUrls && filteredUrls.rejectUrls);
}

//

function isKnownUrl (url, filteredUrls) {
    return isUrlInList(url, filteredUrls && filteredUrls.knownUrls);
}

// check for broken results if xfactor option specified

function isXFactor (result, options) {
    Expect(result).is.an.Object();
    Expect(options).is.an.Object();
    if (options.xfactor === XFACTOR_MISSING) { 
        const missing = _.isEmpty(result.url) || _.isEmpty(result.title) || _.isEmpty(result.summary);
        if (missing) {
            console.log(`${JSON.stringify(result)}`);
        }
        return missing;
    }
    if (options.xfactor === XFACTOR_UNSCORED) {
        return _.isEmpty(result.score);
    }
    if (options.xfactor === XFACTOR_BADSCORE && _.isObject(result.score)) {
        return result.score.wordsInArticle < result.score.wordsInSummary;
    }
    return false;
}

// filter out rejected urls

function filterSearchResultList (resultList, wordList, filteredUrls, options, filepath) {
    Expect(resultList).is.an.Array();
    Expect(wordList).is.an.Array().and.not.empty();
    Expect(options).is.an.Object();
    let loggedXFactor = false;
    // make any clue words with a space into multiple words.
    let wordCount = _.chain(wordList).map(word => word.split(' ')).flatten().size().value();
    
    Debug(`++filterResultList for ${wordList}`);
    if (filteredUrls) {
        Debug(    `, known(${filteredUrls.knownUrls.length})` +
                  `, reject(${filteredUrls.rejectUrls.length})`);
    }
    if (options.verbose) {
        Debug(`${JSON.stringify(filteredUrls)}`);
    }
    // first filter out all rejected, unscored, known (if applicable), and below-score results
    return Promise.filter(resultList, result => {
        //Debug(`result: ${_.entries(result)}`);
        if (options.xfactor && isXFactor(result, options)) {
            if (!loggedXFactor) {
                console.log(`x: ${filepath}`);
                loggedXFactor = true;
            }
            return false;
        }
        if (isRejectUrl(result.url, filteredUrls) || _.isUndefined(result.score)) {
            Debug(`filtering reject or unscored url, ${result.url}, score ${result.score}`);
            return false;
        }
        if (options.filter_known_urls && isKnownUrl(result.url, filteredUrls)) {
            Debug(`filtering known url, ${result.url}`);
            return false;
        }
        return Score.wordCountFilter(result.score, wordCount, options);
    }).then(resultList => {
        return {
            src:     wordList.toString(),
            urlList: resultList.map(result => result.url), // only good results remain; map to urls
            known:   ClueManager.getKnownClueNames(wordList)
        };
    });
    // TODO: .catch()
}

//

function loadFilteredUrls (dir, wordList, options) {
    Expect(dir).is.a.String();
    Expect(wordList).is.an.Array();
    Expect(options).is.an.Object();
    let filteredFilename = SearchResult.makeFilteredFilename(wordList);
    Debug(`filtered filename: ${filteredFilename}`);
    return Fs.readFile(Path.format({ dir, base: filteredFilename }), 'utf8')
        .then(content => {
            Debug(`resolving filtered urls for: ${wordList}`);
            return JSON.parse(content);
        }).catch(err => {
            if (err && err.code !== 'ENOENT') throw err;
            Debug(`no filtered urls, ${wordList}`);
            return undefined;
        });
}

//

function hasRemaining (wordListArray, remaining) {
    Expect(wordListArray).is.an.Array();
    Expect(remaining).is.an.Array().and.not.empty();
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
    Expect(nameMap).is.an.Object();       // must be object if defined
    Expect(wordList).is.an.Array().with.property('length').above(0); // at.least(1)
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
    Expect(dir).is.a.String('fSRD dir');
    Expect(fileMatch).is.a.String('fSRD filematch');
    Expect(options).is.an.Object();
    return new Promise((resolve, reject) => {
        let filteredList = [];
        let rejectList = [];
        Dir.readFiles(dir , {
            match:     new RegExp(fileMatch),
            exclude:   /^\./,
            recursive: false
        }, function(err, content, filepath, next) {
            if (err) throw err; // TODO: test
            Debug(`filename: ${filepath}`);
            let wordList = SearchResult.makeWordlist(filepath);
            // filter out rejected word combos
            if (ClueManager.isRejectSource(wordList)) return next();
            if (!_.isUndefined(options.nameMap) && !isInNameMap(options.nameMap, wordList)) return next();
            // temp

            /*
            if (!_.isUndefined(options.nameMap)) {
                console.log(`filename: ${filepath}`);
                return next();
            }
             */
            loadFilteredUrls(Path.dirname(filepath), wordList, options)
                .then(filteredUrls => {
                    return filterSearchResultList(JSON.parse(content), wordList, filteredUrls, options, filepath);
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
                    console.log(`filterSearchResultDir, path: ${filepath}`, err, err.stack);
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
// for each result file path in pathlist
//   load result file
//   make word list from filename
//   

function filterPathList (pathList, dir, options) {
    Expect(pathList).is.an.Array();
    Expect(options).is.an.Object();

    let rejectList = [];
    // map to array of result || undefined
    return Promise.map(pathList, path => {
        let filename = Path.basename(path);
        Debug(`filename: ${filename}`);
        let wordList = SearchResult.makeWordlist(filename);
        return Promise.all(
            [Fs.readFile(path), loadFilteredUrls(Path.dirname(path), wordList, options)])
            .then(([content, filteredUrls]) => {
                return filterSearchResultList(JSON.parse(content), wordList, filteredUrls, options, path);
            }).then(filterResult => {
                // TODO: this is probably wrong for rejects
                if (_.isEmpty(filterResult.urlList) && _.isEmpty(filterResult.known)) {
                    Debug(`rejecting, ${wordList}, no urls or known clues`); 
                    rejectList.push(filterResult);
                    return undefined;
                }
                return filterResult;
            }).catch(err => {
                // report & eat all errors
                // THIS IS SO VERY VERY WRONG. SPITTING AN ERROR OUT TO A FILTERED OUTPUT
                // MAKES NO SENSE IN SOME CASES - SHOULD ABORT
                console.log(`filterSearchResultFiles, path: ${path}`, err, err.stack);
            });
    }).then(resultList => resultList.filter(result => !_.isUndefined(result)))
    .then(filteredList => {
        return {
            filtered: filteredList,
            rejects:  rejectList
        };
    });
}

// TODO: move to Util.js

function loadCsv (filename) {
    return Fs.readFile(filename, 'utf8')
        .then(csvContent => csvParse(csvContent, { relax_column_count: true }));
}

//

function buildNameMap (wordListArray) {
    let map = {};
    for (const wordList of wordListArray) {
        for (const word of wordList) {
            let remaining = _.difference(wordList, [word]);
            Expect(remaining.length).is.above(0, `${wordList}, ${word}`); // at.least(1)
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

function getPathList (dir, fileMatch, nameMap) {
    Expect(dir).is.a.String('gPL dir');
    Expect(fileMatch).is.a.String('gPL filematch');
    return new Promise((resolve, reject) => {
        Dir.files(dir, (err, pathList) => {
            if (err) reject(err);
            let match = new RegExp(fileMatch);
            let filtered = _.filter(pathList, path => {
                let filename = Path.basename(path);
                let wordList = SearchResult.makeWordlist(filename);
                // filter out rejected word combos
                if (ClueManager.isRejectSource(wordList)) return false;
                if (!_.isUndefined(nameMap) && !isInNameMap(nameMap, wordList)) return false;
                return match.test(filename);
            });
            Debug(`files(${pathList.length}), match(${filtered.length})`);
            resolve(filtered);
        });
    });
}

//

function writeFilterResults (resultList, stream, options) {
    Expect(resultList).is.an.Array();
    for (const result of resultList) {
        if (ClueManager.isRejectSource(result.src)) continue;
        const knownList = ClueManager.getKnownClueNames(result.src);
        if (_.isEmpty(knownList) && _.isEmpty(result.urlList)) continue;

        My.logStream(stream, `${Markdown.Prefix.source}${result.src}`);
        for (const url of result.urlList) {
            My.logStream(stream, url);
        }
        if (options.add_known_clues && !_.isEmpty(knownList)) {
            My.logStream(stream, Filter.KNOWN_CLUES_URL);
            for (const name of knownList) {
                My.logStream(stream, name); // `${Markdown.Prefix.known}${name}`);
            }
        }
        My.logStream(stream, '');
    }
}

// TODO: move to util.js

function mailTextFile(options) {
    let pipe = false;
    let fd = Fs.openSync(options.path, 'r');
    let child = ChildProcess.spawn('mail', ['-s', `${options.subject}`, `${options.to}`], {
        stdio: [pipe ? 'pipe' : fd, 1, 2]
    });
    if (pipe) {
        let s = Fs.createReadStream(null, { fd });
        s.pipe(child.stdin);
        s.on('data', (data) => {
            console.log('s.data');
        });
        s.on('end', () => {
            console.log('s.end');
            child.stdin.end();
        });
    }
}

// TODO: move to util.js, comment purpose

function copyTextFile(path) {
    let fd = Fs.openSync(path, 'r');
    let textutil = ChildProcess.spawn(
        'textutil', ['-format', 'txt', '-convert', 'rtf', '-stdout', `${path}`],
        { stdio: [fd, 'pipe', 2] });    
    let pbcopy = ChildProcess.spawn('pbcopy', ['-Prefer', 'rtf']);
    textutil.stdout.pipe(pbcopy.stdin);
    textutil.stdout.on('end', () => {
        Debug('textutil.end');
        Fs.closeSync(fd);
    });
}

//
//
//
async function main () {
    const opt = Options.parseSystem();
    const options = opt.options;

    if (opt.argv.length > 1) {
        usage('only one non-switch FILE argument allowed');
    }
    const filename = opt.argv[0];

    if (options.parse) {
        const resultList = Filter.parseFileSync(filename, options);
        if (_.isEmpty(resultList)) {
            console.log('no results');
        } else {
            for (const result of resultList) {
                console.log(result);
            }
        }
        return undefined;
    }
    options.dir = options.dir || '2';
    if (options['known-urls']) options.filter_known_urls = true;
    if (options['add-known']) options.add_known_clues = true;

    ClueManager.loadAllClues({ clues: Clues.getByOptions(options) });

    // default to title filter if no filter specified
    if (!options.article && !options.title) {
        options.title = true;
    }
    let nameMap;
    if (opt.argv.length > 0) {
        let wordListArray = await loadCsv(filename);
        nameMap = buildNameMap(wordListArray);
    }
    const dir = SearchResult.DIR + options.dir;
    const filterOptions = {
        filterArticle: options.article,
        filterTitle:   options.title,
        filterRejects: options.rejects,
        xfactor:       _.toNumber(options.xfactor)
    };
    let start = new Date();
    const pathList = await getPathList(dir, SearchResult.getFileMatch(options.match), nameMap)
              .catch(err => { throw err; });
    const getDuration = new Duration(start, new Date()).milliseconds;
    start = new Date();
    const result = await filterPathList(pathList, dir, filterOptions)
              .catch(err => { throw err; });
    const filterDuration = new Duration(start, new Date()).milliseconds;
    if (options.count) {
        console.log(`Results: ${_.size(result.filtered)}` +
                    `, Urls: ${getUrlCount(result.filtered)}` +
                    `, Rejects: ${_.size(result.rejects)}` +
                    `, get(${PrettyMs(getDuration)})` +
                    `, filter(${PrettyMs(filterDuration)})`);
        return undefined;
    }

    if (options.xfactor) return undefined;

    // result.filtered is a list of result objects,
    //   which contain urlLists
    // if options.word, we want to re-score all of those
    //   urls using only options.word
    // re-scoring should give us a list of "scored results"
    //   similar to what is saved in result dir. but we don't
    //   need to save those results this time.
    // then we want to call something similar to filterSearchResultList
    //   which just calls score.wordCountFilter() on those
    //   re-scored results, and without filtering "filtered" URLs
    // now we have two result.filtered lists. one contains the
    //   original URLs that matched the filename words,
    //   and one contains the URLS that match options.word
    // so the second one we can just use, right?
    //
    // other considerations:
    //
    // maybe we want to save the score data for the --word in the
    //   original results file. that way a subsequent --word filter
    //   woudln't have to do the wikipedia search.
    //
    // of course, we could just save the wikipedia download data as well.
    //   is there a database that auto-indexes all words in a document?
    //

    const useTmpFile = options.note || options.copy;
    return Promise.resolve().then((_ => {
        if (useTmpFile) {
            return My.createTmpFile(Boolean(options.keep)).then(([path, fd]) => {
                return [path, Fs.createWriteStream(null, { fd })];
            });
        }
        return [null, process.stdout];
    })).then(([path, stream]) => {
        writeFilterResults(options.rejects ? result.rejects : result.filtered, stream, options);
        if (useTmpFile) {
            stream && stream.end();
            if (options.mail) {
                mailTextFile({
                    to:      'zippy@pinhead.com',
                    subject: `filtered @Worksheets.new`,
                    path:    path
                });
            } else if (options.copy) {
                copyTextFile(path);
            }
        }
        return undefined;
    });
}

//

main().catch(err => {
    console.log(err, err.stack);
});


