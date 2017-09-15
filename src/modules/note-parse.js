/*
 * note-parse.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('note-parse');
const Expect           = require('should/as-function');
const Fs               = require('fs-extra');
const My               = require('./util');

//

const RejectSuffix =  'x'; // copied from update.js :(

// all options are boolean
//   .urls:     parse urls (http:// prefix)
//   .clues:    parse clues (no prefix)
//
// returns (eventually)
//   if only one options is supplied, returns a list of that option type
//   if two options are supplied returns a list of objects, which each contain a
//     singular outer option as a scalar property, and the inner option as a list.
//     for example, .sources & .urls returns:
//       [ { source: value1, urls: [url1, .,. urlN] },
//         { source: value2, urls: [url1, .., urlN] } ]
//     .sources &  .urls & .clues returns:
//       [ { source: value1, urls: [{ url: url1, clues: [clue1, .., clueN] },
//                                  { url: url2, clues: [clue1, .., clueN] }] },
//         { source: value2, urls: [{ url: url1, clues: [clue1, .., clueN] },
//                                  { url: url2, clues: [clue1, .., clueN] }] } ]
//
function parse (text, options = {}) {
    Expect(options.clues).is.undefined('not yet supported');

    const sourceExpr = />@([^<]+)</g;
    const urlExpr =    />(http[s]?\:[^<]+)</g;
    const clueExpr =   />([^<]+)</g;

    let resultList = [];
    let sourceResult;
    let urlResult = urlExpr.exec(text);
    let prevSourceResult;
    let prevUrlResult;
    let sourceElement;
    let prevSourceElement;
    let done = false;
    Debug(`urlResult: ${urlResult}`);
    while (!done) {
	prevSourceElement = sourceElement;
	prevSourceResult = sourceResult;
	sourceResult = sourceExpr.exec(text);
	let sourceIndex;
	if (sourceResult) {
	    // result[1] = 1st capture group
	    const [_, sourceLine] = My.hasCommaSuffix(sourceResult[1], RejectSuffix);
	    Debug(sourceLine);
	    sourceElement = (options.urls) ? { source: sourceLine } : sourceLine;
	    resultList.push(sourceElement);
	    sourceIndex = sourceResult.index;
	} else {
	    sourceIndex = Number.MAX_SAFE_INTEGER;
	    done = true;
	}
	if (!options.urls || !prevSourceElement) continue;
	
	// for each (previous) source, get list of urls
	let urlList = [];
	// move urlResult to a position past the previous sourceResult
	while ((urlResult !== null) && (urlResult.index < prevSourceResult.lastIndex)) {
	    Debug('advancing urlResult');
	    urlResult = urlExpr.exec(text);
	}
	Debug(`url.index = ${urlResult.index}, sourceIndex = ${sourceIndex}`);
	// while urlResult is at a position before the current sourceResult
	while ((urlResult !== null) && (urlResult.index < sourceIndex)) {
	    const urlLine = urlResult[1];
	    Debug(urlLine);
	    //const [_, urlLine] = My.hasCommaSuffix(urlResult[1], RejectSuffix); // ValidSuffixes
	    let urlElement = (options.clues) ? { url: urlLine, clues: [] } : urlLine;
	    urlList.push(urlElement);
	    prevUrlResult = urlResult;
	    urlResult = urlExpr.exec(text);
	}
	prevSourceElement.urls = urlList;
	Debug(`found ${urlList.length} urls for ${prevSourceElement.source}`);
    }
    return resultList;
}

//

function parseFile (filename, options) {
    return Fs.readFile(filename)
	.then(content => {
	    return parse(content.toString(), options);
	});
}

//

module.exports = {
    parse,
    parseFile
};
