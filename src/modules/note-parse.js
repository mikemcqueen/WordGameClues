/*
 * note-parse.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('note-parse');
const Expect           = require('should/as-function');
const Fs               = require('fs-extra');
const Markdown         = require('./markdown');

// all options are boolean
//   .urls:     parse urls (http:// prefix)
//   .clues:    parse clues (no prefix, or '-' remove prefix)
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
    Debug(`++note-parse.parse()`);
    if (_.isBuffer(text)) text = text.toString();
    Expect(text).is.a.String();
    
    const sourceExpr = />(@[^<]+)</g;
    const urlExpr =    />(http[s]?\:[^<]+)</g;
    const clueExpr =   />([^<]+)</g;

    let resultList = [];
    let sourceResult;
    let prevSourceResult;
    let urlResult = urlExpr.exec(text);
    let prevUrlResult;
    let clueResult = clueExpr.exec(text);
    let prevClueResult;
    let sourceElement;
    let prevSourceElement;
    let done = false;
    while (!done) {
	prevSourceElement = sourceElement;
	prevSourceResult = sourceResult;
	sourceResult = sourceExpr.exec(text);
	let sourceIndex;
	if (sourceResult) {
	    // result[1] = 1st capture group
	    const match = sourceResult[1];
	    const [sourceLine, suffix] = Markdown.getSuffix(match);
	    Debug(`sourceLine: ${sourceLine}, suffix: ${suffix}`);
	    sourceElement = options.urls ? { source: sourceLine } : sourceLine;
	    if (options.urls && suffix) sourceElement.suffix = suffix;
	    resultList.push(sourceElement);
	    sourceIndex = sourceResult.index;
	} else {
	    sourceIndex = Number.MAX_SAFE_INTEGER;
	    done = true;
	}
	if (!options.urls || !prevSourceElement) continue;
	
	// parse urls for previous source

	// move urlExpr position to prevSourceResult
	while ((urlResult !== null) && (urlResult.index < prevSourceResult.index)) {
	    //urlExpr.lastIndex = prevSourceResult.index;
	    //Debug(`advanced urlExpr.lastIndex to ${urlExpr.lastIndex}`);
	    urlResult = urlExpr.exec(text); // for while loop
	}
	Debug(`urlResult.index = ${urlResult.index}, sourceIndex = ${sourceIndex}`);
	let urlList = [];
	// while urlResult is at a position before the current sourceResult
	while ((urlResult !== null) && (urlResult.index < sourceIndex)) {
	    const urlLine = urlResult[1];
	    Debug(`urlLine: ${urlLine}`);
	    // NOTE: suffix on a url inside a note is within a separate <div> block
	    //const [_, urlLine] = My.hasCommaSuffix(urlResult[1], RejectSuffix); // ValidSuffixes
	    let urlElement = options.clues ? { url: urlLine, clues: [] } : urlLine;
	    urlList.push(urlElement);
	    prevUrlResult = urlResult;
	    urlResult = urlExpr.exec(text);
	    if (!options.clues) continue;

	    // parse clues for previous url

	    // clue parsing ends at next url or next source, whichever is first
	    let urlIndex = urlResult ? urlResult.index : Number.MAX_SAFE_INTEGER;
	    let endClueIndex = urlIndex < sourceIndex ? urlIndex : sourceIndex;
	    let clueList = [];
	    // move clueExpr position to prevUrlResult
	    while ((clueResult !== null) && (clueResult.index < prevUrlResult.index)) {
		clueResult = clueExpr.exec(text);
	    }
	    let count = 0;
	    while ((clueResult !== null) && (clueResult.index < endClueIndex)) {
		const clueLine = clueResult[1].trim();
		let debugMsg;
		// <a> element inner text (with url) gets picked up by clueExpr regex
		if (_.startsWith(clueLine, 'http')) {// TODO: Filter.isUrl/or My.isUrl/or Markdown.isUrl
		    // ignore if first, fail if > first
		    if (count > 0) {
			throw new Error(`encountered http where clue was expected, ${clueLine}, count ${count}`);
		    }
		    debugMsg = 'ignored';
		} else if (clueLine.charAt(0) === ',') { 
		    // a comma is a valid clue starting character only if it is in fact not a clue,
		    // but a instead suffix to a url.  which means it must immediately follow a url,
		    // which was actually ignored as the first "clue", above.
		    // here (count > 1) allows for the possibility that there is no inner text to the
		    // <a> element (which would be the url ignored, above).
		    if (count > 1) {
			throw new Error(`encountered unexpected comma where clue was expected: ${clueLine}`);
		    }
		    let [base, suffix] = Markdown.getSuffix(clueLine);
		    if (!suffix) {
			throw new Error(`encountered invalid comma or suffix: ${clueLine}`);
		    }
		    urlElement.suffix = suffix;
		    debugMsg = `adding suffix to URL, ${suffix}`;
		} else if (Markdown.hasSourcePrefix(clueLine)) {
		    throw new Error(`encountered unexpected source where clue was expected, ${clueLine}`);
		} else if (!_.isEmpty(clueLine)) {
		    // TODO: 'note', 'need' markdowns
		    let [line, prefix] = Markdown.getPrefix(clueLine);
		    if (prefix) {
			clueLine = line;
		    }
		    clueList.push({ clue: clueLine, prefix });
		    debugMsg = 'added';
		} else {
		    debugMsg = 'empty';
		}
		Debug(`clueLine: ${clueLine || ''} (${debugMsg})`);
		clueResult = clueExpr.exec(text);
		count += 1;
	    }
	    urlElement.clues = clueList;
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
