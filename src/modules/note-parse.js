/*
 * note-parse.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('note-parse');
const DomParser        = require('xmldom').DOMParser;
const Expect           = require('should/as-function');
const Filter           = require('./filter'); // shouldn't be necessary
const Fs               = require('fs-extra'); 
const Markdown         = require('./markdown');

//

const Tag = {
    break: 'br',
    div:   'div',
    link:  'a',
    text:  '#text'
};

//  div: defines a line 
//   all #text concatenate.
//   if an outer div has an inner div, outer div cannot have text.

function processDiv (node, div, queue) {
    if (div) {
	div.noText = true;
    }
    queue.push({});
}

// #text:
//  cannot exist outside of a div
//  fail if div.noText (current div has inner div)
//  cannot exist in div that contains a break
//  fail if follows link & not comma prefixed
//  fail if follows link & link suffix already supplied

function processText (node, div, queue) {
    Expect(div).is.ok();
    Expect(div.noText).is.not.true();
    Expect(div.break).is.not.true();
    const text = node.textContent;
    if (div.link) {
	if (!div.linkText) {
	    Expect(_.startsWith(text, 'http')).is.true();
	    div.linkText = true;
	} else {
	    Expect(div.linkSuffix).is.not.true();
	    Expect(text.charAt(0)).is.equal(',');
	    div.linkSuffix = true;
	}
    }
    return text;
}

// a: link
//  must be in div
//  only one per div
//  cannot exist in div that contains a break
//  cannot have preceding text
//  can have one succeeding text with ',' prefix
//  next tag must be Tag.text, with http prefix

function processLink (node, div, divQueue) {
    Expect(div).is.ok();
    Expect(div.link).is.not.true();
    Expect(div.break).is.not.true();
    Expect(div.text).is.not.ok();
    div.link = true;
    div.nextTag = Tag.text;
    //return node.textContent;
}

// br: break.
//  cannot exist outside of a div
//  only one per div?
//  link not allowed in same div
//  text not allowed in same div

function processBreak (node, div, divQueue) {
    Expect(div).is.ok();
    Expect(div.break).is.not.true();
    Expect(div.link).is.not.true();
    Expect(div.text).is.not.ok();
    div.break = true;
    return '';
}

//

function isDiv (node) {
    return node.nodeName === Tag.div;
}

// Nodes we care about:
//  a:     only one per div, cannot have preceding text, can have one succeeding text with ',' prefix
//  br:    empy line, currenly only allow in div without text
//  div:   defines a line (all text concatenate. if an outer div has an inner div, outer div cannot have text.
//  #text: cannot exist outside of a <div>
//

const tagMap = {
    [Tag.break]: processBreak,
    [Tag.div]:   processDiv,
    [Tag.link]:  processLink,
    [Tag.text]:  processText
};

function parseDomLines (lines, node, queue, options) {
    if (_.has(tagMap, node.nodeName)) {
	let div = _.last(queue);
	let expected;
	if (div && div.nextTag) {
	    Expect(div.nextTag).is.equal(node.nodeName);
	    div.nextTag = undefined;
	}
	let text = tagMap[node.nodeName](node, div, queue);
	if (text) {
	    Expect(div).is.ok();
	    if (!div.text) div.text = text;
	    else div.text += text;
	}
    }
    if (node.childNodes) {
	Array.prototype.forEach.call(node.childNodes, child => {
	    parseDomLines (lines, child, queue, options);
	});
    }
    if (isDiv(node)) {
	let div = queue.pop();
	Expect(div).is.ok();
	let text = div.text && div.text.trim();
	if (text && !_.isEmpty(text)) {
	    Debug(`line: ${text}`);
	    lines.push(text);
	} else if (!_.isEmpty(_.last(lines))) {
	    Debug('empty line');
	    lines.push('');
	}
    }
}

// 
    
function parseDom (xmlText, options = {}) {
    const doc = new DomParser().parseFromString(xmlText);
    let node = doc.documentElement;
    let lines = [];
    let divQueue = [];
    parseDomLines(lines, node, divQueue, options);
    return lines;
}

// all options are boolean
//   .urls:     parse urls (http(s) prefix)
//   .clues:    parse clues (removed, maybe)
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
    Debug(`++parse()`);
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
	    Debug(`sourceLine: ${sourceLine} suffix: ${suffix}`);
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
	Debug(`urlResult.index = ${urlResult && urlResult.index}, sourceIndex = ${sourceIndex}`);
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
		let clueLine = clueResult[1].trim();
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
		    // makeClueElem belongs somewhere else, then can remove Fitler dependency
		    const clueElem = Filter.makeClueElem(clueLine);
		    if (clueElem.prefix) {
			if (clueElem.prefix === Markdown.Prefix.maybe) {
			    debugMsg = 'maybe';
			} else if (clueElem.prefix === Markdown.Prefix.remove) {
			    debugMsg = 'remove';
			}
		    } else {
			debugMsg = 'clue';
		    }
		    if (clueElem.note) debugMsg += ' with note';
		    clueList.push(clueElem);
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
    Debug(`--parse`);
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
    parseDom,
    parseFile
};
