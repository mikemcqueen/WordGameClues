//
// filter.js
//

'use strict';

const _            = require('lodash');
const Debug        = require('debug')('filter');
const Duration     = require('duration');
const Expect       = require('should/as-function');
const Fs           = require('fs-extra');
const My           = require('./util');
const Path         = require('path');
const Readlines    = require('n-readlines');
const Stringify    = require('stringify-object');

//

const SRC_PREFIX      = '@';
const MAYBE_PREFIX    = ':';
const KNOWN_PREFIX    = '#';

const URL_PREFIX      = 'http';

//

function hasPrefix (line, prefix) {
    return line[0] === prefix;
}

//

function isSource (line) {
    return hasPrefix(line, SRC_PREFIX);
}

//

function isUrl (line) {
    return _.startsWith(line, URL_PREFIX);
}

//

function isMaybe (line) {
    return hasPrefix(line, MAYBE_PREFIX);
}

//

function isKnown (line) {
    return hasPrefix(line, KNOWN_PREFIX);
}

//

function parseFile (filename, options) {
    let readLines = new Readlines(filename);
    let line;
    let resultList = [];
    let urlList;
    let clueList;
    while ((line = readLines.next()) !== false) {
	line = line.toString().trim();
	if (_.isEmpty(line)) continue;
	Debug(line);
	if (isSource(line)) {
	    clueList = undefined;
	    urlList = [];
	    const sourceElement = options.urls ? { source: line, urls: urlList } : line;
	    resultList.push(sourceElement);
	} else if (isUrl(line)) {
	    clueList = [];
	    const urlElement = (options.clues) ? { url: line, clues: clueList } : line;
	    urlList.push(urlElement);
	} else {
	    // clue, known, or maybe
	    // currently requires URL, but i suppose could eliminate that requirement with some work.
	    Expect(clueList).is.an.Array();
	    clueList.push(line);
	}
    }
    return resultList;
}

//

function makeSourceMap (list) {
    let map = {};
    for (let elem of list) {
	let source = elem.source;
	Expect(source).is.ok();
	Expect(_.has(map, source)).is.false();
	map[source] = elem.urls;
    }
    return map;
}

//
// Return elements in listB that are not in listA.
//

function diff (listA, listB) {
    Expect(listA).is.an.Array();
    Expect(listB).is.an.Array();
    const mapA = makeSourceMap(listA);
    Debug(`mapA: ${Stringify(mapA)}`);
    let resultList = [];
    for (const elemB of listB) {
	if (_.has(mapA, elemB.source)) continue;
	// FYI: not a copy
	Debug(`${elemB.source} is not in mapA`);
	resultList.push(elemB);
    }
    return resultList;
}

//

function dumpList (list, options) {
    for (const [index, sourceElem] of list.entries()) {
	if (index > 0) console.log();
	if (options.json) {
	    const prefix = (index === 0) ? '[' : ', ';
	    const suffix = (index === list.length - 1) ? ']' : '';
	    console.log(`${prefix}${Stringify(sourceElem)}${suffix}`);
	}
	else {
	    const source = sourceElem.source || sourceElem;
	    console.log(source);
	    if (!_.isObject(sourceElem)) continue;
	    for (const urlElem of sourceElem.urls) {
		const url = urlElem.url || urlElem;
		console.log(url);
		if (!_.isObject(urlElem)) continue;
		for (const clue of urlElem.clues) {
		    console.log(clue);
		}
	    }
	}
    }
}

//

module.exports = {
    diff,
    dumpList,
    isKnown,
    isMaybe,
    isSource,
    parseFile,
    SRC_PREFIX,
    MAYBE_PREFIX,
    KNOWN_PREFIX
};
