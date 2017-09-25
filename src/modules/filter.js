//
// filter.js
//

'use strict';

const _            = require('lodash');
const Debug        = require('debug')('filter');
const Duration     = require('duration');
const Expect       = require('should/as-function');
const Fs           = require('fs-extra');
const Markdown     = require('./markdown');
const My           = require('./util');
const Path         = require('path');
const Readlines    = require('n-readlines');
const Stringify    = require('stringify-object');

//

const URL_PREFIX      = 'http';

//

function isUrl (line) {
    return _.startsWith(line, URL_PREFIX);
}

// rename parseFileSync?

function parseFile (filename, options = {}) {
    let readLines = new Readlines(filename);
    let line;
    let resultList = [];
    let urlList;
    let clueList;
    while ((line = readLines.next()) !== false) {
	line = line.toString().trim();
	if (_.isEmpty(line)) continue;
	Debug(line);
	if (Markdown.isSource(line)) {
	    Debug(`source: ${line}`);
	    clueList = undefined;
	    urlList = [];
	    const sourceElement = options.urls ? { source: line, urls: urlList } : line;
	    resultList.push(sourceElement);
	} else if (isUrl(line)) {
	    Debug(`url: ${line}`);
	    clueList = [];
	    const urlElement = (options.clues) ? { url: line, clues: clueList } : line;
	    urlList.push(urlElement);
	} else {
	    Debug(`clue: ${line}`);
	    // clue, known, or maybe
	    // currently requires URL, but i suppose could eliminate that requirement with some work.
	    Expect(urlList).is.not.empty();
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
    return listB.filter(elemB => {
	const inA = _.has(mapA, elemB.source);
	Debug(`${elemB.source}${inA ? 'is' : 'not'} in mapA`);
	return !inA;
    });
}

// listFiltered has some potential for sanity checking. i could load the _filtered.json
// file and confirm that all of the remaining urls, per clue source, are not rejected

function filterUrls (listToFilter, listFiltered, options)  {
    if (options.noFilterUrls) return [listToFilter, 0];
    //const map = makeSourceMap(listFiltered);
    let filterCount = 0;
    let list = listToFilter.filter(sourceElem => {
	Expect(sourceElem).is.an.Object(); 
	//const filteredUrls = map[sourceElem.source].urls;
	sourceElem.urls = sourceElem.urls.filter(urlElem => {
	    const reject = (urlElem.suffix === Markdown.Suffix.reject);
	    if (reject) {
		Debug(`reject: ${urlElem.url}`);
		filterCount += 1;
	    }
	    return !reject;
	});
	// keep any source with a url, or marked as a clue
	// (or valid suffix, but not reject, eventually)
	return !_.isEmpty(sourceElem.urls) || (sourceElem.suffix === Markdown.Suffix.clue);
    });
    return [list, filterCount];
}

//

function filterSources (listToFilter, options)  {
    let filterCount = 0;
    let filteredList = listToFilter.filter(sourceElem => {
	Expect(sourceElem).is.an.Object(); 
	const reject = (sourceElem.suffix === Markdown.Suffix.reject);
	if (reject) {
	    Debug(`reject source: ${sourceElem.source}`);
	    filterCount += 1;
	}
	return !reject;
    });
    return [filteredList, filterCount];
}

//

async function dumpList (list, options) {
    Expect(options.fd).is.a.Number(); // for now. no other use case yet.
    const dest = options.fd;
    for (const [index, sourceElem] of list.entries()) {
	if (index > 0) { // empty line between entries
	    await My.writeln(dest, '');
	}
	if (options.json) { // JSON format
	    const prefix = (index === 0) ? '[' : ', ';
	    const suffix = (index === list.length - 1) ? ']' : '';
	    await My.writeln(dest, `${prefix}${Stringify(sourceElem)}${suffix}]`);
	}
	else { // filter format
	    let source = sourceElem.source || sourceElem;
	    if (sourceElem.suffix) source += `,${sourceElem.suffix}`;
	    await My.writeln(dest, `${source}`);
	    if (!_.isObject(sourceElem)) continue;
	    for (const urlElem of sourceElem.urls || []) {
		let url = urlElem.url || urlElem;
		if (urlElem.suffix) url += `,${urlElem.suffix}`;
		await My.writeln(dest, url);
		if (!_.isObject(urlElem)) continue;
		for (const clueElem of urlElem.clues || []) {
		    let clue = clueElem.clue;
		    if (clueElem.prefix) clue = `${clueElem.prefix}${clue}`;
		    await My.writeln(dest, clue);
		}
	    }
	}
    }
    /*
    if (!options.json) { // empty line at end of filter format
    	await My.writeln(dest, '').then(result => {
	    if (_.isString(dest)) dest = result;
	});
    }
     */
    return dest;
}

//

function count (list) {
    let sources = 0;
    let urls = 0;
    let clues = 0;
    
    for (let src of list) {
	sources += 1;
	for (let url of src.urls || []) {
	    urls += 1;
	    for (let clue of url.clues || []) {
		clues += 1;
	    }
	}
    }
    return { sources, urls, clues };
}

//

function getRemovedClues (list) {
    let removedClues = new Map();
    for (let srcElem of list) {
	for (let urlElem of srcElem.urls || []) {
	    for (let clueElem of urlElem.clues || []) {
		if (clueElem.prefix === Markdown.Prefix.remove) {
		    if (!_.has(removedClues, srcElem.source)) {
			removedClues.set(srcElem.source, new Set());
		    }
		    removedClues.get(srcElem.source).add(clueElem.clue);
		}
	    }
	}
    }
    return removedClues;
}

//

module.exports = {
    count,
    diff,
    dumpList,
    filterSources,
    filterUrls,
    getRemovedClues,
    parseFile
};
