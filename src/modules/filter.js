//
// filter.js
//

'use strict';

const _            = require('lodash');
const ClueManager  = require('../clue-manager');
const Clues        = require('../clue-types');
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

const URL_PREFIX       = 'http';
const KNOWN_CLUES_URL  = 'https://known.clues';

//

function optLog (options, message) {
    Expect(options).is.an.Object();
    if (options.verbose) {
	console.log(message);
    } else {
 	Debug(message);
    }
}

//

function isUrl (line) {
    return _.startsWith(line, URL_PREFIX);
}

//

function parseLines (lines, options = {}) {
    let filterList = [];
    let urlList;
    let clueList;
    for (let line of lines) {
	line = line.trim();
	Debug(line);
	if (Markdown.hasSourcePrefix(line)) {
	    clueList = undefined;
	    urlList = [];
	    const [source, suffix] = Markdown.getSuffix(line);
	    Debug(`source: ${source} suffix: ${suffix}`);
	    filterList.push({ source, suffix,  urls: urlList });
	} else if (isUrl(line)) {
	    clueList = [];
	    const [url, suffix] = Markdown.getSuffix(line);
	    Debug(`url: ${url} suffix: ${suffix}`);
	    urlList.push({ url, suffix, clues: clueList });
	} else if (!_.isEmpty(line)) {
	    Debug(`clue: ${line}`);
	    // clue, known, or maybe
	    // currently requires URL, but i suppose could eliminate that requirement with some work.
	    Expect(urlList).is.an.Array().and.not.empty();
	    Expect(clueList).is.an.Array();
	    clueList.push(makeClueElem(line));
	}
    }
    return filterList;
}

//

function parseFile (filename, options = {}) {
    let readLines = new Readlines(filename);
    let lines = [];
    let line;
    while ((line = readLines.next()) !== false) {
	lines.push(line.toString().trim());
    }
    return parseLines(lines, options);
}

// rename parseFileSync?

function old_parseFile (filename, options = {}) {
    let readLines = new Readlines(filename);
    let line;
    let resultList = [];
    let urlList;
    let clueList;
    while ((line = readLines.next()) !== false) {
	line = line.toString().trim();
	if (_.isEmpty(line)) continue;
	Debug(line);
	if (Markdown.hasSourcePrefix(line)) {
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
	    Expect(urlList).is.an.Array().and.not.empty();
	    Expect(clueList).is.an.Array();
	    clueList.push(makeClueElem(line));
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
    if (0) Debug(`mapA: ${Stringify(mapA)}`);
    return listB.filter(elemB => {
	const inA = _.has(mapA, elemB.source);
	Debug(`${elemB.source} ${inA ? 'is' : 'not'} in mapA`);
	return !inA;
    });
}

// filter rejects
// listFiltered has some potential for sanity checking. i could load the _filtered.json
// file and confirm that all of the remaining urls, per clue source, are not rejected

function filterRejectUrls (listToFilter, listFiltered, options)  {
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

function filterRejectSources (listToFilter, options)  {
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

// belongs in modules/clue.js

function getClueText (clue, options) {
    // used to be options.all, but i think that i was wrong
    //if (options.remove && (clue.prefix === Markdown.Prefix.remove)) {
    //return undefined;
    //}
    let text = clue.clue;
    if (!text) {
	console.log(Stringify(clue));
	Expect(text).is.ok();
    }
    if (clue.prefix) text = clue.prefix + text;
    if (clue.note)   text += `,${clue.note}`;
    return text;
}

function makeClueElem (clueLine) {
    // TODO: 'note', 'need' markdowns
    let [line, prefix] = Markdown.getPrefix(clueLine);
    if (prefix) {
	clueLine = line;
    }
    let note;
    //TODO: Markdown.hasSuffix(line, []) // allow truly any suffix
    let commaIndex = clueLine.indexOf(',');
    if (commaIndex > -1) {
	// TODO: process note for need
	note = clueLine.slice(commaIndex + 1, clueLine.length);
	clueLine = clueLine.slice(0, commaIndex);
    }
    return { clue: clueLine, prefix, note };
}

// options:
//  fd       file descriptor.  required
//  json     json format, else filter format
//  removed  if (false) don't save removed clues

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
		if (urlElem.suffix) url = `${url},${urlElem.suffix}`;
		await My.writeln(dest, url);
		if (!_.isObject(urlElem)) continue;
		for (const clueElem of urlElem.clues || []) {
		    const text = getClueText(clueElem, options);
		    if (!text) {
			Debug(`excluding removed clue: ${source} | ${clueElem.clue}`);
			continue;
		    }
		    await My.writeln(dest, text);
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

function save (filterList, path) {
    return Fs.open(path, 'w')
	.then(fd => {
	    return dumpList(filterList, { fd });
	}).then(fd => Fs.close(fd))
	.then(_ => path);
}

//

function getUpdateFilePath(noteName, options) {
    return `${Clues.getDirectory(Clues.getByOptions(options))}/updates/${noteName}`;
}

//

async function saveAddCommit (noteName, filterList, options) {
    const path = getUpdateFilePath(noteName, options);
    if (options.dry_run) return path;
    Debug(`saving ${noteName} to: ${path}`);
    return save(filterList, path)
	.then(_ => {
	    // TODO MAYBE: options.wait
	    // no return = no await completion = OK
	    return (options.production) ? My.gitAddCommit(path, 'parsed live note') : undefined;
	}).then(_ => path); // return the path we saved to
}

//

function getRemovedClues (filterList) {
    let removedClueMap = new Map();
    for (let srcElem of filterList) {
	for (let urlElem of srcElem.urls || []) {
	    for (let clueElem of urlElem.clues || []) {
		if (clueElem.prefix === Markdown.Prefix.remove) {
		    if (!removedClueMap.has(srcElem.source)) {
			removedClueMap.set(srcElem.source, new Set());
		    }
		    removedClueMap.get(srcElem.source).add(clueElem.clue);
		    Debug(`getRemovedClue: ${srcElem.source}` +
			  `[${removedClueMap.get(srcElem.source).size}] = ${clueElem.clue}`);
		}
	    }
	}
    }
    return removedClueMap;
}

// don't pass command line options here. just specific
// options for this function
//
// options
//  all - include all clues (even removed)
//
function getClues (urlList, options = {}) {
    Expect(urlList).is.an.Array();  // require this
    let clues = new Set();
    for (let urlElem of urlList) {
	for (let clueElem of urlElem.clues) {
	    Expect(clueElem.clue).is.ok();
	    // NOTE: test this
	    //if ((clueElem.prefix === Markdown.Prefix.remove) && !options.all) continue;
	    clues.add(clueElem.clue);
	}
    }
    return clues;
}

//

function addKnownClues (filterList) {
    let added = 0;
    for (let srcElem of filterList) {
	let allClues = getClues(srcElem.urls);
	// fancy way of saying slice(1,length);
	let [source, prefix] = Markdown.getPrefix(srcElem.source, Markdown.Prefix.source);
	let knownClues = ClueManager.getKnownClueNames(source) // or just KnownClues() eventually
		.filter(name => !allClues.has(name)) // filter out entries in allClues
	        // map to 'clueElem' object. kinda dumb. could just use clue object except
           	// we're expecting "name" to be "clue" here due to poor decision in noteParse.
		.map(name => { 
		    return {
			clue: name
//			note: clue.note
		    };
		});
	Expect(knownClues).is.ok();
	if (!_.isEmpty(knownClues)) {
	    Debug(`${srcElem.source} known: ${knownClues}`); // todo: map:clue->clue.name
	    const moreCluesUrl = _.find(srcElem.urls, { url: KNOWN_CLUES_URL });
	    if (moreCluesUrl) {
		// add known clues to existing more.clues URL elem
		moreCluesUrl.clues.push(...knownClues);
	    } else {
		// create new more.clues URL elem with known clues
		srcElem.urls.push({
		    url:   KNOWN_CLUES_URL,
		    clues: knownClues
		});
	    }
	    added += knownClues.length;
	}
    }
   return added;
}

// remove clues from filterList that are in removedClueMap
// removedClueMap is a Map<@sourceCsv, Set<clueName>>

function removeRemovedClues (filterList, removedClueMap, options = {}) {
    Debug('removeRemovedClues');
    let removed = 0;
    for (const srcElem of filterList) {
	if (!removedClueMap.has(srcElem.source)) continue;
	const removedSet = removedClueMap.get(srcElem.source);
	// for each urlElem of this source
	for (const urlElem of srcElem.urls) {
	    // filter out any removed clues
	    // lazy Seq here faster? would be neat to test. with Immutable & 1million iters.
	    urlElem.clues = urlElem.clues.filter(clueElem => {
		if (removedSet.has(clueElem.clue)) {
		    Expect(Markdown.Prefix.remove === clueElem.prefix);
		    removed += 1;
		    return false;
		}
		return true;
	    });
	}
    }
    Debug(`removed: ${removed}`);
    return removed;
}

// remove clues from removedClueMap that are known by ClueManager (i.e. they aren't removed)
// removedClueMap is a Map<@sourceCsv, Set<clueName>>
// why is this in Filter? it totally doesn't belong here. 
// clue-manager or clue-utils or removed-clues maybe

function removeKnownClues (removedClueMap, options = {}) {
    Debug('removeKnownClues');
    let removed = 0;
    for (let source of removedClueMap.keys()) {
	let nameCsv = Markdown.hasSourcePrefix(source)
		? source.slice(1, source.length) : source;
	const nameList = nameCsv.split(',').sort();
	const result = ClueManager.getCountListArrays(nameCsv, { remove: true });
	if (!result || _.isEmpty(result.known)) {
	    continue;
	}

	// copy (potentially) multiple lists of clue names to a set
	let knownSet = new Set();
	for (let known of result.known) {
	    known.nameList.forEach(name => knownSet.add(name));
	}

	// for each clue word in removed set
	let removedSet = removedClueMap.get(source);
	for (let name of removedSet.keys()) {
	    if (knownSet.has(name)) {
		removedSet.delete(name); // safe when iterating
		Debug(`removed ${source} -> ${name}`);
		removed += 1;
	    }
	}
	if (_.isEmpty(removedSet)) {
	    removedClueMap.delete(source); // safe when iterating
	}
    }
    Debug(`removed: ${removed}`);
    return removed;
}

//

module.exports = {
    addKnownClues,
    count,
    diff,
    dumpList,
    filterRejectSources,
    filterRejectUrls,
    getClueText,
    getRemovedClues,
    getUpdateFilePath,
    makeClueElem,
    parseFile,
    parseLines,
    removeRemovedClues,
    removeKnownClues,
    save,
    saveAddCommit,

    KNOWN_CLUES_URL
};
