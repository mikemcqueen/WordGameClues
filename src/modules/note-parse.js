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

//

function parse (text) {
    const wordExpr = />@([^<]+)\</g;
    let wordsList = [];
    while (true) {
	const result = wordExpr.exec(text);
	if (!result) break;
	// result[1] = 1st capture group
	const [_, line] = My.hasCommaSuffix(result[1], RejectSuffix);
	wordsList.push(line); 
    }
    return wordsList;
}

//

function parseFile (filename) {
    return Fs.readFile(filename)
	.then(content => {
	    return parse(content.toString());
	});
}

//

module.exports = {
    parse,
    parseFile
};
