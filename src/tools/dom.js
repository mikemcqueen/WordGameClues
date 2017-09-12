/*
 * dom.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('note');
const DOMParser        = require('xmldom').DOMParser;
const Fs               = require('fs');
const Stringify        = require('stringify-object');

//

function spaces (count) {
    return ' '.repeat(count * 2);
}

// 

function show (node, indent, suffix) {
    let text = '';
    if (node.nodeName === '#text') text = node.textContent;
    console.log(`${spaces(indent)}${node.nodeName} ${text} ${suffix || ''}`);
    
    if (node.childNodes) {
	//console.log(`childNodes ${_.isArray(node.childNodes) ? "array" : "object"}`);
	Array.prototype.forEach.call(node.childNodes, child => {
	    show (child, indent + 1); //, 'child');
	});
    }
//    if (node.nextSibling) show(node.nextSibling, indent, 'sib');
}

// 
    
async function main () {
    const filename = process.argv[2] || './tmp/smallnote.enml';

    let content = Fs.readFileSync(filename);
    const doc = new DOMParser().parseFromString(content.toString());

    let node = doc.documentElement;
    show(node, 0);
}

//

main().catch(err => {
    console.log(err, err.stack);
});

