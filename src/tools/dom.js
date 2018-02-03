/*
 * dom.js
 */

'use strict';

//

const _                = require('lodash');
const Debug            = require('debug')('dom');
const DomParser        = require('xmldom').DOMParser;
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
    if (node.nodeName === 'a') {
        text = node.attributes.getNamedItem('href');
    }
    console.log(`${spaces(indent)}${node.nodeName} ${text} ${suffix || ''}`);
    
    if (node.childNodes) {
        Array.prototype.forEach.call(node.childNodes, child => {
            show (child, indent + 1);
        });
    }
}

// 
    
async function main () {
    const filename = process.argv[2] || './tmp/smallnote.enml';

    let content = Fs.readFileSync(filename);
    const doc = new DomParser().parseFromString(content.toString());

    let node = doc.documentElement;
    show(node, 0);
}

//

main().catch(err => {
    console.log(err, err.stack);
});

