//
// GOOGLERESULT.JS
//

'use strict';

//

let Google     = require('google');
let Ms         = require('ms');
let PrettyMs   = require('pretty-ms');
let Between    = require('../util/between');

//

function get(text, pages, cb) {
    let resultList = [];
    let count = 0;
    Google(text, function (err, result) {
	if (err) return cb(err);
	resultList.push(...result.links.map(link => Object({
	    title:   link.title,
	    url:     link.href,
	    summary: link.description
	})));
	count += 1;
	if (count < pages && result.next) {
	    let msDelay = Between(Ms('30s'), Ms('60s'));
	    console.log(`Delaying ${PrettyMs(msDelay)} for next page of results...`);
	    setTimeout(result.next, msDelay);
	}
	else {
	    return cb(null, resultList);
	}
    });
}

//

module.exports = {
    get            : get
};

