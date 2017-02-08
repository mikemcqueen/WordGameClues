'use strict';

var google = require('google');
var Delay = require('../util/delay');

var GoogleResultExports = {
    get : get
};

module.exports = GoogleResultExports;

function get(text, callback) {
    var resultList = [];
    var counter = 0;

    google(text, function (err, res) {
	var waitForNext;
	var delay;
	
	if (err) {
	    callback(err);
	    return;
	}
 
	res.links.forEach(function(link) {
	    resultList.push({
		title:   link.title,
		url:     link.href,
		summary: link.description
	    });
	});
	    
	waitForNext = false;
	if (counter < 1) {
	    counter += 1;
	    if (res.next) {
		delay = Delay.between(30, 60, Delay.Seconds);
		console.log('Delaying ' + (delay / 1000) + ' seconds for next page of results...');
		setTimeout(res.next, delay);
		waitForNext = true;
	    }
	}
	if (!waitForNext) {
	    callback(null, resultList);
	    return;
	}
    });
}
