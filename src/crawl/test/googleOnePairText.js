//
// googleOnePairText.js
//

'use strict';

var fs = require('fs'),
    google = require('google'),
    Delay = require('../../util/delay');

describe('google one word pair, show result in text', function() {
    var counter = 0;

    this.timeout(75000);

    it('google', function(done) {
	var wiki = 'site:wikipedia.org';

	google('pardee brown' + ' ' + wiki, function (err, res) {
	    var waitForNext;
	    var delay;

	    if (err) console.error(err)
 
	    res.links.forEach(function(link) {
		console.log(link.title + ' - ' + link.href)
		console.log(link.description + '\n')
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
		done();
	    }
	});
    });
});
