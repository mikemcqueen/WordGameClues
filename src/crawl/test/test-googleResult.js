//
// googleOnePairText.js
//

'use strict';

var googleResult = require('../googleResult');

describe('google one word pair, show result', function() {
    var counter = 0;
    var wiki = 'site:en.wikipedia.org';

    this.timeout(75000);

    it('test one pair', function(done) {
...	googleResult.get('' + ' ' + wiki, function(err, res) {
	    if (err) {
		console.log(err);
	    }
	    else {
		console.log(res);
	    }
	    done();
	});
    });
});
