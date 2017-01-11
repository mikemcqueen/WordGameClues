'use strict';

// export a singleton

module.exports = exports = new Util();

//
//

function Util() {
}

// Given a number N, return all combinations of addends
// that have a sum of N.

Util.prototype.getAllAddends = function(sum, max) {
    var addends = [];

    switch (sum) {
    case 2:
	addends[0] = [ 1, 1 ];
	break;

    case 3:
	addends[0] = [ 2, 1 ];
	if (max > 2) {
	    addends[1] = [ 1, 1, 1 ];
	}
	break;

    case 4:
	addends[0] = [ 1, 3 ];
	addends[1] = [ 2, 2 ];
	if (max > 2) {
	    addends[2] = [ 1, 1, 2];
	}
	if (max > 3) {
	    addends[3] = [ 1, 1, 1, 1 ];
	}
	break;

    default:
	throw new Error('invalid sum');
    }

    return addends;
}

