/*
 * log.js
 */

'use strict';

const _ = require('lodash');
const Options = require('./options');//.options;

module.exports = exports = createLog;

function createLog (namespace) {
    if (!namespace) throw new Error('namespace required for now');

    function log (message) {
	log.message(message);
    }

    log.debug = require('debug')(namespace);

    log.message = function (message) {
	if (!Options.quiet) {
	    console.log(message);
	} else {
	    log.debug(message);
	}
    };

    log.info = function (message) {
	//console.log(`log.info, Options = ${_.entries(Options)}`);
	if (Options.verbose) {
	    log.message(message);
	} else {
	    log.debug(message);
	}
    };

    return log;
}
