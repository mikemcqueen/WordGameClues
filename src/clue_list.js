'use strict';

var clueListExports = {
    makeNew : makeNew,
    makeFrom: makeFrom
};

module.exports = clueListExports;

//

var Fs            = require('fs');
var Np            = require('named-parameters');

//
//

function makeNew() {
    return assignMethods(Object([]));
}

// makeFrom()
//
// args:
//   filename: filename string
//   optional: optional flag suppresses file error
//   array:    js array

function makeFrom(args) {
    var object;

    args = Np.parse(args).values();
    object = objectFrom(args);

    return assignMethods(Object(object)).init();
}

// objectFrom(args)
//
// args: see makeFrom()

function objectFrom(args) {
    var clueList = [];
    var buffer;
    var key;

    if (args.filename) {
	try {
	    buffer = Fs.readFileSync(args.filename, 'utf8');
	}
	catch (e) {
	    if (!args.optional) {
		throw e;
	    }
	}
	if (buffer) {
	    try {
		clueList = JSON.parse(buffer);
	    }
	    catch (e) {
		//console.log('parsing file: ' + args.filename);
		throw e;
	    }
	}
    }
    else if (args.array) {
	if (!Array.isArray(args.array)) {
	    throw new Error('bad array');
	}
	clueList = args.array;
    }
    else {
	console.log('args:');
	for (key in args) { 
	    console.log('  ' + key + ' : ' + args[key]);
	}
	throw new Error('missing argument');
    }
    
    return clueList;
}

//
//

function assignMethods(list) {
    list.display       = display;
    list.init          = init;
    list.makeKey       = makeKey;
    list.toJSON        = toJSON;
    list.save          = save;

    return list;
}

//
//

function display() {
    var arr = [];
    
    this.forEach(function(clue) {
	arr.push(clue.name);
    });
    console.log(arr.toString());
    
    return this;
}

//
//

function persist(filename) {
    if (Fs.exists(filename)) {
	throw new Error('file already exists: ' + filename);
    }
    Fs.writeFileSync(filename, this.toJSON());
}

//

function toJSON() {
    var result = '[\n';
    var first = true;

    this.forEach(clue => {
	if (!first) {
	    result += ',\n';
	}
	else { 
	    first = false;
	}
	result += "  " + clueToJSON(clue);
    });
    if (!first) {
	result += '\n';
    }
    result += ']';

    return result;
}

//

function save(filename) {
    Fs.writeFileSync(filename, this.toJSON(), { encoding: 'utf8' });
}

//

function clueToJSON(clue) {
    var s;

    s = '{';

    if (clue.name) {
	s += ' "name": "'  + clue.name  + '", ' + format2(clue.name, 25);
    }
    s += ' "src": "' + clue.src + '"';

    if (clue.note) {
	s+= ', "note" : "' + clue.note + '"';
    }
    if (clue.x) {
	s+= ', "x" : ' + clue.x;
    }
    if (clue.ignore) {
	s+= ', "ignore" : ' + clue.ignore;
    }
    else if (clue.skip) {
	s+= ', "skip" : ' + clue.skip;
    }
    s += ' }';

    return s;
}

//

function format2(text, span)
{
    var result = "";
    for (var len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}

//
//

function init() {
    return this;
}

//
//

function makeKey() {
    var keyArray = [];
    this.forEach(clue => {
	keyArray.push(clue.name);
    });
    keyArray.sort();

    return keyArray.toString();
}

