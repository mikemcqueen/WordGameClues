'use strict';

var _           = require('lodash');
var Duration    = require('duration');
var Peco        = require('./peco');


var Opt = require('node-getopt')
    .create([
	['c' , 'count=ARG'            , 'primary clue count'],
	['e' , 'exclude=ARG+'         , 'exclude #s'],
	['l',  'list=ARG+',             'test listArray, must specify at least tw' ],
	['r' , 'require=ARG+'         , 'require #s'],
	['v' , 'verbose'              , 'turn on peco logging'],
	['x' , 'max=ARG'              , 'specify maximum # of components to combine'],
	['p' , 'permutations'         , 'permutations flag' ],
    ])
    .bindHelp().parseSystem();


function main() {
    var sum;
    var count;
    var max;
    var permFlag;
    var require;
    var exclude;
    var list;
    var listArray;
    var verbose;

    if (Opt.argv.length < 1) {
	console.log('Usage: node add.js SUM');
	console.log('');
	console.log(Opt);
	return 1;
    }

    // options

    sum = _.toNumber(Opt.argv[0]);
    count = _.toNumber(Opt.options['count']);
    max = _.toNumber(Opt.options['max']);
    require = Opt.options['require'];
    exclude = Opt.options['exclude'];
    permFlag = Opt.options['permutations'];
    list = Opt.options['list'];
    verbose = Opt.options['verbose'];
    
    if (verbose) {
	Peco.logging = true;
    }

    if (!list && !count && !max) {
	console.log('need count or max');
	return 1;
    }
    if (list && !count) {
	console.log('need count');
	return 1;
    }

    if (list && (list.length < 2)) {
	console.log('need to specify at least two -l lists');
	return 1;
    }


    console.log('sum: ' + sum + ', count: ' + count + 
		', max: ' + max + ', perm: ' + permFlag +
		', require: ' + require +
		', exclude: ' + exclude);

    
    if (require) {
	require.forEach((num, index) => { require[index] = Number(num); });
    }
    if (exclude) {
	exclude.forEach((num, index) => { exclude[index] = Number(num); });
    }
    
    if (list) {
	console.log('list: ' + list);
	listArray = makeListArray(list);
//	listArray = list;
	console.log('listArray: ' + listArray);
    }
    showAddends(listArray, sum, count, max, permFlag, require, exclude);
}

//

function makeListArray(list) {
    var listArray;
    listArray = [];
    list.forEach(str => {
	console.log('list: ' + str);
	var countList = str.split(',');
	countList.forEach((count,index) => {
	    countList[index] = Number(count);
	});
	listArray.push(countList);
    });
    return listArray;
}

//

function showAddends(listArray, sum, count, max, permFlag, require, exclude) {
    var peco;
    var result;

    peco = Peco.makeNew(listArray ? {
	listArray: listArray,
	max:       count
    } : {
	sum:   sum,
	count: count,
	max:   max,
	require: require,
	exclude: exclude
    });

    if (permFlag) {
	resultt = peco.getPermutations();
    }
    else {
	result = peco.getCombinations();
    }
    if (result) {
	result.forEach(elem => {
	    console.log(elem);
	});
    }
    else {
	console.log('no results');
    }
}


try {
    main();
}
catch(e) {
    console.log(e.stack);
}
