'use strict';

var Duration    = require('duration');
var Peco        = require('./peco');


var Opt = require('node-getopt')
    .create([
	['c' , 'count=ARG'            , 'use metamorphosis clues'],
	['e' , 'exclude=ARG+'         , 'exclude #s'],
	['r' , 'require=ARG+'         , 'require #s'],
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

    if (Opt.argv.length < 1) {
	console.log('Usage: node add.js SUM');
	console.log('');
	console.log(Opt);
	return 1;
    }

    // options

    sum = Number(Opt.argv[0]);
    count = Opt.options['count'];
    max = Opt.options['max'];
    require = Opt.options['require'];
    exclude = Opt.options['exclude'];
    permFlag = Opt.options['permutations'];

    if (!count && !max) {
	console.log('need count or max');
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

    showAddends(sum, count, max, permFlag, require, exclude);
}

//

function showAddends(sum, count, max, permFlag, require, exclude) {
    var peco;
    var list;

    peco = new Peco({
	sum:   sum,
	count: count,
	max:   max,
	require: require,
	exclude: exclude
    });

    if (permFlag) {
	list = peco.getPermutations();
    }
    else {
	list = peco.getCombinations();
    }
    list.forEach(elem => {
	console.log(elem);
    });
}


try {
    main();
}
catch(e) {
    console.log(e.stack);
}
