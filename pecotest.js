'use strict';

var Duration    = require('duration');
var Peco        = require('./peco');


var Opt = require('node-getopt')
    .create([
	['c' , 'count=ARG'            , 'use metamorphosis clues'],
	['x' , 'max=ARG'              , 'specify maximum # of components to combine'],
	['p' , 'permutations'         , 'specify maximum # of components to combine']
    ])
    .bindHelp().parseSystem();


function main() {
    var sum;
    var count;
    var max;
    var permFlag;

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
    permFlag = Opt.options['permutations'];

    if (!count && !max) {
	console.log('need count or max');
	return 1;
    }

    console.log('sum: ' + sum + ', count: ' + count + 
		', max: ' + max + ', perm: ' + permFlag);

    
    showAddends(sum, count, max, !permFlag);
}

//

function showAddends(sum, count, max, permFlag) {
    var peco;
    var list;

    peco = new Peco({
	sum:   sum,
	count: count,
	max:   max
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
