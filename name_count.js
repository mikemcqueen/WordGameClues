'use strict';

module.exports = NameCount;

// TODO: some way to export a constructor, and a "static" function
// like makeCanonicalName?


function NameCount(name, count, index) {
    var splitList;
    
    if (!name) return;

    this.name  = name;
    this.count = count;
    this.index = index;

    if (!this.count) {
	splitList = name.split(':');
	if (splitList.length > 1) {
	    this.name = splitList[0];
	    this.count = Number(splitList[1]);
	    splitList = splitList[1].split('.');
	    if (splitList.length > 1) {
		this.index = Number(splitList[1]);
	    }
	}
    }
}

//

NameCount.prototype.toString = function() {
    return this.makeCanonicalName(this.name, this.count, this.index);
}

//

NameCount.prototype.makeCanonicalName = function(name, count, index) {
    var s = name;
    if (count) {
	s += ':' + count;
	if (index) {
	    s += '.' + index;
	}
    }
    return s;
}

//

NameCount.prototype.setIndex = function(index) {
    this.index = index;
}

//

NameCount.prototype.log = function() {
    console.log('NameCount: ' + this);
}

//

NameCount.prototype.logList = function(list) {
    var str;

    list.forEach(nc => {
	if (str.length > 0) {
	    str += ',';
	}
	str += nc;
    });
    console.log(str);
}
