
'use strict';

var //Q = require('q'),    // promise package
    //ch = require('chromedriver'),
    fs = require('fs'),
//    test = require('selenium-webdriver/testing'),
//    webdriver = require('selenium-webdriver'),
    google = require('google');

describe('Integration test', function() {
    var driver;

    // for demo purposes
    this.timeout(15000);

    //
    // before tests start
    //

    // build the web driver, pointing HTTPS to local proxy server
    /*
    before(function() {
	driver = new webdriver.Builder()
	    .withCapabilities(webdriver.Capabilities.chrome())
	    .setProxy(wdProxy.manual({https: 'localhost:8080'}))
	    .build();
    });
    //
    // after tests finish
    //

    // quit driver, close server
    after.skip(function() {
	driver.quit() // comment out this line to leave browser window up after test
	console.log('>>driver.quit');
    });
    */

    //
    // the actual tests (in this case, test steps) here
    //

    /*
    it.skip('navigate to https://www.google.com', function(done) {
	this.slow(10000); // > 10s will show up red
	driver.get('https://www.google.com').then(function() {
	    console.log('>>navigate complete')
	    done();
	});
    });

    it.skip('wait a sec', function() {
	return driver.sleep(3000);
    });

    it.skip('enter search term, click search', function(done) {
	var wiki = 'site:wikipedia.org';
	driver.findElement(webdriver.By.name('q')).sendKeys('blue cross' + ' ' + wiki);
	driver.findElement(webdriver.By.name('btnG')).click();
	done();
    });
    */


    it('google', function(done) {
	var wiki = 'site:wikipedia.org';

	google('blue cross' + ' ' + wiki, function (err, res) {
	    if (err) console.error(err)
 
	    for (var i = 0; i < res.links.length; ++i) {
		var link = res.links[i];
		console.log(link.title + ' - ' + link.href)
		console.log(link.description + "\n")
	    }
	    
	    if (nextCounter < 4) {
		nextCounter += 1;
		if (res.next) res.next();
	    }
	    done();
	});
    });
});
