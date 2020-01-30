//
// oauth.js
//
// code copied from stackoverflow, or evernote forums
//

'use strict';

const Evernote = require('evernote');
const http = require("http");
const url = require("url");
const Config = require('../../data/evernote-config.json');

//
const port = 8888;

let global = {};
global.oauthToken = '';
global.oauthSecret = '';

//

function getOauthVerifier (url) {
    let regex = new RegExp("[\\?&]oauth_verifier=([^&]*)");
    let results = regex.exec(url);
    
    return results === null ? "" : decodeURIComponent(results[1].replace(/\+/g, " "));
}

//

let handler = function (request, response) {
    let params = url.parse(request.url);
    let pathname = params.pathname;
    console.log("Request for " + pathname + " received.");
    
    let client = new Evernote.Client ({
        consumerKey:    Config.consumer.key,
        consumerSecret: Config.consumer.secret,
        sandbox:        false
    });
    
    if (pathname == "/"){
        let callbackUrl = 'http://localhost:8888/oauth';
        
        client.getRequestToken(callbackUrl, function(err, oauthToken, oauthSecret, results){
            if(err) {
                console.log(err);
            }
            else {
                global.oauthToken = oauthToken;
                global.oauthSecret = oauthSecret;
                console.log("set oauth token and secret");
                let authorizeUrl = client.getAuthorizeUrl(oauthToken);
                console.log(`authorizedUrl: ${authorizeUrl}`);
                response.writeHead(200, {"Content-Type":"text/html"});
                response.write("Please <a href=\""+authorizeUrl+"\">click here</a> to authorize the application");
                response.end();
            }
        });
    }
    else if (pathname == "/oauth"){
        let verifier = getOauthVerifier(params.search);
        console.log(`verifier: ${verifier}`);
        client.getAccessToken(
            global.oauthToken, 
            global.oauthSecret, 
            verifier,
            function(error, oauthAccessToken, oauthAccessTokenSecret, results) {
                if(error) {
                    console.log("error\n\n\n");
                    console.log(error);
                }
                else {
                    response.writeHead(200, {"Content-Type":"text/html"});
                    response.write(oauthAccessToken);
                    response.end();
                }   
            }
        );
    }
    else {
        response.writeHead(200, {"Content-Type":"text/html"});
        response.write("not a valid URL <a href=\"/\"> GO HOME </a>");
        response.end();
    }
};

//

console.log(`Starting server on port ${port}`);
http.createServer(handler).listen(8888);
