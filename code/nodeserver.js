var express = require("express");
var path = require("path");

var zerorpc = require("zerorpc");
var zeroRPCPort = 4242;

var nodePort = 3000;
var serverUrl = "127.0.0.1";

var socketPort = 8080;

var pythonServer = new zerorpc.Client();

// Initialize node.js server
var nodeServer = express();

nodeServer.use('/', express.static(__dirname))

// Connect to Python server
pythonServer.connect("tcp://" + serverUrl + ":" + zeroRPCPort);

nodeServer.get('/', function(req, res) {
	res.sendFile(path.join(__dirname + '/index.html'));
});

var socketIo = require('socket.io').listen(socketPort); 

socketIo.sockets.on('connection', function (socket) {
	// Wait for the event raised by the web client
	socket.on('event', function (data) {  

		// Calls the method on the Python server
		pythonServer.invoke("hello", "Tung", function(error, reply, streaming) {
			if(error) {
				console.log("ERROR: ", error);
			}

			console.log("Received from ZeroRPC: " + reply);

			// Emit reply flag to web client
			socket.emit('reply', reply);
		});
	});

	socket.on('classify', function (data) {
		pythonServer.invoke('classify', data.url, data.patchDim, 
			function(error, reply, streaming) {
				if(error) {
					console.log("ERROR: ", error);
				}
				else {
					console.log("ok!");
				}
				console.log('done')
				//socket.emit('classify_reply', reply);
			});
	});

	socket.on('location', function (data) {
		pythonServer.invoke("location", data);
	});
});

console.log("Starting web server at " + serverUrl + ":" + nodePort);
nodeServer.listen(nodePort);