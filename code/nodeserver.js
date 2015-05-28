var http = require("http");
var fs = require('fs');

var zerorpc = require("zerorpc");
var zeroRPCPort = 4242;

var nodePort = 3000;
var serverUrl = "127.0.0.1";

var socketPort = 8080;

var pythonServer = new zerorpc.Client();

// Connect to Python server
pythonServer.connect("tcp://" + serverUrl + ":" + zeroRPCPort);

var nodeServer = http.createServer(function(req, res) {
	if(req.url == "/index.html" || req.url == "/") {

		fs.readFile("index.html", function(err, text){
			res.setHeader("Content-Type", "text/html");
			res.end(text);
		});

		return;
	}

	res.setHeader("Content-Type", "text/html");
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

	socket.on('location', function (data) {
		pythonServer.invoke("location", data);
	});
});

console.log("Starting web server at " + serverUrl + ":" + nodePort);
nodeServer.listen(nodePort, serverUrl);