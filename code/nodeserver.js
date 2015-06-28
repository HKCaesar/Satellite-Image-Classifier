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

nodeServer.get('/', function (req, res) {
  res.sendFile(path.join(__dirname + '/index.html'));
});

var socketIo = require('socket.io').listen(socketPort);

socketIo.sockets.on('connection', function (socket) {
  socket.on('classify_small', function (data) {
    pythonServer.invoke('classify_small', data.url, data.patchDim,
      function (error, reply, streaming) {
        if (error) {
          console.log("ERROR: ", error);
        } else {
          console.log("ok!");
        }
        console.log('done small')
        socket.emit('classify_reply', reply);
      });
  });

  socket.on('classify_large', function (data) {
    pythonServer.invoke('classify_large', data.url, data.patchDim,
      function (error, reply, streaming) {
        if (error) {
          console.log("ERROR: ", error);
        } else {
          console.log("ok!");
        }
        console.log('done large')
        socket.emit('classify_reply', reply);
      });
  });

  socket.on('location', function (data) {
    pythonServer.invoke("location", data);
  });
});

console.log("Starting web server at " + serverUrl + ":" + nodePort);
nodeServer.listen(nodePort);