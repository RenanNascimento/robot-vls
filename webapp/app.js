/* Express */

const express = require('express');
const app = express()
const http = require('http').createServer(app);

app.use(express.static('public'));
app.set('view engine', 'ejs')

app.get('/', function (req, res) {
  res.render('index');
})

http.listen(3000);

/* TCP Socket Server */

const net = require('net');

var server = net.createServer();  
server.on('connection', handleConnection);

server.listen(8080, 'localhost');

timestamp = 0

function handleConnection(conn) {
  var remoteAddress = conn.remoteAddress + ':' + conn.remotePort;
  console.log('new client connection from %s', remoteAddress);

  conn.on('data', onConnData);
  conn.once('close', onConnClose);
  conn.on('error', onConnError);

  function onConnData(d) {
    data_str = d.toString('utf8');
    console.log('timestamp: '+timestamp++ + ' | received: ' + data_str);
    io.emit('data', data_str);
    conn.write(d);
  }

  function onConnClose() {
    console.log('connection from %s closed', remoteAddress);
  }

  function onConnError(err) {
    console.log('Connection %s error: %s', remoteAddress, err.message);
  }
}

/* Socket io */
var io = require('socket.io').listen(http);
io.on('connection', function(socket){
    //io.emit('data', 'Connected to server');
});
