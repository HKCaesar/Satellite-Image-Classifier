var map = null;
var mapinfo;
var mapWidth = 816;
var mapHeight = 528;
var patchDim = 48;
var labels = [0,1,2];
var label2Color = {
	0 : 'red',
	1 : 'blue',
	2 : 'yellow'
}
// Establish connection to node server
var socket = io.connect('http://localhost:8080');

// Emit event flag to node server
socket.emit('event');

// Perform action upon receiving reply flag
socket.on('reply', function (data) {
	console.log("Received from Node.js: " + data);
});

function onLoad() {
	$('.border-red').click(function() { redClick(); });
	$('.border-blue').click(function() { blueClick(); });
	$('.border-yellow').click(function() { yellowClick(); });
}
			
function getMap() {
	map = new Microsoft.Maps.Map(document.getElementById('map_id'), 
	{
		credentials: 'AkrBJqwqx23-hEkqsxaxgZFRniylWYEI9pSSfcQz8NZQB0lToABb3ky5lra_rllS',
		width: mapWidth, height: mapHeight,
		enableClickableLogo: false,
		enableSearchLogo: false,
		showDashboard: false,
		showMapTypeSelector:false,
		disableZooming: true,
		showScalebar: false,
		disablePanning: true,
		// 100m zoom
		zoom: 16, 
		mapTypeId: Microsoft.Maps.MapTypeId.aerial,
		center: new Microsoft.Maps.Location(52.298379183128596, 4.942471264136956),
		labelOverlay: Microsoft.Maps.LabelOverlay.hidden
	});
	clickLocationInfo();
	render(generateTestSet());
}

function clickRandom() {
	// Manually found bounding box locations
	var minLat = 52.298379183128596;
	var maxLat = 52.43520441347777;
	var minLong = 4.764510040406664;
	var maxLong = 4.942471264136956;

	var randomLat = Math.random() * (maxLat - minLat) + minLat;
	var randomLong = Math.random() * (maxLong - minLong) + minLong;
	
	map.setView({ zoom: 16, center: new Microsoft.Maps.Location(randomLat, randomLong) });
	console.log(randomLat + ", " + randomLong);

	clickLocationInfo();
}

function clickGetImage(credentials) {
	map.getCredentials(makeImageRequest);
}

function makeImageRequest(credentials) {
	var lat = map.getCenter().latitude;
	var lon = map.getCenter().longitude;
	var link = "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/"+lat+","+lon+"/16?mapSize=" + mapWidth + "," + mapHeight +"&key=" + credentials;
	socket.emit('crop', {link, patchDim});
}

function clickLocationInfo(credentials) {
	map.getCredentials(makeLocationRequest);
}

function makeLocationRequest(credentials) {
	var lat = map.getCenter().latitude;
	var lon = map.getCenter().longitude;
	var link = "http://dev.virtualearth.net/REST/v1/Locations/"+lat+","+lon+"?o=json&key=" + credentials;
	callRestService(link, locationCallback);
}

function callRestService(request, callback) {
    $.ajax({
        url: request,
        dataType: "jsonp",
        jsonp: "jsonp",
        success: function (r) {
            callback(r);
        },
        error: function (e) {
            alert(e.statusText);
        }
    });
}

function locationCallback(result) {
	if (typeof result.resourceSets[0].resources[0] !== 'undefined') {
		var location = result.resourceSets[0].resources[0].name;

		$("#header h2").text(location);
		socket.emit('location', location)
	}
}

function generateTestSet() {
	var testReturnProtocol = "" + mapWidth + "," + mapHeight + "," + patchDim;
	var steps = mapWidth*mapHeight/((patchDim/2)*(patchDim/2));
	for (i = 0; i < steps; i++) {
		var randomClass = labels[Math.floor(Math.random() * labels.length)];
		testReturnProtocol = testReturnProtocol + "," + randomClass;
	}
	return testReturnProtocol;
}

function protocol2Array(protocol) {
	return protocol.split(",");
}

function extractWidth(array) {
	var width = array[0];
	array.shift();
	return width;
}

function extractHeight(array) {
	var height = array[0];
	array.shift();
	return height;
}

function extractPatchSize(array) {
	var patchSize = array[0];
	array.shift();
	return patchSize;
}

function render(protocol) {
	var array = protocol2Array(protocol);
	var width = extractWidth(array);
	var height = extractHeight(array);
	var patchDim = extractPatchSize(array);

	var patchDimHalf = patchDim/2;
	var widthSteps = width/patchDimHalf;
	var heightSteps = height/patchDimHalf;

	for(i = 0; i < heightSteps-1; i++) {
		for(j = 0; j < widthSteps; j++) {
			var x = j * patchDimHalf;
			var y = i * patchDimHalf;
			renderDiv(x,y,label2Color[extractFirstLabel(array)]);
		}
	}
}

function extractFirstLabel(array) {
	var label = array[0];
	array.shift();
	return label;
}

function renderDiv(x, y,color) {
	var div = $('<div>', {class: 'labelClass'}).css('margin-left',x).css('margin-top',y).addClass('border-' + color);
	$('#grid').append(div);
}

// Click on divs
function restore() {
	$('.border-red').css('border-color','red');
	$('.border-blue').css('border-color','blue');
	$('.border-yellow').css('border-color','yellow');
}

function redClick() {
	restore();
	$('.border-blue').css('border-color','gray');
	$('.border-yellow').css('border-color','gray');
}

function blueClick() {
	restore();
	$('.border-red').css('border-color','gray');
	$('.border-yellow').css('border-color','gray');
}

function yellowClick() {
	restore();
	$('.border-blue').css('border-color','gray');
	$('.border-red').css('border-color','gray');
}


