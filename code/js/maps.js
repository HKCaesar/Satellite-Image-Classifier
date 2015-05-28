var map = null;
var mapinfo;

// Establish connection to node server
var socket = io.connect('http://localhost:8080');
			
function getMap() {
	map = new Microsoft.Maps.Map(document.getElementById('map_id'), 
	{
		credentials: 'AkrBJqwqx23-hEkqsxaxgZFRniylWYEI9pSSfcQz8NZQB0lToABb3ky5lra_rllS',
		width: 816, height: 528,
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
	var link = "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/"+lat+","+lon+"/16?mapSize=816,528&key=" + credentials;
	console.log(link);
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

		$("#location_name").text("Location: " + location);
		socket.emit('location', location)
	}
}	


// Emit event flag to node server
socket.emit('event');

// Perform action upon receiving reply flag
socket.on('reply', function (data) {
	console.log("Received from Node.js: " + data);
});
