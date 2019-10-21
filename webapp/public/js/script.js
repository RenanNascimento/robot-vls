if ( WEBGL.isWebGLAvailable() === false ) {
    document.body.appendChild( WEBGL.getWebGLErrorMessage() );
}

var camera, controls, scene, renderer;
var size = 20;
var divisions = 20;
var lsi_model;
var geometry, material, robot;
var pose_mem = []
var mem_len = 5
var error_delta = 2.0; // meters

var lsi_width = 7.12;
var lsi_length = 5.75;

/*var markers = {
    "0": { "x": 0.65, "y": 1.5 },
    "1": { "x": 2.30, "y": 1.7 },
    "2": { "x": 3.9, "y": 1.65 },
    "3": { "x": 3.9, "y": 2.85 },
    "4": { "x": 3.9, "y": 4.3 },
    "5": { "x": 3.9, "y": 5.9 },
};*/

var markers = {
    "0": { "x": 1.00, "y": 1.60 },
    "1": { "x": 2.46, "y": 1.60 },
    "2": { "x": 4.05, "y": 1.60 },
    "3": { "x": 4.05, "y": 3.06 },
    "4": { "x": 2.68, "y": 4.04 },
    "5": { "x": 1.30, "y": 4.04 },
    "6": { "x": 4.05, "y": 4.46 },
    "7": { "x": 4.05, "y": 5.96 },
};

init();
animate();


function init() {

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    camera = new THREE.OrthographicCamera( window.innerWidth / - 2, window.innerWidth / 2, window.innerHeight / 2, window.innerHeight / - 2, 1, 1000 );
    camera.position.set(100, 50, 0 );

    // controls
    controls = new THREE.OrbitControls( camera, renderer.domElement );
    controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
    controls.dampingFactor = 0.25;
    controls.screenSpacePanning = false;

    // world
    var gridHelper = new THREE.GridHelper(size, divisions);
    scene.add(gridHelper);
    
    var loader = new THREE.ColladaLoader();
    loader.load( '../models/lsi_model.dae', function (collada) {
        lsi_model = collada.scene;
        scene.add(lsi_model);
    });

    // plot markers
    plot_markers();

    // plot robot
    geometry = new THREE.SphereGeometry(0.1, 32, 32);
    material = new THREE.MeshBasicMaterial( {color: 0xff0000} );
    robot = new THREE.Mesh(geometry, material);
    robot.position.set(0, 0.2, 0);
    scene.add(robot);

    // light
    var light = new THREE.AmbientLight( 0x404040 ); // soft white light
    scene.add( light );

    window.addEventListener( 'resize', onWindowResize, false );

}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );
}

function animate() {
    requestAnimationFrame( animate );
    controls.update(); // only required if controls.enableDamping = true, or if controls.autoRotate = true
    render();
}

function render() {
    renderer.render( scene, camera );
}

// handles socket
var socket = io();

// 528 x 432
socket.on('data', function(data){
    console.log(data)
    data = JSON.parse(data)
    if(data.pose){
        dist_x = (data.pose.dist_x_pixels * 2.5 / 528) + markers[data.pose.marker].x;
        dist_y = (data.pose.dist_y_pixels * 2.5 / 528) + markers[data.pose.marker].y;
        if(canPlot(dist_x, dist_y)){
            plot_pose(parseFloat(dist_x.toFixed(2)), parseFloat(dist_y.toFixed(2)))
            //console.log('dist_x: '+dist_x +' | dist_y: '+dist_y)
        }else{
            console.log('ERROR   dist_x: '+dist_x +' | dist_y: '+dist_y)
        }
    }
});

// plot markers
function plot_markers(){
    for (var key in markers) {
        load_marker(key, markers[key].x, markers[key].y);
    }
}

function load_marker(name, x, y){
    var loader_marker = new THREE.TextureLoader();
    var material_marker = new THREE.MeshLambertMaterial({
      map: loader_marker.load('../img/' + name + '.png')
    });
    var geometry_marker = new THREE.PlaneGeometry(0.42, 0.42);
    var marker = new THREE.Mesh(geometry_marker, material_marker);
    marker.position.set(-y, 2.65, -(lsi_length-x))
    marker.rotation.x = -Math.PI/2;
    marker.rotation.z = Math.PI/2;
    scene.add(marker);
}

// plot pose
function plot_pose(x, y){
    robot.position.set(-y, 0.2, -(lsi_length-x));
}


function pit(oldPoint, newPoint){
    var a = Math.abs(oldPoint.dist_x - newPoint.dist_x);
    var b = Math.abs(oldPoint.dist_y - newPoint.dist_y);
    return Math.sqrt( a*a + b*b );
}

function canPlot(x, y){
    var pointObj = {dist_x: x, dist_y: y};

    if(pose_mem.length == 0){
        pose_mem.push(pointObj)
        return true;
    }else{
        var isGreater = false;
        pose_mem.forEach(pos => {
            if(pit(pos, pointObj) > error_delta) isGreater = true;
        });
        if(!isGreater){
            if(pose_mem.length == mem_len){
                pose_mem.shift();
            }
            pose_mem.push(pointObj);
            return true;
        }
    }

    return false;
}