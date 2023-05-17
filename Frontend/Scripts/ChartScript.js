var tokenRecord;

function start() {
  document.getElementById('stream').src="../../Backend/Image/StreamRead.png"  + "?" + Date.now()
    fetch('http://localhost:5000/result', {
        method:'GET',
        headers:{
            'Content-Type':'application/json'
        }
    })
    .then(resp => resp.json())
    .then((data) => plotChart(data))
    .catch(error => console.log(error));
  }

async function plotChart (data) {

    document.getElementById('container').innerHTML = "";
    
    // our data from bulbapedia
   var data1 = [
     {x: "neutral", value: parseFloat(data.neutral)},
     {x: "happy", value: parseFloat(data.happy)},
     {x: "surprise", value: parseFloat(data.surprise)},
     {x: "disgust", value: parseFloat(data.disgust)},
     {x: "fear", value: parseFloat(data.fear)},
     {x: "sad", value: parseFloat(data.sad)},
     {x: "angry", value: parseFloat(data.angry)}
   ];

//    var data2 = [
//      {x: "Speed", value: 45},
//      {x: "HP", value: 45},
//      {x: "Defense", value: 49},
//      {x: "Special Defense", value: 65},
//      {x: "Special Attack", value: 65},
//      {x: "Attack", value: 49}
//    ];  

//    var data3 = [
//      {x: "Speed", value: 43},
//      {x: "HP", value: 44},
//      {x: "Defense", value: 65},
//      {x: "Special Defense", value: 64},
//      {x: "Special Attack", value: 50},
//      {x: "Attack", value: 48}
//    ];  

   // create radar chart
   var chart = anychart.radar();

   // set chart yScale settings
   chart.yScale()
     .minimum(0)
     .maximum(100)
     .ticks({'interval':20});

    // color alternating cells
    chart.yGrid().palette(["gray 0.1", "gray 0.2"]);

    // create first series
    chart.area(data1).name('DeepFace').markers(true).fill("#E55934", 0.3).stroke("#E55934")
    // create second series
    // chart.area(data2).name('Bulbasaur').markers(true).fill("#9BC53D", 0.3).stroke("#9BC53D")
    // // create third series
    // chart.area(data3).name('Squirtle').markers(true).fill("#5BC0EB", 0.3).stroke("#5BC0EB")

    // set chart title
    chart.title("User emotions in real time")
      // set legend
      .legend(true);

    // set container id for the chart
    chart.container('container');
    // initiate chart drawing
    chart.draw();
}

function startRecord (){
  fetch('http://localhost:5000/start', {
        method:'GET',
        headers:{
            'Content-Type':'application/json'
        }
    });

  // document.getElementById('start').disable = true;
  // document.getElementById('stop').disable = false;
  tokenRecord = setInterval(start, 300)
}

function stopRecord (){
  fetch('http://localhost:5000/stop', {
        method:'GET',
        headers:{
            'Content-Type':'application/json'
        }
    });

  clearInterval(tokenRecord)
  // document.getElementById('start').disable = false;
  // document.getElementById('stop').disable = true;
}



