var tokenRecord;
var timeRecord;
var time = 0;
var recordStart = false;

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
    

  if (data.hasOwnProperty("DeepFace")){
   var data1 = [
     {x: "neutral", value: parseFloat(data.DeepFace.neutral)},
     {x: "happy", value: parseFloat(data.DeepFace.happy)},
     {x: "surprise", value: parseFloat(data.DeepFace.surprise)},
     {x: "disgust", value: parseFloat(data.DeepFace.disgust)},
     {x: "fear", value: parseFloat(data.DeepFace.fear)},
     {x: "sad", value: parseFloat(data.DeepFace.sad)},
     {x: "angry", value: parseFloat(data.DeepFace.angry)}
   ];
  }


  if (data.hasOwnProperty("TransferLearning")){
   var data2 = [
    {x: "neutral", value: parseFloat(data.TransferLearning.neutral)},
    {x: "happy", value: parseFloat(data.TransferLearning.happyTL)},
    {x: "surprise", value: parseFloat(data.TransferLearning.surprise)},
    {x: "disgust", value: parseFloat(data.TransferLearning.disgust)},
    {x: "fear", value: parseFloat(data.TransferLearning.fear)},
    {x: "sad", value: parseFloat(data.TransferLearning.sad)},
    {x: "angry", value: parseFloat(data.TransferLearning.angry)}
  ];
}

if (data.hasOwnProperty("DeepLearning")){
  var data3 = [
    {x: "neutral", value: parseFloat(data.DeepLearning.neutral)},
    {x: "happy", value: parseFloat(data.DeepLearning.happy)},
    {x: "surprise", value: parseFloat(data.DeepLearning.surprise)},
    {x: "disgust", value: parseFloat(data.DeepLearning.disgust)},
    {x: "fear", value: parseFloat(data.DeepLearning.fear)},
    {x: "sad", value: parseFloat(data.DeepLearning.sad)},
    {x: "angry", value: parseFloat(data.DeepLearning.angry)}
  ];  
}

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
    if (data.hasOwnProperty("DeepFace")){
      chart.area(data1).name('DeepFace').markers(true).fill("#E55934", 0.3).stroke("#E55934")
    }
    // create second series
    if (data.hasOwnProperty("TransferLearning")){
      chart.area(data2).name('Transfer Learninig').markers(true).fill("#9BC53D", 0.3).stroke("#9BC53D")
    }
    // create third series
    if (data.hasOwnProperty("DeepLearning")){
      chart.area(data3).name('Deep Learninig').markers(true).fill("#5BC0EB", 0.3).stroke("#5BC0EB")
    }

    // set chart title
    chart.title("")
      // set legend
      .legend(true);

    // set container id for the chart
    chart.container('container');
    // initiate chart drawing
    chart.draw();

    printLogs(data);
}

function printLogs(data){
  if (data.hasOwnProperty("DeepFace")){
    document.getElementById('AngryDF').textContent = data.DeepFace.angry + '%';
    document.getElementById('DisgustDF').textContent = data.DeepFace.disgust + '%';
    document.getElementById('FearDF').textContent = data.DeepFace.fear + '%';
    document.getElementById('HappyDF').textContent = data.DeepFace.happy + '%';
    document.getElementById('NeutralDF').textContent = data.DeepFace.neutral + '%';
    document.getElementById('SadDF').textContent = data.DeepFace.sad + '%';
    document.getElementById('SurpriseDF').textContent = data.DeepFace.surprise + '%';

    document.getElementById('secondModel').style.borderColor = "#E55934";
    document.getElementById('secondModelTop').style.background = "#E55934";
  }

  if (data.hasOwnProperty("TransferLearning")){
    document.getElementById('AngryTL').textContent = data.TransferLearning.angry + '%';
    document.getElementById('DisgustTL').textContent = data.TransferLearning.disgust + '%';
    document.getElementById('FearTL').textContent = data.TransferLearning.fear + '%';
    document.getElementById('HappyTL').textContent = data.TransferLearning.happy + '%';
    document.getElementById('NeutralTL').textContent = data.TransferLearning.neutral + '%';
    document.getElementById('SadTL').textContent = data.TransferLearning.sad + '%';
    document.getElementById('SurpriseTL').textContent = data.TransferLearning.surprise + '%';

    document.getElementById('firstModel').style.borderColor = "#9BC53D";
    document.getElementById('firstModelTop').style.background = "#9BC53D";
  }

  if (data.hasOwnProperty("DeepLearning")){
    document.getElementById('AngryDL').textContent = data.DeepLearning.angry + '%';
    document.getElementById('DisgustDL').textContent = data.DeepLearning.disgust + '%';
    document.getElementById('FearDL').textContent = data.DeepLearning.fear + '%';
    document.getElementById('HappyDL').textContent = data.DeepLearning.happy + '%';
    document.getElementById('NeutralDL').textContent = data.DeepLearning.neutral + '%';
    document.getElementById('SadDL').textContent = data.DeepLearning.sad + '%';
    document.getElementById('SurpriseDL').textContent = data.DeepLearning.surprise + '%';

    document.getElementById('thirdModel').style.borderColor = "#5BC0EB";
    document.getElementById('thirdModelTop').style.background = "#5BC0EB";
  }
}

function plotVoidChart (){
  var data1 = [
    {x: "neutral", value: 0},
    {x: "happy", value: 0},
    {x: "surprise", value: 0},
    {x: "disgust", value: 0},
    {x: "fear", value: 0},
    {x: "sad", value: 0},
    {x: "angry", value: 0}
  ];

  var chart = anychart.radar();

  // set chart yScale settings
  chart.yScale()
    .minimum(0)
    .maximum(100)
    .ticks({'interval':20});

   // color alternating cells
   chart.yGrid().palette(["gray 0.1", "gray 0.2"]);

    // create first series
    chart.area(data1).name('NO MODELS').markers(true).fill("gray", 0.3).stroke("gray")

    // set chart title
    chart.title("")
      // set legend
      .legend(true);

    // set container id for the chart
    chart.container('container');
    // initiate chart drawing
    chart.draw();
}

function startRecord (){
  if (!recordStart){
    fetch('http://localhost:5000/start', {
        method:'GET',
        headers:{
            'Content-Type':'application/json'
        }
    });

    // document.getElementById('start').disable = true;
    // document.getElementById('stop').disable = false;
    recordStart = true;
    tokenRecord = setInterval(start, 300)
    time = 0;
    timeRecord = setInterval(() => {
      time = time +1;
      HHMMSStime = new Date(time * 1000).toISOString().slice(11, 19);
      document.getElementById("time").textContent = HHMMSStime
    }, 1000)
  } else{
    alert("recording is started!")
  }
}

function stopRecord (){
  if(taskStart){
    stopTask();
  }
  fetch('http://localhost:5000/stop', {
        method:'GET',
        headers:{
            'Content-Type':'application/json'
        }
    });

  recordStart = false;
  clearInterval(tokenRecord)
  clearInterval(timeRecord)
  time = 0
  document.getElementById("time").textContent = "00:00:00"
  // document.getElementById('start').disable = false;
  // document.getElementById('stop').disable = true;
}

plotVoidChart()


