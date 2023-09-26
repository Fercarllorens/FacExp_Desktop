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
    
  if (data.hasOwnProperty("DeepFace")){
   var data1 = [
     {x: "neutral", value: parseFloat(data.DeepFace.neutralDF)},
     {x: "happy", value: parseFloat(data.DeepFace.happyDF)},
     {x: "surprise", value: parseFloat(data.DeepFace.surpriseDF)},
     {x: "disgust", value: parseFloat(data.DeepFace.disgustDF)},
     {x: "fear", value: parseFloat(data.DeepFace.fearDF)},
     {x: "sad", value: parseFloat(data.DeepFace.sadDF)},
     {x: "angry", value: parseFloat(data.DeepFace.angryDF)}
   ];
  }


  if (data.hasOwnProperty("TransferLearning")){
   var data2 = [
    {x: "neutral", value: parseFloat(data.TransferLearning.neutralTL)},
    {x: "happy", value: parseFloat(data.TransferLearning.happyTL)},
    {x: "surprise", value: parseFloat(data.TransferLearning.surpriseTL)},
    {x: "disgust", value: parseFloat(data.TransferLearning.disgustTL)},
    {x: "fear", value: parseFloat(data.TransferLearning.fearTL)},
    {x: "sad", value: parseFloat(data.TransferLearning.sadTL)},
    {x: "angry", value: parseFloat(data.TransferLearning.angryTL)}
  ];
}

if (data.hasOwnProperty("DeepLearning")){
  var data3 = [
    {x: "neutral", value: parseFloat(data.DeepLearning.neutralDL)},
    {x: "happy", value: parseFloat(data.DeepLearning.happyDL)},
    {x: "surprise", value: parseFloat(data.DeepLearning.surpriseDL)},
    {x: "disgust", value: parseFloat(data.DeepLearning.disgustDL)},
    {x: "fear", value: parseFloat(data.DeepLearning.fearDL)},
    {x: "sad", value: parseFloat(data.DeepLearning.sadDL)},
    {x: "angry", value: parseFloat(data.DeepLearning.angryDL)}
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
    document.getElementById('AngryDF').textContent = data.DeepFace.angryDF + '%';
    document.getElementById('DisgustDF').textContent = data.DeepFace.disgustDF + '%';
    document.getElementById('FearDF').textContent = data.DeepFace.fearDF + '%';
    document.getElementById('HappyDF').textContent = data.DeepFace.happyDF + '%';
    document.getElementById('NeutralDF').textContent = data.DeepFace.neutralDF + '%';
    document.getElementById('SadDF').textContent = data.DeepFace.sadDF + '%';
    document.getElementById('SurpriseDF').textContent = data.DeepFace.surpriseDF + '%';

    document.getElementById('secondModel').style.borderColor = "#E55934";
    document.getElementById('secondModelTop').style.background = "#E55934";
  }

  if (data.hasOwnProperty("TransferLearning")){
    document.getElementById('AngryTL').textContent = data.TransferLearning.angryTL + '%';
    document.getElementById('DisgustTL').textContent = data.TransferLearning.disgustTL + '%';
    document.getElementById('FearTL').textContent = data.TransferLearning.fearTL + '%';
    document.getElementById('HappyTL').textContent = data.TransferLearning.happyTL + '%';
    document.getElementById('NeutralTL').textContent = data.TransferLearning.neutralTL + '%';
    document.getElementById('SadTL').textContent = data.TransferLearning.sadTL + '%';
    document.getElementById('SurpriseTL').textContent = data.TransferLearning.surpriseTL + '%';

    document.getElementById('firstModel').style.borderColor = "#9BC53D";
    document.getElementById('firstModelTop').style.background = "#9BC53D";
  }

  if (data.hasOwnProperty("DeepLearning")){
    document.getElementById('AngryDL').textContent = data.DeepLearning.angryDL + '%';
    document.getElementById('DisgustDL').textContent = data.DeepLearning.disgustDL + '%';
    document.getElementById('FearDL').textContent = data.DeepLearning.fearDL + '%';
    document.getElementById('HappyDL').textContent = data.DeepLearning.happyDL + '%';
    document.getElementById('NeutralDL').textContent = data.DeepLearning.neutralDL + '%';
    document.getElementById('SadDL').textContent = data.DeepLearning.sadDL + '%';
    document.getElementById('SurpriseDL').textContent = data.DeepLearning.surpriseDL + '%';

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

plotVoidChart()


