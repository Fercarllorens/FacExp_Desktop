function changePageConfig(){
    stopRecord();
    var oldBody = document.getElementById("bodyMain");
    var newBody = document.getElementById("bodyCfg");

    var count = 0; 

    var selectWebcam = document.getElementById("webcam")

        oldBody.style.display = "none";
        newBody.style.display = "block";

        navigator.mediaDevices
        .enumerateDevices()
        .then((devices) => {
          devices.forEach((device) => {
            if(device.kind == 'videoinput'){
                selectWebcam.innerHTML += `<option id='webcam${count}' value='${count}'>${device.label}</option>`
                count++;
            }
          });
        })
}

function changePageMain(){
    var oldBody = document.getElementById("bodyCfg");
    var newBody = document.getElementById("bodyMain");

        oldBody.style.display = "none";
        newBody.style.display = "block";
}

function submitModels(){
    DeepFace = document.getElementById("DeepFaceEnable").checked
    Transfer = document.getElementById("TransferEnable").checked
    Deep = document.getElementById("DeepLearningEnable").checked

    fetch('http://localhost:5000/models', {
        method:'POST',
        headers:{
            'Accept':'application/json',
            'Content-Type':'application/json'
        },
        body:JSON.stringify({
            "DeepFace": DeepFace,
            "Transfer": Transfer,
            "Deep": Deep
        })
    });

    getStatus()
}

function submitLogs(){
    Folder =  document.getElementById("LogsPath").value;
    Name = document.getElementById("LogsName").value;

    Route = Folder + "//" + Name + ".csv";

    Enabled =  document.getElementById("LogsEnable").checked;

    fetch('http://localhost:5000/log', {
        method:'POST',
        headers:{
            'Accept':'application/json',
            'Content-Type':'application/json'
        },
        body:JSON.stringify({
            "Logs": Enabled,
            "LogsFolder": Route
        })
    });

    getStatus();
}

function submitRecords(){
    Folder =  document.getElementById("RecordsPath").value;
    Name = document.getElementById("RecordsName").value;

    RouteVideo = Folder + "//" + Name + ".mp4";
    RouteScreen = Folder + "//" + Name + '_Screen' + ".mp4";

    EnabledVideo =  document.getElementById("WebcamEnable").checked;
    EnabledScreen =  document.getElementById("ScreenEnable").checked;

    fetch('http://localhost:5000/video', {
        method:'POST',
        headers:{
            'Accept':'application/json',
            'Content-Type':'application/json'
        },
        body:JSON.stringify({
            "Video": EnabledVideo,
            "VideoFolder": RouteVideo
        })
    });

    fetch('http://localhost:5000/screen', {
        method:'POST',
        headers:{
            'Accept':'application/json',
            'Content-Type':'application/json'
        },
        body:JSON.stringify({
            "Screen": EnabledScreen,
            "ScreenFolder": RouteScreen
        })
    });

    getStatus();
}

function submitWebcam(){
    Webcam = document.getElementById('webcam').value;

    fetch('http://localhost:5000/webcam', {
        method:'POST',
        headers:{
            'Accept':'application/json',
            'Content-Type':'application/json'
        },
        body:JSON.stringify({
            "Webcam": Webcam
        })
    });

    getStatus();
}

function submitScreen(){
    screen = document.getElementById('screen').value;

    fetch('http://localhost:5000/screencam', {
        method:'POST',
        headers:{
            'Accept':'application/json',
            'Content-Type':'application/json'
        },
        body:JSON.stringify({
            "Screen": screen
        })
    });

    getStatus();
}

function startTask(){
    fetch('http://localhost:5000/task', {
        method:'GET',
        headers:{
            'Content-Type':'application/json'
        }
    })
    .then(resp => resp.json())
    .then((data) => console.log(data))
    .catch(error => console.log(error));  
}

function stopTask(){
    fetch('http://localhost:5000/notask', {
        method:'GET',
        headers:{
            'Content-Type':'application/json'
        }
    })
    .then(resp => resp.json())
    .then((data) => console.log(data))
    .catch(error => console.log(error));  
}

function getStatus(){
    fetch('http://localhost:5000/status', {
        method:'GET',
        headers:{
            'Content-Type':'application/json'
        }
    })
    .then(resp => resp.json())
    .then((data) => changeStatus(data))
    .catch(error => {console.log(error); changeConection();});
}

function changeStatus(data){
    document.getElementById('Connection').textContent = "Connected"
    
    document.getElementById("DeepFaceEnable").checked = data["DeepFace"]
    document.getElementById("TransferEnable").checked = data["Transfer Learning"]
    document.getElementById("DeepLearningEnable").checked = data["Deep Learning"]

    if (data["DeepFace"]){
        document.getElementById("DeepFaceText").innerHTML = 'Enabled <i class="bi-check-circle-fill icon-success"></i>'
    }
    else{
        document.getElementById("DeepFaceText").innerHTML = 'Disabled <i class="bi-x-circle-fill icon-fail"></i>'
    }

    if (data["Transfer Learning"]){
        document.getElementById("TransferText").innerHTML = 'Enabled <i class="bi-check-circle-fill icon-success"></i>'
    }
    else{
        document.getElementById("TransferText").innerHTML = 'Disabled <i class="bi-x-circle-fill icon-fail"></i>'
    }

    if (data["Deep Learning"]){
        document.getElementById("DeepText").innerHTML = 'Enabled <i class="bi-check-circle-fill icon-success"></i>'
    }
    else{
        document.getElementById("DeepText").innerHTML = 'Disabled <i class="bi-x-circle-fill icon-fail"></i>'
    }

    document.getElementById("Video-Folder").textContent = data["Video Folder"] 
    document.getElementById("Screen-Folder").textContent = data["Screen Folder"] 
    document.getElementById("Logs-Folder").textContent = data["Logs Folder"] 

    document.getElementById("LogsEnable").checked = data["Logs"];
    document.getElementById("WebcamEnable").checked = data["Video"];
    document.getElementById("ScreenEnable").checked = data["Screen"];

    document.getElementById("webcam").value = data['Webcam']
    document.getElementById("screen").value = data['Screencam']
    
}



function changeConection(){
    document.getElementById('Connection').textContent = "Disconnected"
}