function changePageConfig(){
    stopRecord();
    var oldBody = document.getElementById("bodyMain");
    var newBody = document.getElementById("bodyCfg");

        oldBody.style.display = "none";
        newBody.style.display = "block";
}

function changePageMain(){
    var oldBody = document.getElementById("bodyCfg");
    var newBody = document.getElementById("bodyMain");

        oldBody.style.display = "none";
        newBody.style.display = "block";
}