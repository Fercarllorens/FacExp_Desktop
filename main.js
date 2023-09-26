const {app, BrowserWindow, Menu} = require('electron');
//require('electron-reload')(__dirname);

function createWindow () {
    const mainWindow = new BrowserWindow({
        width:1920,
        height:1080
    });

    mainWindow.loadFile("Frontend/UI/index.html");
    mainWindow.webContents.openDevTools();
    Menu.setApplicationMenu(null);
}

app.whenReady().then(() => 
    createWindow()
);