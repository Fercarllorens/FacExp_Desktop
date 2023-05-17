const {app, BrowserWindow, Menu} = require('electron');
// require('electron-reload')(__dirname)

function createWindow () {
    const mainWindow = new BrowserWindow({
        width:800,
        height:600
    });

    mainWindow.loadFile("Frontend/UI/index.html");
    mainWindow.webContents.openDevTools();
    Menu.setApplicationMenu(null);
}

app.whenReady().then(() => 
    createWindow()
);