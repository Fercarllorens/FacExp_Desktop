import torch 
import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion(nn.Module):
    def __init__(self):
        # Estrctura de la red neuronal planteada en el paper, son declaraciones de variables para la implementación
        super(Deep_Emotion, self).__init__()

        # Capas convolution - extración de características
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        # Maxpool para selección del mayor componente
        self.pool2 = nn.MaxPool2d(2,2)

        # Capas convolution - extración de características
        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        # Maxpool para selección del mayor componente
        self.pool4 = nn.MaxPool2d(2,2)

        # Normalización
        self.norm = nn.BatchNorm2d(10)

        # capas full conect finales
        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

        # red de localización
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Capa full connect de la red de localización
        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x): # parte de localización del modelo, UNIDAD DE ATENCION DE LA RED NEURONAL
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        #Implementación de la red neuronal
        out = self.stn(input) # UNIDAD DE ATENCIÓN

        out = F.relu(self.conv1(out))    #Convolución (extracción de características) + función de activación ReLu (cancela los negativos)
        out = self.conv2(out)            #Convolución (extracción de características)
        out = F.relu(self.pool2(out))    #MaxPool (selección del mayor de la pool) + función de activación ReLu (cancela los negativos)

        out = F.relu(self.conv3(out))    #Convolución (extracción de características) + función de activación ReLu (cancela los negativos)
        out = self.norm(self.conv4(out)) #Convolución (extracción de características) + normalización
        out = F.relu(self.pool4(out))    #MaxPool (selección del mayor de la pool) + función de activación ReLu (cancela los negativos)

        out = F.dropout(out)             #Dropout (elimina algunos nodos neuronales aleatoriamente reduciendo el overfitting)
        out = out.view(-1, 810)          #Reshape
        out = F.relu(self.fc1(out))      #Capa fullconnect 1 que pasa de 810 nodos a 50 + función de activación ReLu
        out = self.fc2(out)              #Capa fullconnect 2 que pasa de 50 nodos a 7 (clasificación de emociones)

        return out