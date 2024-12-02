import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyautogui

#Number Classifier Class
#***********************************************************************
class Net(nn.Module):

        #Define functions
        def __init__(self):
            super().__init__()

            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10))
        

        #Implement functions
        def forward(self, x):
            x = self.flatten(torch.as_tensor(x))#self.flatten(x)
            logits = self.linear_relu_stack(x)
            return  logits

net = Net()
#***********************************************************************

#Create number recognition model

#Divide up the model into a grid
#Can be done without the use of AI

#Divide squares into 3 groups: revealed blank, revealed with number, and unrevealed
#This can be done without the use of AI

#The revealed squares with numbers are the ones that must be passed to the model to identify the number
#Once a square is processed, it need not be processed again

#The revealed squares that are blank can be left alone


#The unrevealed squares will be assigned a percent chance of a mine as a function of 
# the numbers in the squares around it, the number of unrevealed tiles left, and the number of mines left
# This algorithm is probably enough work for one person, rest can be 

#Size of board
rows=16
cols=30

#Input image of board
img = cv2.imread('board1.png',0) 

#Arrays to keep track of tile centers
tile_center_rows=np.zeros(rows)
tile_center_cols=np.zeros(cols)



#Array to keep track of what tile is processed
processed=np.zeros((rows,cols))

#tile types

blank=-1
number=-2
covered=-3


#Algorithm that returns score for tile
#***********************************************************************
def algorithm(row, col):
    a=1
#***********************************************************************

# get_tile_type()
#Function that get type of tile
#***********************************************************************
def get_tile_type(i,j,img):
    i=int(i)
    j=int(j)
    
    
    #If there is white to the left, this tile is covered
    for step in range(15):
        if img[i,j-step] >250:          
            return covered

    #Grayscale value of blank tile
    if img[i,j]>180:
        
        return blank

    #img[int(i)-14:int(i)+14,int(j)-14:int(j)+14]=0 
    
    return number
#***********************************************************************

# build_tile_centers()
# Gets the center points of a ll the tiles on the board
#***********************************************************************
def build_tile_centers(img):

    #Find start
    start=0
    for i in range(len(img[200,:])):
        if img[200,i]<130:
            start=i
            break
    
    #Locate columns
    start_peak=start
    end_peak=0
    count=0


    #Walk across board
    for i in range(start_peak+10,800):

        #If we reached a peak
        if img[200,i]>240:

            #Mark it
            end_peak=i

            #If it is not the same as the previous peak
            if end_peak-start_peak >5 and count != cols:
                #Draw a line between the peaks
                #img[:, int((start_peak+end_peak)/2)]=0

                #store the column
                tile_center_cols[count]=int((start_peak+end_peak)/2)
                
                start_peak=end_peak

                count=count+1


    #Find start
    start_horz=2
    for i in range(20,len(img[:,199])):
        if img[i,199]==255:
            start_horz=i
            break
    start_peak=start_horz
    end_peak=0
    count=0


    for i in range(start_peak+10,500):

        #If we reached a peak
        if img[i,200]>240:

            #Mark it
            end_peak=i

            #If it is not the same as the previous peak
            if end_peak-start_peak >5 and count != rows+1:

                #store the column
                if count!=0:
                    #img[int((start_peak+end_peak)/2),:]=0
                    tile_center_rows[count-1]=int((start_peak+end_peak)/2)
                
                
                start_peak=end_peak

                count=count+1

    #img[int(tile_center_rows[0])-14:int(tile_center_rows[0])+14,int(tile_center_cols[0])-14:int(tile_center_cols[0])+14]=0
    return img
#***********************************************************************

    

# process()
#Function to process board
#***********************************************************************
def process(tile_type,img):

    for i in range(rows):
        for j in range(cols):

            #If a square has already been processed, no need to process again
            if processed[i,j] !=1:
                
                
                tile_t=get_tile_type(tile_center_rows[i],tile_center_cols[j],img)
                
            
                #temp vars
                a=int(tile_center_rows[i])
                b=int(tile_center_cols[j])

                if tile_t==blank:
                    #Mark as processed
                    #img[a-10:a+10,b-10:b+10]=0
                    tile_type[i,j]=blank
                    processed[i,j]=1

                if tile_t==number:
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
                    batch_size = 4

                    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                        download=True, transform=transform)
                    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                            shuffle=False, num_workers=1)



                    # dataiter = iter(testloader)
                    # images, labels = next(dataiter)
                    # outputs = net(images)
                    # _, predicted = torch.max(outputs, 1)
                    # print(predicted)
                    # for image in images:
                    #     plt.imshow(image[0,:,:],cmap='gray')
                    #     plt.show()

 
                    test_img=img[int(tile_center_rows[i])-14:int(tile_center_rows[i])+14,int(tile_center_cols[j])-14:int(tile_center_cols[j])+14]
                    test_img=255-test_img

                    transform = transforms.ToTensor()
                    outputs = net(transform(test_img))
                    _, predicted = torch.max(outputs, 1)
                    print(predicted[0].item())

            
                    #img[a-10:a+10,b-10:b+10]=0
                    #Mark as processed
                    tile_type[i,j]=predicted[0].item()
                    processed[i,j]=1

                if tile_t==covered:
                    #Call algorithm to assign score
                    #img[int(tile_center_rows[i])-14:int(tile_center_rows[i])+14,int(tile_center_cols[j])-14:int(tile_center_cols[j])+14]=0 
                    score=algorithm(i,j)
                    tile_type[i,j]=covered
    return tile_type
#***********************************************************************

#Screen capture function
#***********************************************************************
def minesweeper(region, d_size):
    """Screen Capture Function"""
    try:

        tile_type=np.zeros((rows,cols))
        rows_cols_set=False

        while True:
            img = cv2.cvtColor(np.array(pyautogui.screenshot(region=region)), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, d_size)
            if(rows_cols_set==False):
                img = build_tile_centers(img)
                rows_cols_set=True
            tile_type=process(tile_type, img)

            print(tile_type[0,:])
            cv2.imshow("Minesweeper Helper", img)
            if cv2.waitKey(0) == ord(' '):
                break
    finally:
        cv2.destroyAllWindows()
#***********************************************************************


#***********************************************************************
def main():
    w,h = pyautogui.size()
    size = (800, 500)
    # coordinates = ((w-size[0])//2,
    #                (h-size[1])//2, *size,
    # )
    coordinates=(0,150,*size)
    
    #build_tile_centers()
    
#     process()


    #Receieve live view of board

    #loop that processes unprocessed tiles

    #model is only called when a new number is overturned, reducing computation time

    #Define transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    batch_size = 8

    #Training Data
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    # #Test Data
    # testset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                     download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=1)

 

    # class Net(nn.Module):

    #     #Define functions
    #     def __init__(self):
    #         super().__init__()

    #         self.flatten = nn.Flatten()
    #         self.linear_relu_stack = nn.Sequential(
    #         nn.Linear(784, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 10))
        

    #     #Implement functions
    #     def forward(self, x):
    #         x = self.flatten(x)
    #         logits = self.linear_relu_stack(x)
    #         return  logits

    # # #Init net
    # net = Net()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=.0001)#, momentum=0.9)

    
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # dataiter = iter(testloader)
    # images, labels = next(dataiter)
    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)


    # # print images
    # for image in images:
    #     plt.imshow(image[0,:,:],cmap='gray')
    #     #print(f'{classes[predicted[0]]:5s}')
    #     print(predicted)
    #     print(labels)
    #     plt.show()

    #print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    
    minesweeper(region=coordinates, d_size=(800, 500))
#***********************************************************************
    

if __name__ == '__main__':
    main()
                