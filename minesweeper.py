#Copyright 2024 Thomas Kelly kellthom@bu.edu
#Copyright 2024 Jerry Zheng jzheng7@bu.edu


#To run this help:

# 1: Go to https://minesweeperonline.com/ and set page zoom to 75%.  Put the board in the left hand corner of the browser.
#  The board only need to approximately be in this position for the code to properly process it.  You can troubleshoot later if needed.

# 2: Run the code and wait for the model to train.  Make sure to have the game open and on the screen when the model is done training.

# 3: Each time you want an updated board helper, simply close the pop up window and another will appear.  The tiles are scaled from
#  blue (safe) to red (dangerous).

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
#This class was defined with the help of online tutorials and ChatGPT
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

# Weight map for each value
maps = {-3: 0.0,  # Unrevealed square
    -1: 1.0,  # Revealed square
    0: 0.0,   # Boundary
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 4.0,
    5: 5.0,
    6: 6.0,
    7: 7.0,
    8: 8.0}

#Size of board
rows=16
cols=30

#test input image of board
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

# algoritm()
#Algorithm that returns score for tile
#***********************************************************************
def algorithm(row, col):
    #Look at 8 tiles around tile
    #Do those tiles have mines unaccounted for
    return 0
#***********************************************************************

# color()
#***********************************************************************
def color(probability):
    """
    Map a probability (0 to 1) to a color gradient between blue and red.
    """
    probability = max(0, min(1, probability))
    r = int(probability * 255)      # Red component = increased probability
    b = int((1-probability) * 255)  # Blue component = decreased probability
    g = 0

    return r, g, b
#***********************************************************************

# mine_probability
#Determines probability of mine
#***********************************************************************
def mine_probability(matrix, weights):
    """
    Calculate the probability of the center square being a mine based on the surrounding squares.
    """
    # Extract surrounding elements (exclude the center element)
    elements=matrix
    # revealed empty squares = -1
    revealed=np.sum(elements == -1)
    # Check for revealed numbers and calculate the required mines
    numbers=[value for value in elements if value > 0]
    total_required_mines = sum(numbers)

    # If there is one revealed square and total_required_mines >= 1
    if revealed >= 7 and total_required_mines >= 1:
        return 1.0

    # Apply weights to surrounding elements
    weighted=np.array([weights[value] * np.sum(elements == value)
        for value in np.unique(elements)])
    total_weighted_counts = np.sum(weighted)

    # If total weighted counts are zero, return zero probability
    if total_weighted_counts == 0:
        return 0.0
    probability = total_weighted_counts/8

    return max(0, min(1, probability))
#***********************************************************************

# gen_8_array()
#Returns 8-element array describing surrounding tiles
#***********************************************************************
def gen_8_array(row, col, tile_type,img):

    vals=np.zeros(8)

    a=int(tile_center_rows[row])
    b=int(tile_center_cols[col])
    img[a-14:a+14,b-14:b+14]=0

    #Top Left Square
    if row !=0 and col !=0:
        vals[0]=tile_type[row-1,col-1]

    #Top Middle Square
    if row !=0:
        vals[1]=tile_type[row-1,col]

    #Top Left Square
    if row !=0 and col !=cols-1:
        vals[2]=tile_type[row-1,col+1]

    #Left Square
    if col !=0:
        vals[3]=tile_type[row,col-1]

    #Right Square
    if col !=cols-1:
        vals[4]=tile_type[row,col+1]

    #Bottom Left Square
    if row !=rows-1 and col !=0:
        vals[5]=tile_type[row+1,col-1]

    #Bottom Middle Square
    if row !=rows-1:
        vals[6]=tile_type[row+1,col]

    #Top Left Square
    if row !=rows-1 and col !=cols-1:
        vals[7]=tile_type[row+1,col+1]

    return vals
#***********************************************************************


# get_tile_type()
#Function that get type of tile
#***********************************************************************
def get_tile_type(i,j,img):

    #Ensure indices are integers
    i=int(i)
    j=int(j)
    
    #If there is white to the left, this tile is covered
    for step in range(15):
        if img[i,j-step] >250:          
            return covered

    #Grayscale value of blank tile
    if img[i,j]>180:
        return blank

    #Debug
    #img[int(i)-14:int(i)+14,int(j)-14:int(j)+14]=0 
    
    #Otherwise, the tile is a number
    return number
#***********************************************************************

# build_tile_centers()
# Gets the center points of all the tiles on the board
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

    #Locate rows
    for i in range(start_peak+10,500):

        #If we reached a peak
        if img[i,200]>240:

            #Mark it
            end_peak=i

            #If it is not the same as the previous peak
            if end_peak-start_peak >5 and count != rows+1:

                #store the row (skip the first)
                if count!=0:
                    #img[int((start_peak+end_peak)/2),:]=0
                    tile_center_rows[count-1]=int((start_peak+end_peak)/2)
                
                
                start_peak=end_peak
                count=count+1

    #debug
    #img[int(tile_center_rows[0])-14:int(tile_center_rows[0])+14,int(tile_center_cols[0])-14:int(tile_center_cols[0])+14]=0

    return img
#***********************************************************************

    

# process()
#Function to process board
#***********************************************************************
def process(tile_type,img):

    #Go through each tile
    for i in range(rows):
        for j in range(cols):

            #If a square has already been processed, no need to process again
            if processed[i,j] !=1:
                
                #Get the type of the tile
                tile_t=get_tile_type(tile_center_rows[i],tile_center_cols[j],img)
                
            
                #Variables for debug
                a=int(tile_center_rows[i])
                b=int(tile_center_cols[j])

                if tile_t==blank:
                    #img[a-10:a+10,b-10:b+10]=0
                    tile_type[i,j]=blank

                    #Mark as processed
                    processed[i,j]=1

                if tile_t==number:
                    
                    #The next few lines that handle model prediction were written with the help of online tutorial and ChatGPT
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
                    batch_size = 4

                    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                        download=True, transform=transform)
                    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                            shuffle=False, num_workers=1)

 
                    test_img=img[int(tile_center_rows[i])-14:int(tile_center_rows[i])+14,int(tile_center_cols[j])-14:int(tile_center_cols[j])+14]
                    test_img=255-test_img

                    transform = transforms.ToTensor()
                    outputs = net(transform(test_img))
                    _, predicted = torch.max(outputs, 1)

                    #print(predicted[0].item())
                    #img[a-10:a+10,b-10:b+10]=0

                    #Predicted
                    val=predicted[0].item()
                    if val == 9:
                        val=4
                    if val ==7:
                        val=2

            
                    #img[a-10:a+10,b-10:b+10]=0
                    #Mark as processed
                    tile_type[i,j]=val
                    processed[i,j]=1

                if tile_t==covered:
                    #Call algorithm to assign score
                    #img[int(tile_center_rows[i])-14:int(tile_center_rows[i])+14,int(tile_center_cols[j])-14:int(tile_center_cols[j])+14]=0 
                    score=algorithm(i,j)
                    tile_type[i,j]=covered
        #print("*********************************")
    return tile_type
#***********************************************************************

#Screen capture function
#***********************************************************************
def minesweeper(region, d_size):
    """Screen Capture Function"""
    try:

        #Init board
        tile_type=np.zeros((rows,cols))

        #Rows and cols have not been set yet
        rows_cols_set=False

        while True:

            #Get screen
            img = cv2.cvtColor(np.array(pyautogui.screenshot(region=region)), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, d_size)

            #Get screen for output
            output_img = cv2.cvtColor(np.array(pyautogui.screenshot(region=region)), cv2.COLOR_BGR2RGB)
            output_img = cv2.resize(output_img, d_size)

            #Build rows and columns
            if(rows_cols_set==False):
                img = build_tile_centers(img)
                rows_cols_set=True

            #Process board
            tile_type=process(tile_type, img)

            #debug
            for i in range(rows):
                for j in range(cols):

                    #If the tile is uncovered
                    if tile_type[i,j]==-3:

                        #get values of surrounding tiles
                        vals=gen_8_array(i,j, tile_type,img)

                        #score tile
                        score=mine_probability(vals,maps)

                        #shade tile                
                        if (~np.isin(vals, [-3,0])).any():

                            a=int(tile_center_rows[i])
                            b=int(tile_center_cols[j])

                            output_img[a-5:a+5,b-5:b+5]=color(score)

            #Show the image
            cv2.imshow("Minesweeper Helper", output_img)
            if cv2.waitKey(0) == ord(' '):
                break
    finally:
        cv2.destroyAllWindows()
#***********************************************************************


#***********************************************************************
def main():
    w,h = pyautogui.size()
    size = (800, 500)
    coordinates=(0,150,*size)

    #Define transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    batch_size = 8


    #All code below this comment was written with the help of tutorial and ChatGPT

    #Training Data and params
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=.0001)#, momentum=0.9) #.0001 and 30 works best so far

    
    #Train the model
    #***********************************************************************
    for epoch in range(20):  # loop over the dataset multiple times

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
    #***********************************************************************

    #Save the model
    PATH = './ms_vision.pth'
    torch.save(net.state_dict(), PATH)
    
    minesweeper(region=coordinates, d_size=(800, 500))
#***********************************************************************
    

if __name__ == '__main__':
    main()



    # #Test Data
    # testset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                     download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=1)
    #Testing
    #***********************************************************************
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
    #***********************************************************************


# The code that handles the building and training of the image classification model as well as extraction of predictions was made with the help of online pytorch resources combined with input from ChatGPT.

# First, I took an image classification tutorial (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and asked ChatGPT how this code could be adapted for a number recognition model. 
#  It noted that certain parameters need to be updated and provided those updates.  It also suggested using an Adam optimizer rather than the default.

# The model layers still weren't working quite right, so I showed ChatGPT how layers were constructed in a separate MNIST pytorch tutorial (https://medium.com/@ramamurthi96/a-simple-neural-network-model-for-mnist-using-pytorch-4b8b148ecbdc).  
# It then suggested changes to my current layer structure given the new ones I had prompted it with.

# So, ChatGPT helped explain how the different dimensions change across operations and helped me adapt the existing classification and MNIST tutorials for a Minesweeper number classification problem.  As someone with next to no ML experience, this was a rewarding exercise because I was able to understand how the model was coming together.  First, I defined the layers and how the data would flow beteen them before a prediction was made.  Then, I pulled from the training dataset and trained the model, with weights adjusted according to the optimizer.  Finally, I was able to feed it data from the board an receive predictions back.

# The code that was generated with the help of tutorials and ChatGPT is denoted in the script.