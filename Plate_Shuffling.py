import numpy as np 
import pandas as pd

def Shuffle_1536to1536(seed):
    '''# Here we write a script randomizing the library from 1536 format: (after mating )
    # 0)We create four columns of donor plates. 
    # 1)For each recipient step we choose 1 plate from each of the four column (+ take it out of column)
    # 2) we make sure that a source plate can be chosen only once
    # This function blocks on some seeds (e.g. 1, 4) because towards the end the plates left in the last column are already picked from the first columns
    '''    
    np.random.seed(seed)
    A = np.arange(1,14)
    B = np.arange(1,14)
    C = np.arange(1,14)
    D = np.arange(1,14)

    shuffle =[]
    for plate in range(1,14):
        print(plate)
        red = np.random.choice(A) #choose randomly element in array
        A = np.delete(A,red==A) #delete that element 
        blue = np.random.choice(np.setdiff1d(B,[red])) #choose randomly element in B excluding the element already picked in A
        B = np.delete(B,blue==B)
        yellow  = np.random.choice(np.setdiff1d(C,[red,blue]))
        C = np.delete(C,yellow== C )
        green = np.random.choice(np.setdiff1d(D,[red,blue,yellow]))
        D = np.delete(D,green== D )
        shuffle.append([red,blue,yellow,green])
        

    #format table:  
    shuffle = np.stack(shuffle)
    order = pd.DataFrame(data=np.hstack([np.arange(1,14).reshape(-1,1),shuffle]), columns=[ 'Destination','Red', 'Blue','Yellow','Green'])
    return order
    
def Shuffle_384to1536(seed): 
    '''#for each 1536 destination plates, 4 384 source plates need to contribute
    #source plates cannot contribute more than once to the same destination plate (cannot load a same source plate at different spots on the rotor)
    # this cell is clumsy but it permutes the 13 source plates and add them to an array where destinations are rows and source columns
    # each time a column is added it checks that source plates are not repeated on the same row'''
   
    np.random.seed(seed)
    source=np.zeros((13,4))
    Red= np.random.permutation(np.repeat(np.arange(1,14),1))
    source[:,0] = Red

    Unique=0
    while Unique < 2*13:
        Blue =  np.random.permutation(np.repeat(np.arange(1,14),1))
        Unique=sum([len(np.unique(row)) for  row in np.stack((Red,Blue)).T])
        #print('blue', Blue)

    while Unique < 3*13:
        Yellow =  np.random.permutation(np.repeat(np.arange(1,14),1))
        Unique=sum([len(np.unique(row)) for  row in np.stack((Red,Blue, Yellow)).T])
       # print('yellow',Yellow)

    while Unique < 4*13:
        Green =  np.random.permutation(np.repeat(np.arange(1,14),1))
        Unique=sum([len(np.unique(row)) for  row in np.stack((Red,Blue, Yellow, Green)).T])
       # print('green',Green)


    order=np.stack(( np.arange(1,14),Red,Blue, Yellow, Green)).T


    order = pd.DataFrame(data=order, columns=[ 'Destination','Red', 'Blue','Yellow','Green'])
    return order

def shuffled_finder(orig_p, orig_r, orig_c, order):
    '''finds the shuffled position of a gene given its original positions (in 384 format) and the order table
    orig_p counts from 1; returns counts from 1
    orig_r counts from 0; returns counts from 0
    orig_c counts from 0; returns counts from 0
    '''
    p = np.where(order[['Red','Blue', 'Yellow', 'Green']] == orig_p)[0]+1 
    rbyg =  np.where(order[['Red','Blue', 'Yellow', 'Green']] == orig_p)[1] #the patch color on which the source plate was loaded
    r=[]
    c=[]
    for pos in rbyg:
        #if source plate was in red, the row/col are doubled
        if pos == 0:
            r.append(orig_r*2)
            c.append(orig_c*2)
        #if source plate was in blue, the row/col are doubled and shifted to the right 
        elif pos == 1:
            r.append(orig_r*2)
            c.append(orig_c*2 +1)
        #if source plate was in yellow, the row/col are doubled and shifted to down
        elif pos == 2:
            r.append( orig_r*2 +1)
            c.append(orig_c*2 )
            
                #if source plate was in green, the row/col are doubled and shifted to down and right
        elif pos == 3:
            r.append(orig_r*2 +1)
            c.append(orig_c*2 +1)
    return list(p), r, c

def Undo_shuffle(order):
    ''' takes a shuffled order and reconstitutes the pinning order to place quadruplicates close to each others'''

    Red=[]
    Blue=[] 
    Yellow=[] 
    Green=[]
    for i in range(1,13+1):
        Red.append(order[order['Red']==i]['Destination'].item())
        Blue.append(order[order['Blue']==i]['Destination'].item())
        Yellow.append(order[order['Yellow']==i]['Destination'].item())
        Green.append(order[order['Green']==i]['Destination'].item())

    Reordered = np.stack([Red, Blue, Yellow, Green]).T
    Reordered = pd.DataFrame(data=np.hstack([np.arange(1,14).reshape(-1,1),Reordered]), columns=[ 'Destination','Red', 'Blue','Yellow','Green'])
    return(Reordered)

def plate_map_maker(Order,MapFilePath):
    '''creates a plate_map (a dict storing the gene name in 32x48 format) given the plate order and the original map text file'''
    ### creates the map of randomized genes on the plate
    plate_map=dict()
    for plate_num in range(1,14):
        plate_map[plate_num]=np.empty((32,48), dtype='<U16')


    #read txt file storing plate mapping and conver to 32x48 plate format
    file = open(MapFilePath,'r')#
    corpus=file.read().split('\n')
    plate_num=0

    for line in corpus:
        #print(line)
        if 'Plate' in line:
            plate_num+=1
            r=0
        else:
            c=0
            for gene in line.split(' ')[2:-1]: #[2:-1] gets rid of the two letters and the last letter flanking the row at each line
                shuffled_plates, shuffled_r, shuffled_c = shuffled_finder(plate_num, r, c, Order)
                for i in range(4):
                    plate_map[shuffled_plates[i]][shuffled_r[i]][shuffled_c[i]] = gene
                #plate_map[plate_num][r:r+3,c:c+3]=gene
                c+=1
            r+=1
    return plate_map

def Order2order(order2, order1):
    '''makes the table to obtain order2 from order1'''
    Red=[]
    Blue=[] 
    Yellow=[] 
    Green=[]
    for i in range(0,13):
       # print(order[order['Red']==i]['Destination'].item())
        #finds plate number in order2 and finds its location in order1
        #checks plate number in order2 and finds its location in order1
        Red.append(order1[order2.loc[i,'Red']==order1['Red']]['Destination'].item())
        Blue.append(order1[ order2.loc[i,'Blue']==order1['Blue'] ]['Destination'].item())

        Yellow.append(order1[ order2.loc[i,'Yellow']==order1['Yellow'] ]['Destination'].item())
        Green.append(order1[order2.loc[i,'Green']==order1['Green']]['Destination'].item())
    
    Reordered = np.stack([Red, Blue, Yellow, Green]).T
    Reordered = pd.DataFrame(data=np.hstack([np.arange(1,14).reshape(-1,1),Reordered]), columns=[ 'Destination','Red', 'Blue','Yellow','Green'])
    return(Reordered)

def Combine_Plates(red, blue, yellow, green):
    '''mimicks combining 4 plates into 1 on the rotor'''
    #makes sure we are combining same size arrays
    assert (red.shape == blue.shape) & (yellow.shape == green.shape) & (red.shape == green.shape)
   
    
    Combined= np.empty((red.shape[0]*2,red.shape[1]*2)).astype(object)

    Combined[::2,::2 ]= red
    Combined[::2,1::2 ]= blue
    Combined[1::2,::2 ]= yellow
    Combined[1::2,1::2 ]= green

    return(Combined)
    

