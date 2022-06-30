import os
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rnd

'''
- Written by Suleyman Bihi, Karen Pacho Dominguez, Stephen Solomou and Navraj Eari.
- Program simulates the kinetics of polymerisation as if it were a real life experiment.
- How to use:
     - Save this file into a folder anywhere on your computer.
     - Open as a .ipynb file.
     - Run this cell (information on how to change varibales detailed below.)
     - While the program is running, .txt files representing the system at that real life time (in the name) and        simulated time (first row of file) will be created.
     - Once the program has finished running, 2 graphs will be produced:
           - Probabilty against chain length.
           - Polydispersity against real time. 
     - Finished!
'''

# Variables to change:

RunTime = 3600 # Change in accordance to how long you want to run each simulation within the program for each combination of I and T, in real life seconds.
CFT = 10 # Change in accordance to how often you want files to be created, in real life seconds.

V = 0.1 # volume in L^-1
a = 1/(V*( 0.602 *(10**23) )) # DONT CHANGE 1/ (Vx Na)  used to convert between molecules and concentrations
M = [(5 * 10** (-3))/a] # number of monomers, the first number n in [n* (10 ** (-3*)) / a ] shoud be in mM 
I = [(1 * 10** (-6))/a, (1 * 10** (-5))/a] # number of intiators, /a converts conc. into no. molecules 
T = [0,(1* 10** (-6))/a,(1 * 10** (-5))/a] # program mainly works in molecules rather than in conc.

kxctn_dict = { 'k1.1' : 0.36 * (1/3600),
                'k2.1': 3.6 * (10**17) * (1/3600) , 
                'k3.1': 3.6 * (10**10) * (1/3600), # applies to 4.1
                'k3.2': 18 *  (10**8) * (1/3600),  # applies to 3.3, 4.2, 4.3 
                'k5.1': 3.6 * (10**11) * (1/3600), # applies to all 5.n reactions 
                } ### values for kinetic rates and keeps proportions for later on

#################################################################################################################
# original value for K2.1 (3.6 * (10**7) ) from pdf only produces a simulation that only has up to two different # chain lengths r0 and r1. values of k2.1 greater than (3.6 * (10**10)) allows the simulation to surpass        # chain lengths greater than 1 which allows the simulation to continue. since the orignal value of k2.1 doesnt  # work with the given volume, you can eithier change k2.1, or change the volume to a magnitude smaller than     # (0.1*(10**-5)). we changed k2.1 to (3.6 * (10**17)), and kept the volume as 0.1.
#################################################################################################################

# Global Variables.

stime = time.time() # Contains real life time at which the program is run.
t = 0 # Overall time of the system, in seconds.
tst = CFT # Create File Time, stores the initial value of CFT for later convenience.
RT = RunTime

rxctn_dict = { 'r1.1' : 1,
                'r2.1': 0,
                'r3.1': 0,
                'r3.2': 0,
                'r3.3': 0,
                'r4.1': 0,
                'r4.2': 0,
                'r4.3': 0,
                'r5.1': 0,
                'r5.2': 0,
                'r5.3': 0,
                'r5.4': 0,} ### values for reaction rates
 
TRn = np.array([0 for i in range( 100)]) # initialises an array so big that the index is never out of range
Px = np.array([0 for i in range( 100)]) # polymer species

## Radical species
_Rn = np.array([0 for i in range( 100)]) # format --> [R0, R1, R2, ..., Rn, Rm, Rs,]
# index _Rn[0] corresponds to R0 radical, index [1] is chain length of r0+1 
# all species arrays with chain lengths follow the same format
_TRn =np.array([0 for i in range( 100)])
_Rn_TRn = np.array([[0 for i in range(100)] for j in range(100)])
        

# keeps species in one place so its easier to access and change the number of molecules later on

class Rates:

        # Calculates the rate of reaction for all reactions 1.1 - 5.4. 
        
    def __init__(self, sList ,rxdicts = rxctn_dict , kxdicts = kxctn_dict ):

        #initialises class rate with the following attributes.
        self.I = sList[1][0]   # takes input for conc. of initiators
        self.T = sList[2][0]   # takes input for raft agent concentration 
        self.M = sList[0][0]   # takes input for monomer conc. 
        self.sList= sList            
        self.rxdicts = rxdicts  # Rx values
        self.kxdicts = kxdicts  # Kinetic constants kx
        self.m = 0 
        self.s = 0
        
        
    def rconstants(self):
        # calculates rates 
        # inputs: 
            # k constants from kxctn_dict
            # concentration of the species (uses conversion factor - a - to convert from number of molecules to                concentration)
        # outputs:
            # updated rxctn_dict with values = the rates of reaction 1.1 - 5.4
      
        self.rxdicts['r1.1'] = 2 * self.I * self.kxdicts['k1.1'] * a # calculates k1.1[i]*2 = r.1.1     
       
        self.rxdicts['r2.1'] = self.M *  self.kxdicts['k2.1'] * a*a * np.sum(self.sList[5]) # k2.1[M][.Rn]

        self.rxdicts['r3.1'] = self.T * self.kxdicts['k3.1'] *  a*a * np.sum(self.sList[5]) # k3.1[T][.rn]
        # np.sum adds all the chain lengths that exist for that species
        self.rxdicts['r3.2'] = self.kxdicts['k3.2'] * a * np.sum(self.sList[6]) # k3.2 [.Trn]   
        self.rxdicts['r3.3'] = self.rxdicts['r3.2'] # both 3.2 and 3.1 have the same rate equation as they have                                                        the same reactants
        
        self.rxdicts['r4.1'] = self.kxdicts['k3.1'] *  a*a   *np.sum(self.sList[5]) * np.sum(self.sList[3])              # k3.1[.Rn][TRm] 
        # Note k for 4.1 == k 3.1
        self.rxdicts['r4.2'] = self.kxdicts['k3.2'] * a *np.sum(self.sList[7][0]) # k3.2 = k4.2 = k4.3 rate =            k3.2 [Rn.TRm] 
        self.rxdicts['r4.3'] = self.rxdicts['r4.2'] # since R4.2= r4.3 both have same rate equation
       
        if np.sum(self.sList[5]) != 0 : # only calculates s and m once radicals exist in the system 
            self.s =  int(probs.chainP(Species_List[5]))  # used to calculate the [Rm] in r5.1
            self.m = int(probs.chainP(Species_List[5]))   # used to calculate [Rs] by giving position of latest                                                              reacted _Rs chain in r5.4       
             
        self.rxdicts['r5.1'] = self.kxdicts['k5.1'] * a*a * np.sum(self.sList[5])* (np.sum(self.sList[5]) -              self.sList[5][self.m]) # [.Rn][.Rm]  a2 conerts to concentration
        self.rxdicts['r5.2'] = self.rxdicts['r5.1'] 
        self.rxdicts['r5.3'] = self.kxdicts['k5.1'] * a*a  * np.sum(self.sList[7][0]) * (np.sum(self.sList[5]) -         self.sList[5][0]) # [Rn.Trm][.Ro]

        self.rxdicts['r5.4'] = self.kxdicts['k5.1'] *  a*a  * np.sum(self.sList[7][0]) * (np.sum(self.sList[5])          - self.sList[5][self.s]) # [Rn.Trm][.Rs]
       
        reaction_chosen= probs.reactionchooser(self.rxdicts) # holds value for which reaction is chosen 
        probs.relativeP(reaction_chosen) # calls method that calculates which chain lengths will be reacted
 
class probs:

    # These 2 functions are for choosing one out of Reactions 1.1 -5.4 that you are evolving

    def probability(dictionary): 

        # this function finds the probability of each reaction occuring from the rates of reaction
        # input - A dictionary with the key corresponding to each of the potential reactions that can occur and            the value having that the corresponding rate of reaction
        # output - A dictionary with the key corresponding to each of the potential reactions that can occur and           the value having the probability of that reaction to occur

        rtot2 = 0  
        for i in dictionary:  
            rtot2 = rtot2 + dictionary[i] # sums up all the values of the rates for all the reactions
        for i2 in dictionary:  
            dictionary[i2] = (dictionary[i2])/rtot2 # divides each rate value by the sum of rates to get a                                                             probability of each reaction occuring
        return dictionary # gives out a dictionary 

    def reactionchooser(rxctn_dict):

        # this function takes the dictionary with the probabilities and then chooses a random reaction 
        # input - dictionary created in method probability with the key as the reaction and the value as the                       probability
        # output - gives you a random reaction of 1.1 - 5.4 depending on its probability

        probs2 = probs.probability(rxctn_dict) # creates a new dictionary that has the reaction as the key and                                                    the probability as the value
        listofrates = list(probs2.keys()) # turns the keys of the rates into a list so that they can randomly be                                             chosen
        listofprobabilities = list(probs2.values()) # turns the probabilities into a list so that they can be                                                          the probabilities of each key
        chooser = rnd.choice(listofrates, size = 1, replace = True, p =listofprobabilities ) # picks a random            reaction using the probability and gives out a reaction
        return chooser # gives out a random reaction
    
    def chainP(Rn):

        # this function picks a random radical length given a specific reaction has been chosen
        # input - takes an array of all the different lengths being the index and the number of that length as                     the value
        # output - gives out a random length of a radical in the reaction chosen
         
        probabilities = Rn/np.sum(Rn) # finds the probabilities of each value by dividing each value by the                                              sum
        newdict = (dict(enumerate(probabilities))) # makes a dictionary with the key being the index and the                                                          value being the probability
        listoflengths = list(newdict.keys()) # turns the keys of the lengths into a list so that they can                                                       randomly be chosen
        listforprobabilities = list(newdict.values()) # turns the probabilities into a list so that they can be                                                          the probabilities of each key
        randlength = rnd.choice(listoflengths, size = 1, replace = True, p =listforprobabilities ) # picks a             random length with the given probability
        return randlength # gives out a length of a radical in the specific reaction
    
    def relativeP(chooser):

      # this function takes the random reaction and then accesses the species list at different index positions          depending on the reactants in that reaction, then proceeds to evolve the system.
      # input - the randomly chosen reaction from 1.1 - 5.4
      # output - gives a index postion of chain length choosen by method chainp
      # Each reaction has different reactants so needs to access different indexes of the species list
        
        if chooser == "r1.1":
            evolve.Initiation(model,chooser) # calls the function to evolve it using initiation
            
        elif chooser == "r2.1" or "r3.1": # if 3.1 is chosen propagation() will call pre_eq() to evolve it using                                              pre_eq.
            a = probs.chainP(Species_List[5]) # finds the _Rn radical by using indexing of arrays in the species                                                 list
            evolve.Propagation(model,chooser,a) # calls the function to evolve it using propagation
    
        elif chooser == "r3.2" or "r3.3":
            b = probs.chainP(Species_List[6]) # finds the _TRn radical by using indexing of arrays in the                                                        species list
            evolve.Pre_equilibrium(model,b) # calls the function to evolve it using pre equilibrium
            
        elif chooser == "r4.1":
            c = probs.chainP(Species_List[5]) # finds the _Rn radical by using indexing of arrays in the species                                                 list
            d = probs.chainP(Species_List[6]) # finds the _TRm radical by using indexing of arrays in the                                                        species list
            evolve.Pre_equilibrium(model,chooser, c,d) # calls the function to evolve it using equilibrium with                                                           all the reactants
            
        elif chooser == "r4.2" or "r4.3" or "r5.3":
            e = probs.chainP(Species_List[7][0]) # finds and accesses the first row of _Rn _TRn which                                                               corrosponds to the Rn section of that species 
            f = probs.chainP(Species_List[7][1]) # finds and accesses row for TRm radical, second row of _Rn _TRn
            evolve.Pre_equilibrium(model,chooser, e,f) # calls the function to evolve it with all the reactants
        elif chooser == "r5.1" or "r5.2":
            g = probs.chainP(Species_List[5]) # finds the _Rn radical by using indexing of arrays in the species                                                 list
            h = probs.chainP(Species_List[5]) # finds the _Rm radical 
            evolve.Pre_equilibrium(model,chooser, g,h) # calls the function to evolve it with all the reactants
        elif chooser == "r5.4":
            i = probs.chainP(Species_List[7][0]) # finds and accesses the row of the radical Rm
            j = probs.chainP(Species_List[7][1]) # finds and accesses row for TRm radical
            k = probs.chainP(Species_List[5]) # finds the _Rs radical by using indexing of arrays in the species                                                 list
            evolve.Pre_equilibrium(model,chooser, i,j,k) # calls the function to evolve it with all the reactants
            
class evolve(Rates):

    # The evolving of the system during the simulation as reactions occur
    # Inputs: Species list, probabilities
    # Outputs: Updated species list


    def __init__():
        super().__init__()

        # Class evolve inherits all the variables from the class Rates
        # Inputs: Variables from class rates
        # Outputs: Update molecules from species list

    def Initiation(self, reaction ):

        # Formation of radicals using the initiators
        # Inputs: Initiators
        # Outputs: radicals

        # I --> 2_R0
        self.sList[5][0]+=2 # add 2 to the position 0 in _Rn  in species list 
        self.I-= 1 # minuses 1 from the index position i in the species list, to reduce the number                                    intiators
        
    def Propagation(self, chooser, p=0):

        # Formation of living polymer radicals with various chain lengths 
        # Inputs: radicals
        # Outputs: Longer lengths of living polymer radicals

        #_Rn + M --> _Rn+1
        if chooser == 'r2.1':
            self.sList[5][(p+1)] +=1 # add 1 to the position p+1 in _Rn  in species list 
            self.M -=1 # minuses 1 from the index position M 
        else:
            evolve.Pre_equilibrium(self,chooser , p) # if chooser is not these reactions above, it shall move                                                           onto next set of reactions     

    def Pre_equilibrium(self, chooser, p=0):

        # Formation of the _TRn which formed from the radical and the RAFT agent
        # Inputs: living polymer radicals
        # Outputs: formation of _TRn

        #_Rn +T -->_TRn
        if chooser == 'r3.1':
            self.sList[5][p]-=1 # minuses one to index position to p in _Rn
            self.sList[6][p]+=1 # adds one to index position to p in _TRn
            self.T -= 1 # minuses one from the T in the species list
        #_TRn --> _Rn +T       
        elif chooser== 'r3.2':
            self.sList[6][p]-=1 # minuses one to index position to p in _TRn
            self.sList[5][p]+=1 # adds one to index position to p in _Rn
            self.T +=1 # adds one from the T in the species list
        #_TRn -->_R0 + TRn           
        elif chooser == 'r3.3':
            self.sList[6][p]-=1 # minuses one to index position to p in _TRn
            self.sList[5][0]+=1 # adds one to index position to 0 in _Rn
            self.sList[3][p]+=1 # minuses one to index position to p in TRn
        else:
            evolve.Core_Equilibrium(self,chooser) # if chooser is not these reactions above, it shall move onto                                                      next set of reactions

    def Core_Equilibrium(self, chooser,n=0, m=0):

        # The formation of the adduct (_Rn_TRn) living polymer radicals
        # Inputs: living radical polymer
        # Outputs: adduct living polymer

        #_Rn + TRm --> _Rn_TRm 
        lists= self.sList[7]
        if chooser == 'r4.1':
            self.sList[5][n]-=1 # minuses 1 from position n from _Rn
            self.sList[3][m]-=1 # minuses 1 from position m from TRn
            self.sList[7][0][n]+=1 # adds 1 from position n from _Rn_TRm
            self.sList[7][1][m]+=1 # adds 1 from position m from _Rn_TRm

        #_Rn_TRm --> _Rn+_TRm
        elif chooser == 'r4.2':
            self.sList[7][0][n]-=1 # minuses 1 from position n from _Rn_TRm
            self.sList[7][1][m]-=1 # minuses 1 from position m from _Rn_TRm
            self.sList[5][n]+=1 # adds one to index position to n in _Rn
            self.sList[6][m]+=1 # adds one to index position to m in TRn 

        # _Rn_TRm --> TRn+_Rm 
        elif chooser == 'r4.3':
            self.sList[7][0][n]-=1 # minuses 1 from position n from _Rn_TRm
            self.sList[7][1][m]-=1 # minuses 1 from position m from _Rn_TRm
            self.sList[5][m]+=1 # adds one to index position to m in _Rm 
            self.sList[6][n]+=1 # adds one to index position to n in TRn 
        else:
            evolve.termination(self,chooser,n,m) # if chooser is not these reactions above, it shall move onto                                                      next set of reactions
    
    def termination(self,chooser,n,m,s=0):

        # Terminates the reaction by creating polymers
        # Inputs: living radical polymers, index positions n, m and s for radical chains 
        # Outputs: dead polymers

        lists = self.sList[4]  
        if chooser == 'r5.1' : 
             # _Rn + _Rm --> Pn+m
            self.sList[5][n] -= 1 # minuses 1 from position n from _Rn
            self.sList[5][m] -= 1 # minuses 1 from position m from _Rn 
            self.sList[4][(n+m)] += 1 # adds one to index position n+m in Px
           
        elif chooser == 'r5.2':
            # _Rn + _Rm --> Pn+ Pm
            self.sList[5][n] -= 1 # minuses 1 from position n from _Rn
            self.sList[5][m] -= 1 # minuses 1 from position m from _Rn
            self.sList[4][n]+= 1  # adds 1 to index postion from _Rn to Px  
            self.sList[4][m] += 1 # adds 1 to index postion from _Rn to Px  
                
        elif chooser == 'r5.3':
            # _Rn_TRn +_R0 --> Pn+m
            self.sList[5][0]-= 1 # minuses 1 from position 0 from _Rn  
            self.sList[7][0][n]-= 1 # minuses 1 from position rn in _Rn_TRM
            self.sList[7][1][m]-= 1 # minuses 1 from position trm in _Rn_TRM
            self.sList[4][(n+m)] +=1  # adds to 1 to the index position Pnm in Px
               
        elif chooser == 'r5.4':
            # _Rn_TRm  + Rs --> Pm+n+s 
            self.sList[5][s] -= 1 # minuses 1 from position s in _Rn 
            self.sList[7][0][n]-= 1 #-1 from position rn in _Rn_TRM
            self.sList[7][1][m]-= 1 #-1 from position trm in _Rn_TRm
            self.sList[4][(n+m+s)] +=1  # adds 1 to index position Pnm in Px  

def LRFileCreator(Species_List, tst, t, CFT):

    # Function to be run every CFT seconds of real life time.
    # Creates a .txt file which represents the system at the 10th real life second.
    # And contains information about the living radicals of length Rn in the system.

    # Inputs: Species_lists (contains living radicals of length Rn), tst, t, CFT (time related.)
    # Outputs: File containing information about the living radicals in the system. 

    LRFileCreator.LR = Species_List[-3]
    LRFileCreator.LR = LRFileCreator.LR[LRFileCreator.LR != 0]
    # Stores an array contained within Species_List in a variable, then removes all occurences of 0 for later          conveince.
    # This array contains the length and number of all Living radicals with length Rn in the system.
    # Its formated as such, [0, 3, 7, 11, 20] (example numbers.)
    # The index position of a value is its chain length, and the value is the number of radicals with that chain       length.
    LRFileCreator.TotLR = np.sum(LRFileCreator.LR)
    # Stores the total number of radicals in the system.
    LRFileCreator.FractionLRArray = np.round((LRFileCreator.LR / LRFileCreator.TotLR), 4)
    # Creates an array, which is the same array as LRFileCreator.LR, but each value is divided by the a                constant, that being the total number of radicals in the system.
    # Therefore, this array contains the fraction of living radicals of length Rn (index postion) for all              radicals in the system. To put simply, the probabilty of finding a radical for each length Rn.
    LRArray = np.array(["Sim Time:", t, "(s)"])
    # Initialises an array, where the first row displays the simulation time.

    for i in range(len(LRFileCreator.FractionLRArray)):
        x = np.array([i, LRFileCreator.FractionLRArray[i], LRFileCreator.LR[i]])
        LRArray = np.vstack((LRArray, x))
        # Creates an array that is formated as follows (example):

        # Sim time: 23.645 (s)
        # 0.0       0.00   0
        # 1.0       0.22   22
        # 2.0       0.66   66
        # 3.0       0.12   12

        # etc.....................
        # The first row displays the simulation time.
        # The first column represents the chain length (n).
        # The second column represents the probabilty (Pn) of finding a radical with that chain length (notice             how the sum of the second column will always = 1).
        # The third column represents the number of living radicals with that chain length. 

    LRFileCreator.SysNum = "system_" + (str(round((tst-CFT)/10, 4))).rstrip('0').rstrip('.').zfill(5)
    np.savetxt(LRFileCreator.SysNum + ".txt", LRArray, fmt="%s")
    # Saves the LRArray mentioned before into a file. 
    # The name of the file is formated as the following: "system_00001"
    # This represents the system at the 10th second in real life time.

    return

Y = [] # An empty list to append calculated values of W 

def WDist():

   # Calculates the width distribution of each file
   # Inputs: An array where each element is a probabilty, and its index postion is its chain length
   # Outputs: Width distribution list

    global Y

    n_mean = (np.dot(LRFileCreator.LR, np.arange(len(LRFileCreator.LR)))) / LRFileCreator.TotLR 
    # Calculates the average chain length
    W = 0
    for i in range(len(LRFileCreator.FractionLRArray)): # for loop to run through the array and calculate W
        W += ((i - n_mean)**2) * LRFileCreator.FractionLRArray[i] # Calculates width distribution
    Y.append(W) # Append values of W to an empty list, Y

    return
    
def TimeUpdater(rxctn_dict):

    # Function which updates the time every time the system "evolves".
    # Inputs: rxctn_dict (dictionary with valueso of rates.)
    # Outputs: Updates simulation time.

    global t

    rtot = sum(rxctn_dict.values())
    # Calcualtes the total rate of reaction by summing all the values in the rxctn dictionary. 
    tau = round((1 / rtot) * np.log(1/random.uniform(0, 1)),3)
    # Calculates the change in time every time the system "evolves".
    t += tau
    # Adds this change in time to the total time.
    return


def TenSecondCheck(t, stime, CFT):

    # Checks if CFT real life seconds has passed, and updates system/program accordingly.
    # Inputs: t (simulation time), stime (real life time the simulation started), CFT.

    global tst

    etime = time.time() - stime
    # Calculates difference in real life time from when the program was first run, to now.
    if etime >= tst:
        tst += CFT
        LRFileCreator(Species_List, tst, t, CFT)
        WDist()
        # Runs LRFileCreator every time CFT seconds of real life time has passed.
        # The global varible tst then increases by CFT each time, ready for when the CFTth real life second                occurs.
    return

def Grapher():

    # this function takes the text files saved and draws a graph with them
    # input - the text files with the first column as the length of each chain and the second column as the           probability of each length, list Y, RunTime, and CFT.
    # output - a graph of the probability of each length against the length of the chain, and                                   polydispersity against real time.
 
    # finds and creates all the values to be used in the graph
    allfiles = sorted(glob.glob('system_*.txt')) # finds all the system file names that increase in                  number over time and sorts them in acsending order
    numberoffiles =(len([name for name in os.listdir(os.curdir) if os.path.isfile(name)])-1) # finds the number      of text files saved and takes away 1 (due to .ipynb file)
    roundedfiles = round((numberoffiles/5), 0) # calculates equal intervals between the number of files, and         rounds them to an integer so that the splicing of files can work
    chosenfiles = allfiles[1::int(roundedfiles)] # Splices every n files to draw a graph. Also turns the             whole number into an integer so the splicing works
    a = [] # creates an empty list
    for i in chosenfiles:
        data = np.loadtxt(i, skiprows=1) # loads the text files to draw a graph
        a.append(data) # adds the contents of the chosen file to an empty list to draw the graph.

    # Probabilty against Chain Length graph plotting
    docs1 = ["-ro", "-bo", "-go", "-mo", "-ko"] # used to differentiate different times with different                                                             colours
    for index, i in enumerate(a): 
        x = i[:, 0] # x axis is the Chain length
        y = i[:, 1] # y axis is the probability of that corresponding chain length
        plt.plot(x, y, docs1[index] , label = chosenfiles[index] , linewidth = 1) # draws a graph with different                  times on the same axis so that they can be compared
    plt.xlabel('Chain Length (n)') 
    plt.ylabel('Probability')
    plt.title('Probabilty against Chain Length '+ title) 
    plt.tight_layout
    plt.legend()
    plt.savefig('Probability Graph_'+ title +'.png')
    plt.show()
 
   
    # Plots a graph of polydispersity against time.
    x_ = np.linspace(0, RunTime, int(RunTime/CFT)) # Creating a list for the x-axis which represents time
    plt.plot(x_,Y, 'm',  linewidth = 3) # plots graph  
    plt.xlabel('Time(s)') 
    plt.ylabel('W') 
    plt.title('Polydispersity against Time '+ title)
    plt.tight_layout
    plt.savefig('Polydisperity Graph_'+ title +'.png')
    plt.show() # shows the graph as an output under the cell which it was run from 
    
    return
 

 
Species_List=[M, I, T, TRn, Px, _Rn, _TRn, _Rn_TRn]
Species_List[1][0]= np.array(I[0]) 
Species_List[2][0]=np.array(T[0])


def run():
    # runs the simulation
    # input : time, reaction dictionary
    #output: graphs x2  (1 set for each combination)
    while (time.time() - stime) <= RunTime:    
       
    # While loop which continuously runs the following code to: 
    # Initializes model with all the atributes of class rates.
        model.rconstants() # Calls method to caluculate the rates and evolves the system.
        TimeUpdater(rxctn_dict) # Updates simulation time.
        TenSecondCheck(t, stime, CFT) # Checks if CFT number of seconds has passed.
    print("Simulation Complete! :)")
    Grapher()# Calls function to produce the Probability and Polydispersity Graph.
    
    # Loop breaks when elapsed time has reached run time


for j in range(len(I)): # for loop runs simulation for all combinations of [I] and [T]
    Species_List[1][0]= np.array(I[j]) 
    model = Rates(Species_List) 
    for i in range(len(T)):
        Species_List=[M, I, T, TRn, Px, _Rn, _TRn, _Rn_TRn]
        Species_List[2][0] = np.array(T[i])
        model = Rates(Species_List)
        title = str(i) + str(j) # used to save the graphs i is the index of I, j is index of T conc. value 
        stime = time.time()
        Runtime = RT # resets runtime and all variables related to time
        tst = CFT        
        Y = []
        run()


# To obtain the lowest possible polydispersity, make k2.1 (propagation) as low as possible, and have an            appropriate value of k1.1 (initiation) to complement the other rates. A really high value of k1.1 will produce   many R0's in the system and therefore also lower the polydispersity because the other reactions 2.1 - 5.4 will   be much less likely to occur because the probabilty of k1.1 is 1. k2.1 has been set as (3.6*(10**17)) but this   value does not give the smallest possible dispersity but a higher polydispersity,therefore illustrating the      results of a more complete simulation.

# When comparing the standard radical polymerisation against RAFTS polymerisation, polydispersity is very          similar but there's a slight increase shown by the plateau values of the polydispersity; 12 (Standard) as compared to 11.5 (RAFTS). 

# When you vary the values of I and T (from T = 0 , 10^-3 and 10^-2, and from I= 10^-2 and 10^-3) in different combinations, the polydispersity is the highest , with the smallest values of I and T (these were done with values of the pdf but altering the value of k2.1 to (3.6*10^17)and the volume 0.1 L). While increasing I and T in different combinations decreases the polydispersity, it has a very small change