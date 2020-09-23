import numpy as np
from numpy import matlib
from psychopy import visual, core, event
import time


class SSVEP_stimuli(object):
    
    def __init__(self, mywin= visual.Window([800, 600], fullscr=True, monitor='testMonitor',units='deg', waitBlanking = False), trialdur = 3.0, numtrials=4, waitdur=2):
        
        self.mywin = mywin
        
        # colour for psychopy
        self.white = [1, 1, 1]
        self.black = [-1, -1, -1]
        self.red = [1, -1, -1]
        
        # frequency = 10Hz -- pattern 1 -- position:top
        self.pattern1_f0 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0, pos=[0, 400], size = 300,
                        color=self.white, colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        # frequency = 10Hz -- pattern 2 -- position:top                  
        self.pattern2_f0 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0, pos=[0, 400], size = 300,
                        color=self.black, colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)
        # frequency = 12Hz -- pattern 1 -- position:right
        self.pattern1_f1 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0, pos=[800, 0], size = 300,
                        color=[1,1,1], colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        # frequency = 12Hz -- pattern 2 -- position:right
        self.pattern2_f1 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0, pos=[800, 0], size = 300,
                        color=[-1,-1,-1], colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)
        # frequency = 15Hz -- pattern 1 -- position:left
        self.pattern1_f2 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0, pos=[-800, 0], size = 300,
                        color=[1,1,1], colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        # frequency = 15Hz -- pattern 2 -- position:left
        self.pattern2_f2 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0, pos=[-800, 0], size = 300,
                        color=[-1,-1,-1], colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)
        # frequency = 30Hz -- pattern 1 -- position:bottom
        self.pattern1_f3 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0, pos=[0, -400], size = 300,
                        color=[1,1,1], colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        # frequency = 30Hz -- pattern 2 -- position:bottom
        self.pattern2_f3 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0, pos=[0, -400], size = 300,
                        color=[-1,-1,-1], colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)

        self.fixPos = [self.pattern1_f0.pos, self.pattern1_f1.pos, self.pattern1_f2.pos, self.pattern1_f3.pos]

        self.fixation = visual.GratingStim(win=self.mywin, color = self.red , size = 10, sf=0, colorSpace='rgb', units='pix')

        # frame array for 10Hz
        self.frame_f0 = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1,1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]

        # frame array for 12Hz
        self.frame_f1 = [1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1]

        # frame array for 15Hz
        self.frame_f2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]

        # frame array for 30Hz
        self.frame_f3 = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        
        self.trialdur = trialdur
        self.numtrials = numtrials
        self.waitdur = waitdur

        # randomise sequence
        self.nBox = 4
        self.capBox = int(self.numtrials/self.nBox)
        self.aBox = np.arange(self.nBox)
        self.unshuffled = np.matlib.repmat(self.aBox, self.capBox, 1)
        self.randperm = np.random.permutation(self.numtrials)
        self.Boxes = self.unshuffled.ravel()
        self.Boxes = self.Boxes[self.randperm]
        print (self.Boxes)

    def stop(self):
        self.mywin.close()
        core.quit()
        
    def start(self):

        self.count = 0
        
        # Loop through all trials
        while self.count < self.numtrials:
            
            self.fixation.pos = (self.fixPos[self.Boxes[self.count]])
            self.fixation.setAutoDraw(True)

            self.Trialclock = core.Clock()
            # start_sample = self.client.last_sample

            # Loop through the required trial duration
            while self.Trialclock.getTime() < self.trialdur:
                #draws square and fixation on screen.
                self.fixation.setAutoDraw(True)
                
                for frameN in range(len(self.frame_f0)):
                    if self.frame_f0[frameN] == 1 :
                        self.pattern1_f0.draw()      
                    if self.frame_f0[frameN] == -1 :
                        self.pattern2_f0.draw()
                    if self.frame_f1[frameN] == 1 :
                        self.pattern1_f1.draw()      
                    if self.frame_f1[frameN] == -1 :
                        self.pattern2_f1.draw()
                    if self.frame_f2[frameN] == 1 :
                        self.pattern1_f2.draw()      
                    if self.frame_f2[frameN] == -1 :
                        self.pattern2_f2.draw()
                    if self.frame_f3[frameN] == 1 :
                        self.pattern1_f3.draw()      
                    if self.frame_f3[frameN] == -1 :
                        self.pattern2_f3.draw()
                    self.mywin.flip()
                
                
            #clean black screen off
            self.mywin.flip()
            self.fixation.setAutoDraw(False)
            #wait certain time for next trial
            core.wait(self.waitdur)
            #reset clock for next trial
            self.Trialclock.reset()    
            #count number of trials
            print("Trial %d Complete" % self.count)
            self.count+=1
            
if __name__ == "__main__":

    stimuli = SSVEP_stimuli()
    stimuli.start()
    stimuli.stop()
