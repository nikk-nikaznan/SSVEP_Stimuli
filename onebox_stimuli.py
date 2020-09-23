import numpy as np
from psychopy import visual, core, event
import time

# set class frequency here -- 10Hz, 12Hz, 15Hz, 30Hz
class_freq = 30

class SSVEP_stimuli(object):
    def __init__(self, class_freq, mywin= visual.Window([800, 600], fullscr=True, monitor='testMonitor',units='deg', waitBlanking = False), trialdur = 3.0, numtrials=10, waitdur=2):
        
        self.mywin = mywin
        
        # colour for psychopy
        self.white = [1, 1, 1]
        self.black = [-1, -1, -1]
        self.red = [1, -1, -1]
        
        self.pattern1 = visual.GratingStim(win=self.mywin, name='pattern1',units='cm', 
                        tex=None, mask=None,
                        ori=0, pos=[0, 0], size=10, sf=1, phase=0.0,
                        color=self.white, colorSpace='rgb', opacity=1, 
                        texRes=256, interpolate=True, depth=-1.0)
        self.pattern2 = visual.GratingStim(win=self.mywin, name='pattern2',units='cm', 
                        tex=None, mask=None,
                        ori=0, pos=[0, 0], size=10, sf=1, phase=0,
                        color=self.black, colorSpace='rgb', opacity=1,
                        texRes=256, interpolate=True, depth=-2.0)

        self.fixation = visual.GratingStim(win=self.mywin, color = self.red , size = 10, sf=0, colorSpace='rgb', units='pix')

        # frame array for 10Hz
        self.frame_f0 = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1,1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]

        # frame array for 12Hz
        self.frame_f1 = [1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1]

        # frame array for 15Hz
        self.frame_f2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]

        # frame array for 30Hz
        self.frame_f3 = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        
        if class_freq == 10:
            self.frame = self.frame_f0
        if class_freq == 12:
            self.frame = self.frame_f1
        if class_freq == 15:
            self.frame = self.frame_f2
        if class_freq == 30:
            self.frame = self.frame_f3
        
        self.trialdur = trialdur
        self.numtrials = numtrials
        self.waitdur = waitdur

    def stop(self):
        self.mywin.close()
        core.quit()
        
    def start(self):

        self.count = 0
        
        # Loop through all trials
        while self.count < self.numtrials:

            self.fixation.setAutoDraw(True)

            self.Trialclock = core.Clock()
            # start_sample = self.client.last_sample

            # Loop through the required trial duration
            while self.Trialclock.getTime() < self.trialdur:
                #draws square and fixation on screen.
                self.fixation.setAutoDraw(True)
                
                for frameN in range(len(self.frame)):
                    if self.frame[frameN] == 1 :
                        self.pattern1.draw()      
                    if self.frame[frameN] == -1 :
                        self.pattern2.draw()
                    self.mywin.flip()
                
                
            #clean black screen off
            self.mywin.flip()
            #wait certain time for next trial
            core.wait(self.waitdur)
            #reset clock for next trial
            self.Trialclock.reset()    
            #count number of trials
            print("Trial %d Complete" % self.count)
            self.count+=1
            
if __name__ == "__main__":

    stimuli = SSVEP_stimuli(class_freq)
    stimuli.start()
    stimuli.stop()
