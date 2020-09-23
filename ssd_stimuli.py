import ssd_predict
import argparse
import cv2
import numpy as np
from numpy import matlib
from psychopy import visual, core, event
import time
from PIL import Image
import torch
from torch.autograd import Variable

class SSVEP(object):
    
    def __init__(self, mywin= visual.Window([800, 600], fullscr=True, monitor='testMonitor',units='deg', waitBlanking = False),trialdur = 3, numtrials=6, waitdur=2):

        self.mywin = mywin
        self.myStim = visual.GratingStim(win=self.mywin, pos=[0,0], units = 'norm')
        # colour for psychopy
        self.white = [1, 1, 1]
        self.black = [-1, -1, -1]
        self.red = [1, -1, -1]

        self.fixation = visual.GratingStim(win=self.mywin, color = self.red, size = 10, sf=0, colorSpace='rgb', units='pix')

        # frame array for 10Hz
        self.frame_f0 = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1,1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
        # frame array for 12Hz
        self.frame_f1 = [1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1]
        # frame array for 15Hz
        self.frame_f2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
        
        self.trialdur = trialdur
        self.numtrials = numtrials
        self.waitdur = waitdur

        self.nBox = 3
        self.numChan = 9
        self.sample_rate = 500
 
        self.capBox = int(self.numtrials/self.nBox)
        self.aBox = np.arange(3)
        self.unshuffled = np.matlib.repmat(self.aBox, self.capBox, 1)
        self.randperm = np.random.permutation(self.numtrials)
        self.Boxes = self.unshuffled.ravel()
        self.Boxes = self.Boxes[self.randperm]
        print (self.Boxes)

    def initBoxes(self):
        self.pattern1_f0 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0,
                        color=self.white, colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        self.pattern2_f0 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0,
                        color=self.black, colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)

        self.pattern1_f1 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0,
                        color=self.white, colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        self.pattern2_f1 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0,
                        color=self.black, colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)
        
        self.pattern1_f2 = visual.GratingStim(win=self.mywin, name='pattern1',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0.0,
                        color=self.white, colorSpace='rgb', opacity=0.8, 
                        texRes=256, interpolate=True, depth=-1.0)
        self.pattern2_f2 = visual.GratingStim(win=self.mywin, name='pattern2',units='pix', 
                        tex=None, mask=None,
                        ori=0, sf=1, phase=0,
                        color=self.black, colorSpace='rgb', opacity=0.8,
                        texRes=256, interpolate=True, depth=-2.0)

    
    def start (self):
        cap = cv2.VideoCapture('/home/nikkhadijah/Videos/video.mp4')
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.sframe = [(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))]
        self.sframe = self.sframe[0]

        self.fixCount = 0
        self.count = 0
        
        while(cap.isOpened()):
        
            while self.count < self.numtrials:
                SSVEP.initBoxes(self)
                ret, img = cap.read()
                
                if ret:
                    im = Image.frombytes("RGB", (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), img.tostring(), "raw", "BGR", 0, 1)
                    
                    self.myStim.setTex(im)
                    self.myStim.draw()
                    self.mywin.flip()

                    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

                    retdet = ssd_predict.predict(img)
                    
                    pt = retdet[0]
                    idxcls = retdet[1]

                    if len(pt) > 0:
                        pt = np.vstack(pt)
                        

                        for ndet in range (0, (len(pt))):
                            
                            # newpoint -- converting opencv format to psychopy
                            newPt = SSVEP.newPoint(self, pt[ndet])
                            
                            # assign the positions and the boxes for stimuli based on new points calculated 
                            if idxcls[ndet] == 0 or idxcls[ndet] == 3:
                                self.pattern1_f0.pos = ((newPt[2]+newPt[0])/2), (newPt[1]+newPt[3])/2
                                self.pattern1_f0.size = (abs(newPt[2]-newPt[0])), (abs(newPt[3]-newPt[1]))
                                self.pattern2_f0.pos = ((newPt[2]+newPt[0])/2), (newPt[1]+newPt[3])/2
                                self.pattern2_f0.size = (abs(newPt[2]-newPt[0])), (abs(newPt[3]-newPt[1]))
                            
                            if idxcls[ndet] == 1 or idxcls[ndet] == 4:

                                self.pattern1_f1.pos = ((newPt[2]+newPt[0])/2), (newPt[1]+newPt[3])/2
                                self.pattern1_f1.size = (abs(newPt[2]-newPt[0])), (abs(newPt[3]-newPt[1]))
                                self.pattern2_f1.pos = ((newPt[2]+newPt[0])/2), (newPt[1]+newPt[3])/2
                                self.pattern2_f1.size = (abs(newPt[2]-newPt[0])), (abs(newPt[3]-newPt[1]))
                                
                            if idxcls[ndet] == 2 or idxcls[ndet] == 5:

                                self.pattern1_f2.pos = ((newPt[2]+newPt[0])/2), (newPt[1]+newPt[3])/2
                                self.pattern1_f2.size = (abs(newPt[2]-newPt[0])), (abs(newPt[3]-newPt[1]))
                                self.pattern2_f2.pos = ((newPt[2]+newPt[0])/2), (newPt[1]+newPt[3])/2
                                self.pattern2_f2.size = (abs(newPt[2]-newPt[0])), (abs(newPt[3]-newPt[1]))

                        fixPos = [self.pattern1_f0.pos, self.pattern1_f1.pos, self.pattern1_f2.pos]
                        
                        self.fixation.pos = (fixPos[self.Boxes[self.count]])
                        
                        self.fixation.setAutoDraw(True)
                        ret, img = cap.read()
                        im = Image.frombytes("RGB", (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), img.tostring(), "raw", "BGR", 0, 1)
                        
                        self.myStim.setTex(im)
                        self.myStim.draw()
                        self.mywin.flip()
                        core.wait(2.0)

                        self.Trialclock = core.Clock()
                        #reset tagging
                        self.should_tag = False

                        while self.Trialclock.getTime() < self.trialdur:
                            
                            self.fixation.setAutoDraw(True)
                            for frameN in range(len(self.frame_f0)):
                                self.myStim.draw()
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
                                self.mywin.flip()

                            self.myStim.draw()

                        self.fixation.setAutoDraw(False)
                        self.fixCount+=1
                        self.mywin.flip()

                        core.wait(self.waitdur)
                        self.Trialclock.reset()      
                        print("Trial %d Complete" % self.count)
                        self.count+=1
                    
                else:
                    print ("End of video")
                    # It is better to wait for a while for the next frame to be ready
                    cv2.waitKey(1000)
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.mywin.close()
        core.quit()

    def newPoint(self, pt):
         
        newPt = np.zeros(4)
        if (pt[0]<(self.sframe[0]/2)):
            newPt[0] = -((self.sframe[0]/2-pt[0]))

        else:
            newPt[0] = (pt[0] - (self.sframe[0]/2))

        if (pt[1]<(self.sframe[1]/2)):
            newPt[1] = ((self.sframe[1]/2)-pt[1])
        else:
            newPt[1] = -(pt[1]-(self.sframe[1]/2))

        if (pt[2]<(self.sframe[0]/2)):
            newPt[2] = -((self.sframe[0]/2)-pt[2])

        else:
            newPt[2] = (pt[2]-(self.sframe[0]/2))

        if (pt[3]<(self.sframe[1]/2)):
            newPt[3] = ((self.sframe[1]/2)-pt[3])
        else:
            newPt[3] = -(pt[3]-(self.sframe[1]/2))

        return newPt


if __name__ == '__main__':

    stimuli = SSVEP()
    stimuli.start()
    stimuli.stop()
