# class 0 - 10Hz : person, potted plant
# class 1 - 12Hz : dog, car
# class 3 - 15Hz : chair, aeroplane
# idxcls = [person, dog, chair, potted plant, aeroplane, car]

import ssd_predict
import eeg_cnn
import argparse
import imutils
import cv2
import numpy as np
from psychopy import visual, core, event
import rdaclient as rc
import time
import Image
import torch
from torch.autograd import Variable
from imutils.video import FPS
from naoqi import ALProxy
import NaocamVideoStream
import math

# writing data to file
experiment_name = "Online_S01"
data_filedir = "~/" + experiment_name + "_" + class_freq + ".npy"
label_filedir = "~/" + experiment_name + "_" + class_freq +  "_labels" + ".npy"
 
# RDA
address = ('192.168.1.158', 51244)     # server address
window = 3000              # plotting window (samples)
 
# # creating a client
client = rc.Client(buffer_size=300000, buffer_window=window)
client.connect(address)
client.start_streaming()
time.sleep(1.0)

class SSVEP(object):
    
    def __init__(self, mywin= visual.Window([800, 600], fullscr=True, monitor='testMonitor',units='deg', waitBlanking = False),
                trialdur = 3, numtrials=3, waitdur=2):

        self.mywin = mywin
        self.myStim = visual.GratingStim(win=self.mywin, pos=[0,0], units = 'norm')
        
        self.arrow1 = visual.ImageStim(win=self.mywin, image = '/home/nikkhadijah/Pictures/arrow1.png', pos=[800, 0], size=100, units='pix')
        self.arrow2 = visual.ImageStim(win=self.mywin, image = '/home/nikkhadijah/Pictures/arrow2.png', pos=[-800, 0], size=100, units='pix')
        self.arrow3 = visual.ImageStim(win=self.mywin, image = '/home/nikkhadijah/Pictures/arrow3.png', pos=[0, -400], size=100, units='pix')

        self.fixation = visual.GratingStim(win=self.mywin, color = [1, -1, -1], size = 10, sf=0, colorSpace='rgb', units='pix')

        self.frame_f0 = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1,1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
        self.frame_f1 = [1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1]
        self.frame_f2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]

        self.trialdur = trialdur
        self.numtrials = numtrials
        self.waitdur = waitdur

        self.nBox = 3
        self.numChan = 9
        self.sample_rate = 500

        # uncomment this for random order
        # self.capBox = self.numtrials/self.nBox
        # self.aBox = np.arange(3)
        # self.unshuffled = np.matlib.repmat(self.aBox, self.capBox, 1)
        # self.randperm = np.random.permutation(self.numtrials)
        # self.Boxes = self.unshuffled.ravel()
        # self.Boxes = self.Boxes[self.randperm]
        # print self.Boxes

        self.Data_sample = np.zeros(shape=(self.numtrials, self.sample_rate*self.trialdur, self.numChan))
        self.labels = []

        self.IP = NaocamVideoStream.IP  # Replace here with your NaoQi's IP address.
        self.PORT = NaocamVideoStream.PORT

        # autonomous life, motion and posture
        self.autoProxy = ALProxy("ALAutonomousLife", self.IP, self.PORT)
        self.motionProxy = ALProxy("ALMotion", self.IP, self.PORT)
        self.postProxy = ALProxy("ALRobotPosture", self.IP, self.PORT)
        self.awareness = ALProxy("ALBasicAwareness", self.IP, self.PORT)

        # grab a pointer to the video stream and initialize the FPS counter
        print("[INFO] sampling frames...")
        self.vc = NaocamVideoStream.NaocamVideoStream().start()

        self.sframe = [1280, 960] # resolution = 3
        self.focalCam = 2.05 # focal length in mm
        self.sensor = [2.40, 1.80 ] # width by height in mm
        self.objectH = [[720, 530, 980], [650, 200, 650], [0, 0, 750]]
        
        self.k = 0.8
        self.partDistance = 0.4
        self.speedConfig = []
        self.totaltime = 0

        self.Data_sample = np.zeros(shape=(self.numtrials*2, self.sample_rate*self.trialdur, self.numChan))
        self.labels = []
        self.accuracy = 0
        self.predictlabel = []
        
    def initNao(self):
        
        self.motionProxy.setStiffnesses("Body", 1.0)
        self.postProxy.goToPosture("StandInit", 0.5)
        self.motionProxy.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 0.2)
        print ("Reset the head")
        self.motionProxy.setStiffnesses("Head", 0.5)
        self.motionProxy.setWalkArmsEnabled(False, False)
        self.motionProxy.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", True]])
        self.awareness.stopAwareness()
        
        # http://doc.aldebaran.com/2-1/naoqi/motion/control-walk.html#move-config
        # we recommend 0.060 meters for StepX for more stability.
        
        self.speedConfig.append(['MaxStepX', 0.05]) # between 0.04 to 0.06
        self.speedConfig.append(['MaxStepFrequency', 0.2])

    
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
        
        # set the order of the class for subject to be fixated 
        self.Boxes = [1, 0, 2]
        self.arrow = [0, 2, 1]

        self.motionProxy.setAngles(["HeadYaw", "HeadPitch"], [0.0, -0.1], 0.2)
        
        self.frame_rate = self.mywin.getActualFrameRate()
        self.Trialclock = core.Clock()

        self.fixCount = 0
        self.count = 0
            
        while self.count < self.numtrials:
            SSVEP.initBoxes(self)
            core.wait(2.0)

            # streaming video from NAO
            naoImage = self.vc.read()
            im = Image.frombytes("RGB", (naoImage[0], naoImage[1]), naoImage[6], "raw", "RGB", 0, 1)
            cvimg = np.array(im)

            self.myStim.setTex(im)
            self.myStim.draw()
            self.mywin.flip()
            core.wait(2.0)

            # send the image frame to ssd_predict function for object detection
            retdet = ssd_predict.predict(cvimg)
            
            acc_dist = 0
            pt = retdet[0]
            idxcls = retdet[1]

            if len(pt) > 0:
                # print ("here")
                pt = np.vstack(pt)
                # print pt

                for ndet in range (0, (len(pt))):
                    
                    # newpoint -- converting opencv format to psychopy
                    newPt = SSVEP.newPoint(self, pt[ndet])
                    
                    # assign the positions and the boxes for stimuli based on new points calculated 
                    if idxcls[ndet] == 0 or idxcls[ndet] == 3:
                        # print "class0"
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
                fixSize= [self.pattern1_f0.size, self.pattern1_f1.size, self.pattern1_f2.size]

                self.fixation.pos = (fixPos[self.Boxes[self.count]])
                
                self.fixation.setAutoDraw(True)
                self.myStim.draw()
                self.mywin.flip() 

                core.wait(1.0)

                self.Trialclock = core.Clock()
                #reset tagging
                self.should_tag = False
                self.labels.append(self.Boxes[self.count])

                self.start = time.time()
                start_sample = client.last_sample
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
                    end_sample = client.last_sample
 
                end_sample = client.last_sample
                while (end_sample - start_sample) < ( self.trialdur * self.sample_rate):
                    # print("wating...")
                    # print(end_sample - start_sample)
                    # print( self.trialdur * self.sample_rate)
                    end_sample = client.last_sample
    
                # pull the required sample from the RDA buffer and add to overall array
                sig = client.get_data(start_sample, end_sample)
                sig = sig[0:1500, :] # remove any extra data
                self.Data_sample[self.fixCount, :, :] = sig[:, 0:9]

                # get prediction goes here
                output_class = eeg_cnn.classification(self.Data_sample[self.fixCount, :, :], self.Boxes[self.count])
                print ("Predicted class is %d" % output_class)

                outputlabel = output_class.cpu()
                outputlabel = outputlabel.data.numpy()
                self.predictlabel.append(outputlabel)
                if outputlabel == self.Boxes[self.count]:
                    self.accuracy += 1
                
                self.endtime = time.time()-self.start
                self.totaltime = self.totaltime +  self.endtime

                self.fixation.setAutoDraw(False)
                
                self.mywin.flip()

                # move towards the object class = self.Boxes[self.count]
                # distance(mm) = focal length (mm) * object height (mm) / image height (mm)
                # distance(mm) = focal length (px) * object height (mm) / image height (px)
                # focal length(px) = frame height (px) * focal length (mm) / sensor height (mm)
                # focaltest = (self.sframe[0] * self.focalCam) /  self.sensor[0]

                self.focalCamPx = (self.sframe[1] * self.focalCam) /  self.sensor[1]
                # print self.focalCamPx
                self.distanceZ = (self.focalCamPx * self.objectH[self.count][self.Boxes[self.count]]) / fixSize[self.Boxes[self.count]][1]
                
                # convert to metre
                self.distanceZ = self.distanceZ/1000
                print self.distanceZ
                core.wait(1.0)

                if self.distanceZ > 10.0 or self.distanceZ < 0.2:
                    print "waiting for the next frame"
                    continue
                
                else:
                    # angle of view (rad) = position x of image (pix) / focal length (pix)
                    self.fixCount+=1
                    self.angleView = fixPos[self.Boxes[self.count]][0] / self.focalCamPx
                    print self.angleView
                        
                    if self.count == 1:
                        self.motionProxy.moveTo(0, 0, -(1.7*self.angleView), self.speedConfig)
                        # print "here"
                    if self.count == 2:
                        self.motionProxy.moveTo(0, 0, -(1.2*self.angleView), self.speedConfig)
                        # print "here"
                    if self.count == 0:   
                        self.motionProxy.moveTo(0, 0, -(1.0*self.angleView), self.speedConfig)
                        # print "here"

                    self.motionProxy.moveTo((self.k*self.distanceZ), 0, 0, self.speedConfig)
                    
                    naoImage = self.vc.read()
                    im = Image.frombytes("RGB", (naoImage[0], naoImage[1]), naoImage[6], "raw", "RGB", 0, 1)
                    cvimg = np.array(im)

                    # trial = predict(cvimg)
                    self.myStim.setTex(im)
                    self.myStim.draw()
                    self.mywin.flip()
                    core.wait(1.0)

                    # position of arrows
                    SSVEP.initBoxes(self)
                    self.pattern1_f0.pos = [800, 0]
                    self.pattern1_f0.size = 200
                    self.pattern2_f0.pos = [800, 0]
                    self.pattern2_f0.size = 200
                    self.pattern1_f1.pos = [-800, 0]
                    self.pattern1_f1.size = 200
                    self.pattern2_f1.pos = [-800, 0]
                    self.pattern2_f1.size = 200
                    self.pattern1_f2.pos = [0, -400]
                    self.pattern1_f2.size = 200
                    self.pattern2_f2.pos = [0, -400]
                    self.pattern2_f2.size = 200

                    fixPos = [self.pattern1_f0.pos, self.pattern1_f1.pos, self.pattern1_f2.pos]
                    fixSize= [self.pattern1_f0.size, self.pattern1_f1.size, self.pattern1_f2.size]
                    self.fixation.pos = (fixPos[self.arrow[self.count]])
                    
                    self.fixation.setAutoDraw(True)
                    self.myStim.draw()
                    self.arrow1.draw()
                    self.arrow2.draw()
                    self.arrow3.draw()
                    self.mywin.flip() 

                    core.wait(2.0)

                    self.Trialclock = core.Clock()
                    #reset tagging
                    self.should_tag = False
                    self.labels.append(self.arrow[self.count])

                    self.start = time.time()
                    start_sample = client.last_sample 
                    while self.Trialclock.getTime() < self.trialdur:
                        
                        for frameN in range(len(self.frame_f0)):
                            self.myStim.draw()
                            self.myStim.draw()
                            self.arrow1.draw()
                            self.arrow2.draw()
                            self.arrow3.draw()
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
                        end_sample = client.last_sample
 
                    end_sample = client.last_sample
                    while (end_sample - start_sample) < ( self.trialdur * self.sample_rate):
                        # print("wating...")
                        # print(end_sample - start_sample)
                        # print( self.trialdur * self.sample_rate)
                        end_sample = client.last_sample
        
                    # pull the required sample from the RDA buffer and add to overall array
                    sig = client.get_data(start_sample, end_sample)
                    sig = sig[0:1500, :] # remove any extra data
                    # print (sig)
                    self.Data_sample[self.fixCount, :, :] = sig[:, 0:9]
                    
                    # get prediction goes here
                    output_class = eeg_cnn.classification(self.Data_sample[self.fixCount, :, :], self.arrow[self.count])
                    print ("Predicted class is %d" % output_class)
                    
                    outputlabel = output_class.cpu()
                    outputlabel = outputlabel.data.numpy()
                    self.predictlabel.append(outputlabel)
                    if outputlabel == self.arrow[self.count]:
                        self.accuracy += 1

                    self.endtime = time.time()-self.start
                    self.totaltime = self.totaltime + self.endtime

                    self.fixation.setAutoDraw(False)
                    self.mywin.flip()

                    if self.arrow[self.count] == 0:
                        self.motionProxy.moveTo(0, 0, (self.k*-1.5708), self.speedConfig)
                        naoImage = self.vc.read()
                        im = Image.frombytes("RGB", (naoImage[0], naoImage[1]), naoImage[6], "raw", "RGB", 0, 1)
                        cvimg = np.array(im)

                        self.myStim.setTex(im)
                        self.myStim.draw()
                        self.mywin.flip()
                        core.wait(2.0)

                    if self.arrow[self.count] == 1:
                        self.motionProxy.moveTo(0, 0, (self.k*1.5708), self.speedConfig)

                        naoImage = self.vc.read()
                        im = Image.frombytes("RGB", (naoImage[0], naoImage[1]), naoImage[6], "raw", "RGB", 0, 1)
                        cvimg = np.array(im)

                        self.myStim.setTex(im)
                        self.myStim.draw()
                        self.mywin.flip()
                        core.wait(2.0)
                    
                    if self.arrow[self.count] == 2:
                        self.motionProxy.moveTo(0, 0, -(self.k*1.5708), self.speedConfig)
                        self.motionProxy.moveTo(0, 0, -(self.k*1.5708), self.speedConfig)

                        naoImage = self.vc.read()
                        im = Image.frombytes("RGB", (naoImage[0], naoImage[1]), naoImage[6], "raw", "RGB", 0, 1)
                        cvimg = np.array(im)

                        self.myStim.setTex(im)
                        self.myStim.draw()
                        self.mywin.flip()
                        core.wait(2.0)
                    
                    core.wait(self.waitdur)
                    self.Trialclock.reset() 
                    print("Trial %d Complete" % self.count)
                    self.fixCount+=1
                    self.count+=1
                
        print self.labels
        print self.totaltime
        print self.predictlabel
        print ("Accuracy %d" % self.accuracy)
        np.save(data_filedir , self.Data_sample)
        np.save(label_filedir, np.asarray(self.labels))
        cv2.destroyAllWindows()

    def stop(self):
        self.vc.stop()
        self.mywin.close()
        core.quit()

    def newPoint(self, pt):
        # print frame
        # print pt
         
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
        
        # print newPt
        return newPt


if __name__ == '__main__':

    stimuli = SSVEP()
    stimuli.initNao()
    stimuli.start()
    stimuli.stop()
