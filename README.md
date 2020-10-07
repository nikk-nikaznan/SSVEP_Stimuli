# SSVEP_Stimuli

Code to accompany our International Conference on Robotics and Automation (ICRA) paper entitled -
[Using variable natural environment brain-computer interface stimuli for real-time humanoid robot navigation](https://arxiv.org/pdf/1811.10280.pdf).

The original code for SSD is from [https://github.com/amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).

The code is structured as follows:

- `onebox_stimuli.py ` contains psychopy code for a simple SSVEP stimuli ; 
- `onebox_stimuli_RDA.py ` contains code for integration between simple SSVEP stimuli with RDA server ;
- `multiclass_stimuli.py ` contains code for multiclass SSVEP stimuli -- 4 frequency classes example ;
- `ssd_stimuli.py  ` contains code for displaying SSD object detection in psychopy as SSVEP stimuli;
- `ssd.py ` contains SSD architecture (original work from [https://github.com/amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch). ;
- `ssd_predict.py ` contains prediction test on the image frame modified from .[https://github.com/amdegroot/ssd.pytorch/blob/master/demo/live.py](https://github.com/amdegroot/ssd.pytorch/blob/master/demo/live.py). ;
- `ssd_stimuli_realtime.py ` our Varible SSVEP Stimuli code using SSD object detection in real time (on-board camera of NAO) ;

## Dependencies and Requirements
The code has been designed to support python 3.6+ only. The project has the following dependencies and version requirements:

- torch=1.6.0+
- numpy=1.16++
- python=3.6.5+
- scipy=1.1.0+
- PsychoPy=3.0
- pynaoqi=2.7+
- rdaclient (https://github.com/belevtsoff/rdaclient.py)

## Training weights
Please refer to [https://github.com/amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) on how to download the training weights.

## Cite

Please cite the associated papers for this work if you use this code:

```
@inproceedings{aznan2019using,
  title={Using variable natural environment brain-computer interface stimuli for real-time humanoid robot navigation},
  author={Aznan, Nik Khadijah Nik and Connolly, Jason D and Al Moubayed, Noura and Breckon, Toby P},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={4889--4895},
  year={2019},
  organization={IEEE}
}

```
