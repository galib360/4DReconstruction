# 4DReconstruction repo

4D Reconstruction using Visual Hull (Shape from silhouettes); Master Thesis. 

This repository contains the frame by frame 3D reconstruction of the dynamic object of interest from multiview dataset, which is the second part of the proposed pipeline.
After the silhouette extraction method (Please see repository "CRFSilhouette"), the segmented silhouettes along with the camera calibration parameters and the initial 
bounding box are used to generate 3D models in time for all the synchronised frames. 

The code also incorporates a latency analysis where a simulated latency in synchronised capture is introduced to carry out a qualitative analysis when there is a latency 
among the capture devices.


Thesis PDF link:
https://drive.google.com/file/d/1V89VL0UPKGOMky5N7zpgZtnxbuTTN7kz/view?usp=sharing
