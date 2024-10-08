# ToothMonitoring

Required packages: 
* python2.7
* opendr
* chumpy
* OSMesa (required when installing opendr)
* lmfit
* numpy, scipy
* vtk

Required data:
* Tooth model
* Ground truth contour for each view
* Initial camera pose

Usage:
```
python FitContour_fixed.py --t_model teeth_model_folder --view1 ground_truth_contour_for_view1 --view2 ground_truth_contour_for_view2 \
                     --view3 ground_truth_contour_for_view3 --view4 ground_truth_contour_for_view4 \
                     --camera_pose camera_pose_filename
                     (Optional: use --rt1 --t1 --rt2 --t2 --rt3 --t3 --rt4 --t4 to set the camera pose)
                     
example:
python FitContour_fixed.py --t_model /home/jiaming/MultiviewFitting/data/upper_segmented/HBF_12681/before/ 
                     --view1 /home/jiaming/MultiviewFitting/data/observation/12681/real_rc1.jpg --view2 /home/jiaming/MultiviewFitting/data/observation/12681/real_rc2.jpg 
                     --view3 /home/jiaming/MultiviewFitting/data/observation/12681/real_rc3.jpg --view4 /home/jiaming/MultiviewFitting/data/observation/12681/real_rc4.jpg 
                     --camera_pose camera_pose.txt 
```
