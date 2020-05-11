# RoboticsProject

This is our project for [CSE 571](https://courses.cs.washington.edu/courses/cse571/20sp/). We implemented Deep Ensemble, Differentiable Particle Filter and a simple behavior cloning algorithm on cartpole system. 
* The master branch is for DPF. For the other two algorithms, please checkout to the corresponding branch. 
* To generate data for training, please checkout the data/ folder for more details.

### step 1: Deep Ensembles estimation
Adapted Tensorflow code from this [repo](https://github.com/vvanirudh/deep-ensembles-uncertainty).  
Tested on sinusoidal toy dataset.

### step 2: Differentiable Particle Filters estimation
Original [implementation](https://github.com/tu-rbo/differentiable-particle-filters) from the authors.

We provided pretrained models for DPF, to run the recurrent model 
```
python cartpole_test.py --mode rec
```
run the end-to-end model
```
python cartpole_test.py --mode e2e
```

### step 3: behavior cloning
