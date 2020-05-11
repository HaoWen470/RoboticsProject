# RoboticsProject

On this branch you will find our implementation of the deep ensembles approach, adapted from Tensorflow code found in this [repo](https://github.com/vvanirudh/deep-ensembles-uncertainty)

## How to run the code

### Generate data set
In order to generate a data set for training and testing do the following:

```    
cd data
python collectData.py --mode train
python collectData.py --mode test
```

### Train the model
The code can either train a model with MLP-regressor or one with a CNN-regressor.
The parameters can be adjusted in the file `DeepEnsembles.py`.

In order to train and test the MLP model run
```
python main.py --mode "MLP"
```
In oreder to train the test the CNN model run
```
python main.py --mode "CNN"
```

### Use the trained model in the HW-example
We adpated the code from HW 1 such that it also shows a cartpole and its metrics that uses our Deep Ensembles predictor.

To run the visualization using the MLP model as predictor run
```
python cartpole_test.py --mode "MLP"
```
To run the visualization using the CNN model as predictor run
```
python cartpole_test.py --mode "CNN"
```
Notes: 
- Since the CNN predictor needs images as input, they have to be produced for each timestamp in each epoch. Therefore, the code takes some time to produce the final visualization.
- The visualizer also shows the average RMSE and the Deep Ensembles NLL for the DeepEnsemble predictor and the Gaussian Processes after each epoch.

### Step 1: Deep Ensembles estimation
Adapted Tensorflow code from this [repo](https://github.com/vvanirudh/deep-ensembles-uncertainty).  
Tested on sinusoidal toy dataset.

### step 2: Differentiable Particle Filters estimation
Original [implementation](https://github.com/tu-rbo/differentiable-particle-filters) from the authors.

### step 3: behavior cloning (bonus)
To train behavior cloning model run 
```
python BehaviorClone.py 
```
Combine pretrained deep ensemble model and behavior cloning model to do full model prediction and control, run
```
python bc_rollout.py --mode "MLP"
```
