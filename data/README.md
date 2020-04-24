This folder contains codes to collect and preprocess data from cartpole simulation.
To collect 100 trajectories from random and swing-up policies respectively, and each trajectory contains 50 time steps, run 
```
python collectData.py --epoch 100 --dp 50
```

The ```CartPoleDataset``` class in dataLoader.py will help you load and preprocess the data into proper format. You can sepcify whether you need image data or not and the number of image stack.  
It is recommended to wrap this dataset into a ```DataLoader``` to make you life easier.
