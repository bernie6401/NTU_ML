# Machine Learning Homework 3

## Configuration

| latent dim. | checkpoint | epoch |   lr   | momentum | optimizer |
| :---------: | :--------: | :---: | :----: | :------: | :-------: |
|     32      |    N/A     |   5   | 0.0005 |   0.9    |   Adam    |

| weight decay | num. iter. | reduced method | Data Aug. | reduced dim. | Normalization |
| :----------: | :--------: | :------------: | :-------: | :----------: | :-----------: |
|   0.00001    |    1000    |      PCA       |   False   |      64      |     False     |
## Model

```python
self.encoder = nn.Sequential(
    nn.Conv2d(image_channels, n_chansl, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
)

self.fc1 = nn.Linear(n_chansl * (self.img_size//2)**2, self.latent_dim)

self.fc2 = nn.Linear(self.latent_dim, n_chansl * (self.img_size//2)**2)

self.decoder = nn.Sequential(
    nn.ConvTranspose2d(n_chansl, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ReLU()
)
```



## Run
* For training
```
python MLHW3.py --reduced_method pca --reduced_dim 64
```
* For testing
```
python MLHW3.py --mode test -c ./epoch4_0.01263_latest_version --reduced_dim 32
```



## Note

I choose another checkpoint that different with the submission on kaggle. The public score of the submission that kaggle choose is 0.78355 and 0.77733 is for private score. My checkpoint here has also submitted on kaggle as well, but not kaggle chose. Is that okay for TA to reproduce? The public score for my latest checkpoint is 0.778 and 0.78377 is for private score.

I'll provide two checkpoints that all have submitted on kaggle. Note that if TA wants to test with kaggle chosen, you must revise a part shown as below.

```python
# In line 400
model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"], strict=False)
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
model.load_state_dict(torch.load(args.checkpoint), strict=False)
```

You should delete `["model_state_dict"]` that is my self-defined format to store checkpoint and use the command provided above for testing.