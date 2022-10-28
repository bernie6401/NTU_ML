# Machine Learning Homework 2

## Configuration

| batch_size |       checkpoint       | epoch |  lr   | momentum | optimizer |
| :--------: | :--------------------: | :---: | :---: | :------: | :-------: |
|    256     | epoch490_acc0.6243.pth |  600  | 0.001 |   0.9    |   Adam    |

| weight decay | gamma | step | Data Aug. |     Early Stop      | Normalization |
| :----------: | :---: | :--: | :-------: | :-----------------: | :-----------: |
|     0.01     |  0.2  |  40  |   True    | True(Threshold = 5) |     True      |
## Model

```python
self.conv_4layer = nn.Sequential(
    nn.Conv2d(1, n_chansl, kernel_size=3, padding=1),
    nn.BatchNorm2d(n_chansl, eps=1e-05, affine=True),
    nn.LeakyReLU(negative_slope=0.05),
    nn.MaxPool2d((2, 2)),   # (Batch_size, n_chansl, 32, 32)->(B, C, H, W)

    nn.Conv2d(n_chansl, n_chansl*4, kernel_size=3, padding=1),
    nn.BatchNorm2d(n_chansl*4, eps=1e-05, affine=True),
    nn.LeakyReLU(negative_slope=0.05),
    nn.MaxPool2d((2, 2)),   # (Batch_size, n_chansl*2, 16, 16)->(B, C, H, W)

    nn.Conv2d(n_chansl*4, n_chansl*8, kernel_size=3, padding=1),
    nn.BatchNorm2d(n_chansl*8, eps=1e-05, affine=True),
    nn.LeakyReLU(negative_slope=0.05),
    nn.MaxPool2d((2, 2)),   # (Batch_size, n_chansl*4, 8, 8)->(B, C, H, W)

    nn.Conv2d(n_chansl*8, n_chansl*16, kernel_size=3, padding=1),
    nn.BatchNorm2d(n_chansl*16, eps=1e-05, affine=True),
    nn.LeakyReLU(negative_slope=0.05),
    nn.MaxPool2d((2, 2)),   # (Batch_size, n_chansl*8, 4, 4)->(B, C, H, W)
)
self.fc_4layer = nn.Sequential(
    nn.Linear(n_chansl*16 * 4 * 4, n_chansl*4 * 4 * 4),
    nn.Linear(n_chansl*4 * 4 * 4, 7)
)
```



## Run
* For training
```
python MLHW.py --epoch 600 --lr 0.001 --gamma 0.2 --step 40 --batch_size 256 --early_stop --data_aug -c ./epoch490_acc0.6243.pth
```
* For testing
```
python MLHW.py --mode test -c ./epoch115_acc0.6318.pth
```