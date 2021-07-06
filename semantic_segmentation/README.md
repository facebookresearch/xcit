# ADE20k Semantic segmentation with XCiT

## Getting started 

Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library

```
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
```

Please follow the datasets guide of [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.

---

## XCiT + Semantic FPN models (80k schedule)

<table>
  <tr>
    <th>Backbone</th>
    <!-- <th>key</th> -->
    <th>patch size</th>
    <th>mIoU</th>
    <th>Config</th>
    <th>Weights</th>
  </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Tiny 12</em></td>
    <td>16x16</td>
    <td>38.1</td>
    <td><a href="configs/xcit/sem_fpn/sem_fpn_xcit_tiny_12_p16_80k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_tiny_12_p16.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Tiny 12</em></td>
    <td>8x8</td>
    <td>39.9</td>
    <td><a href="configs/xcit/sem_fpn/sem_fpn_xcit_tiny_12_p8_80k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_tiny_12_p8.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 12</em></td>
    <td>16x16</td>
    <td>43.9</td>
    <td><a href="configs/xcit/sem_fpn/sem_fpn_xcit_small_12_p16_80k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_small_12_p16.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 12</em></td>
    <td>8x8</td>
    <td>44.2</td>
    <td><a href="configs/xcit/sem_fpn/sem_fpn_xcit_small_12_p8_80k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_small_12_p8.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 24</em></td>
    <td>16x16</td>
    <td>44.6</td>
    <td><a href="configs/xcit/sem_fpn/sem_fpn_xcit_small_24_p16_80k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_small_24_p16.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Small 24</em></td>
    <td>8x8</td>
    <td>47.1</td>
    <td><a href="configs/xcit/sem_fpn/sem_fpn_xcit_small_24_p8_80k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_small_24_p8.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Medium 24</em></td>
    <td>16x16</td>
    <td>45.9</td>
    <td><a href="configs/xcit/sem_fpn/sem_fpn_xcit_medium_24_p16_80k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_medium_24_p16.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Medium 24</em></td>
    <td>8x8</td>
    <td>46.9</td>
    <td><a href="configs/xcit/sem_fpn/sem_fpn_xcit_medium_24_p8_80k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_medium_24_p8.pth">download</a></td>
    </tr>
</table>


## XCiT + UperNet models (160k schedule)

<table>
  <tr>
    <th>Backbone</th>
    <!-- <th>key</th> -->
    <th>patch size</th>
    <th>mIoU</th>
    <th>Config</th>
    <th>Weights</th>
  </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Tiny 12</em></td>
    <td>16x16</td>
    <td>41.5</td>
    <td><a href="configs/xcit/upernet/upernet_xcit_tiny_12_p16_160k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/upernet_xcit_tiny_12_p16.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Tiny 12</em></td>
    <td>8x8</td>
    <td>43.5</td>
    <td><a href="configs/xcit/upernet/upernet_xcit_tiny_12_p8_160k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/upernet_xcit_tiny_12_p8.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 12</em></td>
    <td>16x16</td>
    <td>45.9</td>
    <td><a href="configs/xcit/upernet/upernet_xcit_small_12_p16_160k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/upernet_xcit_small_12_p16.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 12</em></td>
    <td>8x8</td>
    <td>46.6</td>
    <td><a href="configs/xcit/upernet/upernet_xcit_small_12_p8_160k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/upernet_xcit_small_12_p8.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 24</em></td>
    <td>16x16</td>
    <td>46.9</td>
    <td><a href="configs/xcit/upernet/upernet_xcit_small_24_p16_160k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/upernet_xcit_small_24_p16.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Small 24</em></td>
    <td>8x8</td>
    <td>48.1</td>
    <td><a href="configs/xcit/upernet/upernet_xcit_small_24_p8_160k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/upernet_xcit_small_24_p8.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Medium 24</em></td>
    <td>16x16</td>
    <td>47.6</td>
    <td><a href="configs/xcit/upernet/upernet_xcit_medium_24_p16_160k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/upernet_xcit_medium_24_p16.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Medium 24</em></td>
    <td>8x8</td>
    <td>48.4</td>
    <td><a href="configs/xcit/upernet/upernet_xcit_medium_24_p8_160k_ade20k.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/ade/upernet_xcit_medium_24_p8.pth">download</a></td>
    </tr>
</table>

## Training

```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --seed 0  --deterministic --options model.pretrained=<IMAGENET_CHECKPOINT_PATH/URL>
```

For example, using an XCiT-S12/16 backbone with Semantic-FPN

```
tools/dist_train.sh configs/xcit/sem_fpn/sem_fpn_xcit_small_12_p16_80k_ade20k.py 8  --work-dir /path/to/save --seed 0  --deterministic --options model.pretrained=https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth
```

## Evaluation

```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, using an XCiT-S12/16 backbone with Semantic-FPN

```
 tools/dist_test.sh  configs/xcit/sem_fpn/sem_fpn_xcit_small_12_p16_80k_ade20k.py https://dl.fbaipublicfiles.com/xcit/ade/sem_fpn_xcit_small_12_p16.pth  1 --eval mIoU
```

---

## Acknowledgment 

This code is built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library. The optimization hyperparameters we use are adopted from [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) repository.
