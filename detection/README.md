# COCO Object detection and Instance segmentation with XCiT

## Getting started 

Install the [mmdetection](https://github.com/open-mmlab/mmdetection) library
```
pip install mmcv-full==1.3.0 mmdet==2.11.0
```

For mixed precision training , please install [apex](https://github.com/NVIDIA/apex)

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Please follow the dataset guide of [mmdet](https://github.com/open-mmlab/mmdetection/blob/master/docs/1_exist_data_model.md#prepare-datasets) to prepare the MS-COCO dataset.

---

## XCiT + Mask R-CNN models (3x schedule)

<table>
  <tr>
    <th>Backbone</th>
    <!-- <th>key</th> -->
    <th>patch size</th>
    <th>bbox mAP</th>
    <th>mask mAP</th>
    <th>Config</th>
    <th>Weights</th>
  </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Tiny 12</em></td>
    <td>16x16</td>
    <td>42.7</td>
    <td>38.5</td>
    <td><a href="configs/xcit/mask_rcnn_xcit_tiny_12_p16_3x_coco.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_tiny_12_p16.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Tiny 12</em></td>
    <td>8x8</td>
    <td>44.5</td>
    <td>40.3</td>
    <td><a href="configs/xcit/mask_rcnn_xcit_tiny_12_p8_3x_coco.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_tiny_12_p8.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 12</em></td>
    <td>16x16</td>
    <td>45.3</td>
    <td>40.8</td>
    <td><a href="configs/xcit/mask_rcnn_xcit_small_12_p16_3x_coco.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_small_12_p16.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 12</em></td>
    <td>8x8</td>
    <td>47.0</td>
    <td>42.3</td>
    <td><a href="configs/xcit/mask_rcnn_xcit_small_12_p8_3x_coco.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_small_12_p8.pth">download</a></td>
    </tr>
    <tr>
    <!-- <td>XCiT-S12/8</td> -->
    <td><em>XCiT-Small 24</em></td>
    <td>16x16</td>
    <td>46.5</td>
    <td>41.8</td>
    <td><a href="configs/xcit/mask_rcnn_xcit_small_24_p16_3x_coco.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_small_24_p16.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Small 24</em></td>
    <td>8x8</td>
    <td>48.1</td>
    <td>43.0</td>
    <td><a href="configs/xcit/mask_rcnn_xcit_small_24_p8_3x_coco.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_small_24_p8.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Medium 24</em></td>
    <td>16x16</td>
    <td>46.7</td>
    <td>42.0</td>
    <td><a href="configs/xcit/mask_rcnn_xcit_medium_24_p16_3x_coco.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_medium_24_p16.pth">download</a></td>
    </tr>
    <tr>
    <td><em>XCiT-Medium 24</em></td>
    <td>8x8</td>
    <td>48.5</td>
    <td>43.7</td>
    <td><a href="configs/xcit/mask_rcnn_xcit_medium_24_p8_3x_coco.py">config</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_medium_24_p8.pth">download</a></td>
    </tr>
</table>

## Training

```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --seed 0  --deterministic --cfg-options model.pretrained=<IMAGENET_CHECKPOINT_PATH/URL>
```

For example, using an XCiT-S12/16 backbone

```
tools/dist_train.sh configs/xcit/mask_rcnn_xcit_small_12_p16_3x_coco.py 8  --work-dir /path/to/save --seed 0  --deterministic --cfg-options model.pretrained=https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth
```

## Evaluation

```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval bbox segm
```

For example, using an XCiT-S12/16 backbone

```
tools/dist_test.sh  configs/xcit/mask_rcnn_xcit_small_12_p16_3x_coco.py https://dl.fbaipublicfiles.com/xcit/coco/maskrcnn_xcit_small_12_p16.pth  1 --eval bbox segm
```

---

## Acknowledgment 

This code is built using the [mmdetection](https://github.com/open-mmlab/mmdetection) library. The optimization hyperparameters we use are adopted from [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) repository.
