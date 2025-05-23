_base_ = ['./segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

dataset_type = 'BaseSegDataset'
data_root = 'data/RUGD'

metainfo = {
    'classes': ('dirt', 'sand', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'generic-object', 'asphalt',
                'gravel', 'building', 'mulch', 'rock-bed', 'log', 'bicycle', 'person', 'fence', 'bush', 'sign', 'rock',
                'bridge', 'concrete', 'picnic-table'),
    'palette': [(108, 64, 20), (255, 229, 204), (0, 102, 0), (0, 255, 0), (0, 153, 153), (0, 128, 255), (0, 0, 255),
                (255, 255, 0), (255, 0, 127), (64, 64, 64), (255, 128, 0), (255, 0, 0), (153, 76, 0), (102, 102, 0),
                (102, 0, 0), (0, 255, 128), (204, 153, 255), (102, 0, 204), (255, 153, 204), (0, 102, 102),
                (153, 204, 255), (102, 255, 255), (101, 101, 11), (114, 85, 47)]
}

common_dataset_kwargs = {
    'type': dataset_type,
    'data_root': data_root,
    'data_prefix': {
        'img_path': 'raw_frames',
        'seg_map_path': 'annotations_id'
    },
    'img_suffix': '.png',
    'seg_map_suffix': '.png',
    'metainfo': metainfo,
    'reduce_zero_label': True
}

train_dataloader = {
    'batch_size': 4,
    'num_workers': 18,
    'dataset': {
        **common_dataset_kwargs, 'ann_file':
            'annotations/lists/train.txt',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'LoadAnnotations'
        }, {
            'type': 'RandomCrop',
            'crop_size': (688, 550),
            'cat_max_ratio': 0.75
        }, {
            'type': 'RandomFlip',
            'prob': 0.5
        }, {
            'type': 'PackSegInputs'
        }]
    }
}

val_dataloader = {
    'batch_size': 4,
    'num_workers': 18,
    'dataset': {
        **common_dataset_kwargs, 'ann_file':
            'annotations/lists/val.txt',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'Resize',
            'scale': (688, 550),
            'keep_ratio': False
        }, {
            'type': 'LoadAnnotations'
        }, {
            'type': 'PackSegInputs'
        }]
    }
}

test_dataloader = {
    'batch_size': 4,
    'num_workers': 18,
    'dataset': {
        **common_dataset_kwargs, 'ann_file':
            'annotations/lists/test.txt',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'Resize',
            'scale': (688, 550),
            'keep_ratio': False
        }, {
            'type': 'LoadAnnotations'
        }, {
            'type': 'PackSegInputs'
        }]
    }
}

test_evaluator = {'type': 'IoUMetric', 'iou_metrics': ['mIoU']}

model = {'decode_head': {'num_classes': 24}, 'test_cfg': {'mode': 'whole'}}

optim_wrapper = {'optimizer': {'type': 'AdamW', 'lr': 3e-5, 'betas': (0.9, 0.999), 'weight_decay': 0.01}}

train_cfg = {'type': 'IterBasedTrainLoop', 'max_iters': 80000, 'val_interval': 4000}
val_cfg = {'type': 'ValLoop'}
test_cfg = {'type': 'TestLoop'}

default_hooks = {
    'timer': {
        'type': 'IterTimerHook'
    },
    'logger': {
        'type': 'LoggerHook',
        'interval': 100
    },
    'param_scheduler': {
        'type': 'ParamSchedulerHook'
    },
    'checkpoint': {
        'type': 'CheckpointHook',
        'interval': 1000
    },
    'sampler_seed': {
        'type': 'DistSamplerSeedHook'
    },
    'visualization': {
        'type': 'SegVisualizationHook'
    }
}

randomness = {'seed': 42}

visualizer = {'type': 'SegLocalVisualizer',
                  'vis_backends': [
                      {'type': 'LocalVisBackend'},
                      {'type': 'TensorboardVisBackend', 'save_dir': 'work_dirs/segformer-b0-rugd'}
                  ]}
