# Model of TRACE

## Data of TRACE:
**Annotation**   
/home/chaos/data/Chaos/dataset/annotation/AG_vidvrd_format/annotation
-- including train, test, val

**Frames**
/home/chaos/data/Chaos/activity_graph/code/TRACE/data/chaos/frames

## Data Prepation
If there are **any modification** of annotation and ratio of splitting, need to re-run the data prepare.
```
bash prepare_data.sh
```

## TRAINING
Training only one model, which can directly used for evaluation with 3 different Metrics.
```
bash run.sh
```
## EVALIDATION
One Metric use one command. More info in eval.sh. We only evaluate PredCls and SGDet.
```
bash eval.sh
```

