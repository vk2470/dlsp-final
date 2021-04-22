# Studying the Performance Of Self-Supervised Models

This is the repository that has the code to analyse the performance of a self supervised model with increase in 
unlabelled data to train the pretrainer. This codebase uses the CIFAR-10 dataset.

### Self-Supervised Model
The following code will run the end-to-end training of the pretrainer and the finetuner:
```
python trainer.py --pretrainer_num_epochs <num_epochs> --finetuner_num_epochs <num_epochs> --percentage_labelled <% labelled> --percentage_unlabelled <%unlabelled> --finetuner_lr <finetuner_lr> --pretrainer_lr <pretrainer_lr> --pretrainer_backbone_lr <pretrainer_backbone_lr> --batch_size <batch_size>
```
This will also create the loss and accuracy files in the corresponding files in <percentage_labelled>_<percentage_unlabelled> folder 

### Baseline Model

To train the baseline model:
```
python baseline.py --baseline_num_epochs <num_epochs> --percentage_labelled <% labelled> --baseline_lr <baseline learning rate> --batch_size <batch_size>
```
This will also create the loss and accuracy files in the corresponding files in <percentage_labelled>_0.0 folder


autoencoder.ipynb is deprecated, was used for experimentation.
