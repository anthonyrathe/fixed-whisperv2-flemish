---
license: apache-2.0
tags:
- whisper-event
- generated_from_trainer
datasets:
- kul-speech-lab/CGN
metrics:
- wer
model-index:
- name: Whisper Large CGN
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: kul-speech-lab/CGN
      type: kul-speech-lab/CGN
      config: cgn-dev.py
      split: test
    metrics:
    - name: Wer
      type: wer
      value: 9.6159
---

# Whisper Large CGN

This model is a fine-tuned version of [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) on the kul-speech-lab/CGN dataset.
It achieves the following results on the evaluation set:
- Loss: 0.23932012915611267
- Wer: 9.615871912312803

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 32
- eval_batch_size: 16
- gradient_accumulation_steps: 2
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 15000
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.26.0.dev0
- Pytorch 1.13.0
- Datasets 2.7.1.dev0
- Tokenizers 0.13.2

Whisper large model finetuned on Flemish part of Corpus Gesproken Nederlands (CGN).
