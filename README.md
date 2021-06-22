Adversarial Robustness vs Model Compression, or Both?
-----------------------

In this work,  This paper first proposes a framework of concurrent adversarial training and different weight pruning that enables model compression while still preserving the adversarial robustness and essentially tackles the dilemma of adversarial training.  

Cite this work:
Shaokai Ye\*, Kaidi Xu\*, Sijia Liu,  Jan-Henrik Lambrechts, Huan Zhang, Aojun Zhou, Kaisheng Ma, Yanzhi Wang, Xue Lin. ["Adversarial Robustness vs Model Compression, or Both?"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf), ICCV 2019. (\* Equal Contribution)

and

Shaokai Ye, Xiaoyu Feng, Tianyun Zhang, Xiaolong Ma, Sheng Lin, Zhengang Li, Kaidi Xu, Wujie Wen, Sijia Liu, Jian Tang, Makan Fardad, Xue Lin, Yongpan Liu, Yanzhi Wang. ["Progressive DNN Compression: A Key to Achieve Ultra-High Weight Pruning and Quantization Rates using ADMM"](https://arxiv.org/pdf/1903.09769.pdf), arXiv:1903.09769


```
@InProceedings{Ye_2019_ICCV,
  author = {Ye, Shaokai and Xu, Kaidi and Liu, Sijia and Cheng, Hao and Lambrechts, Jan-Henrik and Zhang, Huan and Zhou,  Aojun and Ma, Kaisheng and Wang, Yanzhi and Lin, Xue},
  title = {Adversarial Robustness vs. Model Compression, or Both?},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
}
```

```
@article{ye2019progressive,
  title={Progressive dnn compression: A key to achieve ultra-high weight pruning and quantization rates using admm},
  author={Ye, Shaokai and Feng, Xiaoyu and Zhang, Tianyun and Ma, Xiaolong and Lin, Sheng and Li, Zhengang and Xu, Kaidi and Wen, Wujie and Liu, Sijia and Tang, Jian and others},
  journal={arXiv preprint arXiv:1903.09769},
  year={2019}
}
```

Prerequisites
-----------------------

code is compatible with pytorch 1.0.0



Train a model in natural setting/adversarial setting
-----------------------


main.py or adv_main.py for main program, natural setting and adversarial setting respectively

eval.py for quick checking of the sparsity and do some other stuff

config.yaml.example template of the configuration file. One for each dataset.

run.sh.example  template script for running the code.





Compression in adversarial setting are only supported for MNIST and CIFAR10. 

Only compression in natural setting is supported in ImageNet.


