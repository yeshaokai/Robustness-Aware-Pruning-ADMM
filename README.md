code is compatible with pytorch 1.0.0

code release for

""Adversarial Robustness vs Model Compression, or Both?" and "Progressive dnn compression: A key to achieve ultra-high weight pruning and quantization rates using admm".


main.py or adv_main.py for main program, natural setting and adversarial setting respectively

eval.py for quick checking of the sparsity and do some other stuff

config.yaml.example template of the configuration file. One for each dataset.

run.sh.example  template script for running the code.


Compression in adversarial setting are only supported for MNIST and CIFAR10. 

Only compression in natural setting is supported in ImageNet.


