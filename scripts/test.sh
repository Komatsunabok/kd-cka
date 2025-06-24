# light model
# train teacher
# GPUあり
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model resnet32x4
# GPUなし（CPU）
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model resnet32x4 --gpu_id ''

# train student
# SemCKD
python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --model_s resnet8x4 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0
   
python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --model_s resnet8x4 --distill semckd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0


# cka
cd cka
python ff_network.py --path_t ../save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --path_s ../save/students/models/S~resnet8x4_T~resnet32x4_cifar10_semckd_r~1.0_a~1.0_b~400.0_0/resnet8x4_best.pth
