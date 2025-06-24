# train teacher
# resNet32x4
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model resnet32x4
# resNet8x4
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model resnet8x4
python train_teacher.py --dataset cinic10 --epochs 240 --trial 0 --model resnet8x4

python train_teacher.py --dataset cinic10 --epochs 240 --trial 0 --model vgg8
python train_teacher.py --dataset cinic10 --epochs 240 --trial 0 --model vgg13
# resNet8x4 kd from resNet32x4
python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --dataset cifar10 --model_s resnet8x4 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0
python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cinic10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --dataset cinic10 --model_s resnet8x4 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0

# resNet8x4 SemCKD from resNet32x4
python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --model_s resnet8x4 --distill semckd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0


