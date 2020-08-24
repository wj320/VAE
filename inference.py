import argparse
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utils import *
from model.model_VAE_conv import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

cal_psnr = 1  # calculate PSNR
save_real_img = 0

parser = argparse.ArgumentParser()
parser.add_argument('--use_FC', type=bool, default=0) # use FC (1) or FCN (0)
parser.add_argument('--z_dim', type=int, default=20)  # dimension of latent variable
parser.add_argument('--channel', type=int, default=128)  # hidden layer channel
parser.add_argument('--kernel_size', type=int, default=28)  # conv kernel size
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--save_img_path', type=str, default='result/')
parser.add_argument('--save_model_path', type=str, default='ckpt/')
args = parser.parse_args()

if not os.path.exists(args.save_img_path):
    os.makedirs(args.save_img_path)

transform = transforms.Compose([
    transforms.ToTensor(),  # HxWxC --> CxHxW
    transforms.Normalize([0.5], [0.5]),  # [0,1] --> [-1,1]
])


testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

vae = VAEConv(z_dim=args.z_dim, ch=args.channel, ks=args.kernel_size)
ckpt = torch.load(args.save_model_path)
vae.load_state_dict(ckpt['model_state_dict'])

vae.eval()

for batch_idx, (inputs, targets) in enumerate(testloader):
    # inputs, targets = inputs.to('cpu'), targets.to('cpu')

    if args.use_FC:
        real_imgs = torch.flatten(inputs, start_dim=1)
    else:
        real_imgs = inputs

    gen_imgs, _, _ = vae(real_imgs)

if cal_psnr:
    print('==========PSNR: {}==========='.format(psnr(gen_imgs, real_imgs, 1)))

if args.use_FC:
    fake_imgs = gen_imgs.view(-1, 1, 28, 28)
else:
    fake_imgs = gen_imgs
save_image(fake_imgs, '{}-fake.png'.format(args.save_img_path))

if save_real_img:
    if args.use_FC:
        real_imgs = real_imgs.view(-1,1,28,28)
    save_image(real_imgs, '{}-real.png'.format(args.save_img_path))

torch.cuda.empty_cache()


