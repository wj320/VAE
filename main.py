import argparse
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utils import *
from model.model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print_net = 1  # print network parameters
cal_psnr = 1 # calculate PSNR
resume_train = 0 # resume the training


parser = argparse.ArgumentParser()
parser.add_argument('--use_FC', type=bool, default=0) # use FC (1) or FCN (0)
parser.add_argument('--z_dim', type=int, default=20) # dimension of latent variable
parser.add_argument('--channel', type=int, default=128) # hidden layer channel
parser.add_argument('--kernel_size', type=int, default=28) # conv kernel size
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--lr', type = float, default=0.0003)
parser.add_argument('--save_interval', type=int, default=20) # save ckpt frequency
parser.add_argument('--save_img_path', type=str, default='result/')
parser.add_argument('--save_model_path', type=str, default='ckpt/')
parser.add_argument('--resume_path', type=str, default='')
parser.add_argument('--log_path', type=str, default='log/log-conv')
args = parser.parse_args()

if not os.path.exists(args.save_img_path):
    os.mkdir(args.save_img_path)
if not os.path.exists(args.save_model_path):
    os.mkdir(args.save_model_path)

sys.stdout = Logger(args.log_path)

transform = transforms.Compose([
        transforms.ToTensor(), # HxWxC --> CxHxW
        transforms.Normalize([0.5],[0.5]),  # [0,1] --> [-1,1]
        ])
        
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

if args.use_FC:
    vae = VAEFC(z_dim=args.z_dim)
else:
    vae = VAEFCN(z_dim=args.z_dim, ch=args.channel, ks=args.kernel_size)

optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
start_epoch = 0

if resume_train and len(args.resume_path)>1:
    ckpt = torch.load(args.resume_path)
    start_epoch = ckpt['epoch']
    vae.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])


# print network parameters
print('================parameters===================')
conf = vars(args)
print(conf)
print('===========network architecture==============')
if print_net:
    for layer in vae.named_modules():
        print(layer)

def train(epoch):
    vae.train()
    all_loss = 0.
    flag = 1
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if args.use_FC:
            real_imgs = torch.flatten(inputs, start_dim=1)
        else:
            real_imgs = inputs

        gen_imgs, mu, logvar = vae(real_imgs)

        recon_loss, KL_diver = loss_function(gen_imgs, real_imgs, mu, logvar)
        loss = recon_loss + KL_diver

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Epoch {}, recon loss:{:.6f}, KL:{:.6f}, loss: {:.6f}'.format(epoch+1, recon_loss, KL_diver, all_loss/(batch_idx+1)))

    if cal_psnr:
        print ('==========PSNR: {}==========='.format(psnr(gen_imgs, real_imgs, 1)))

    if args.use_FC:
        fake_imgs = gen_imgs.view(-1,1,28,28)
    else:
        fake_imgs = gen_imgs
    save_image(fake_imgs, '{}/fake_images-{}.png'.format(args.save_img_path, epoch + 1))

    real_imgs = real_imgs.view(-1,1,28,28)
    save_image(real_imgs, '{}/real_images-{}.png'.format(args.save_img_path, epoch+1))

    

for epoch in range(start_epoch, args.max_epoch):
    train(epoch)

    if (epoch+1) % args.save_interval == 0:
        ckpt = {'model_state_dict': vae.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'epoch':epoch, 'z_dim':args.z_dim}
        ckpt_path = '{}/ckpt-epoch-{}.pkl'.format(args.save_model_path, epoch+1)
        #torch.save(vae.state_dict(), 'ckpt/vae-z-{}.pth'.format(args.z_dim))
        torch.save(ckpt, ckpt_path)
torch.cuda.empty_cache()
        
    
        