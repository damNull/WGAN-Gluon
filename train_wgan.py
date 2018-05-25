import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import mxboard

ndf = 64
ngf = 64
nc = 3
nz = 100
nepoch = 1000
c = 0.01
ncritic = 1
ctx = mx.gpu()
batch_size = 64
outf = 'result'

if not os.path.exists(outf):
    os.makedirs(outf)

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]//shape[1]
    m = buf.shape[1]//shape[0]

    sx = (i%m)*shape[0]
    sy = (i//m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img
    return None

def visual(title, X, name):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X - np.min(X))*(255.0/(np.max(X) - np.min(X))), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = buff[:,:,::-1]
    #plt.imshow(buff)
    #plt.title(title)
    #plt.savefig(name)
    return np.swapaxes(buff, 0, 2)


netG = nn.HybridSequential()
with netG.name_scope():
    # input is Z, going into a convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
    netG.add(nn.Activation('tanh'))
    # state size. (nc) x 64 x 64

netD = nn.HybridSequential()
with netD.name_scope():
        # input is (nc) x 64 x 64
    netD.add(nn.Conv2D(ndf, 4, 2, 1, use_bias=False))
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 32 x 32
    netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 16 x 16
    netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 8 x 8
    netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 4 x 4
    netD.add(nn.Conv2D(1, 4, 1, 0, use_bias=False))

logging.basicConfig(level=logging.DEBUG)

def transformer(data, label):
    # resize to 64x64
    data = mx.image.imresize(data, 64, 64)
    # transpose from (64, 64, 3) to (3, 64, 64)
    data = mx.nd.transpose(data, (2,0,1))
    # normalize to [-1, 1]
    data = data.astype(np.float32)/128 - 1
    # if image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = mx.nd.tile(data, (3, 1, 1))
    return data, label

train_data = gluon.data.DataLoader(
    gluon.data.vision.FashionMNIST('./data/fashion_mnist', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.FashionMNIST('./data/fashion_mnist', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False)

# train_data = gluon.data.DataLoader(
#     gluon.data.vision.MNIST('./data/mnist', train=True, transform=transformer),
#     batch_size=batch_size, shuffle=True, last_batch='discard')

# val_data = gluon.data.DataLoader(
#     gluon.data.vision.MNIST('./data/mnist', train=False, transform=transformer),
#     batch_size=batch_size, shuffle=False)

# loss
# loss = gluon.loss.SoftmaxCrossEntropyLoss()

# initialize the generator and the discriminator
netG.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.initialize(mx.init.Normal(0.02), ctx=ctx)

# hybridize
netG.hybridize()
netD.hybridize()

# trainer for the generator and the discriminator
# trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.0002, 'beta1': 0.5})
# trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.0002, 'beta1': 0.5})
# trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'rmsprop', {'learning_rate': 0.00005})
trainerD = gluon.Trainer(netD.collect_params(), 'rmsprop', {'learning_rate': 0.00005})

real_label = mx.nd.ones((batch_size,), ctx=ctx)
fake_label = mx.nd.zeros((batch_size,), ctx=ctx)

metric = mx.metric.Accuracy()
print('Training... ')
stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
one = mx.nd.ones((batch_size, 1), ctx=mx.gpu())
mone = -one

# param clip
# D
netD.forward(mx.nd.ones((batch_size, 3, 64, 64,), ctx=mx.gpu()))
params =  netD.collect_params()
for i in params:
    p = params[i].data()
    mx.nd.clip(p, -c, c, out=p)   

with mxboard.SummaryWriter(logdir='logs') as sw:
    iternum = 0
    for epoch in range(nepoch):
        tic = time.time()
        btic = time.time()
        for data, _ in train_data:
            ############################
            # (1) Update D network: maximize D(x) - D(G(z))
            ###########################
            # train with real_t
            data = data.as_in_context(ctx)
            noise = mx.nd.random.normal(0, 1, shape=(batch_size, nz, 1, 1), ctx=ctx)
            for i in range(ncritic):
                with autograd.record():
                    output = netD(data)
                    output = output.reshape((batch_size, 1))
                    errD_real = output
                    #errD_real.backward(one)
                    #errD_real.attach_grad()
                    metric.update([real_label,], [output,])

                    fake = netG(noise)
                    output = netD(fake.detach())
                    output = output.reshape((batch_size, 1))
                    errD_fake = output
                    #errD_fake.backward(mone)
                    #errD_fake.attach_grad()
                    errD =  errD_fake - errD_real
                    errD = -errD
                    # errD.backward(mx.nd.ones((batch_size, 1), ctx=mx.gpu()))
                    errD.backward()
                    metric.update([fake_label,], [output,])

                trainerD.step(batch_size)

            # param clip
            # D
            params =  netD.collect_params()
            for i in params:
                p = params[i].data()
                mx.nd.clip(p, -c, c, out=p)            

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            with autograd.record():
                output = netD(fake)
                output = output.reshape((-1, 1))
                errG = -output
                errG = -errG
                #errG.attach_grad()
                # errG.backward(mx.nd.ones((batch_size, 1), ctx=mx.gpu()))
                errG.backward()

            trainerG.step(batch_size)
 
            name, acc = metric.get()
            if iternum % 100 == 0:
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info('errD_fake %f, errD_real %f, errG %f'%(mx.nd.mean(errD_fake).asscalar(), mx.nd.mean(errD_real).asscalar(), mx.nd.mean(errG).asscalar()))
                logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d' %(mx.nd.mean(-errD).asscalar(), mx.nd.mean(errG).asscalar(), acc, iternum, epoch))
            sw.add_scalar(tag='dloss', value=mx.nd.mean(-errD).asscalar(), global_step=iternum)
            sw.add_scalar(tag='gloss', value=mx.nd.mean(errG).asscalar(), global_step=iternum)
            if iternum % 100 == 0:
                sw.add_image(tag='gout', image=visual('gout', fake.asnumpy(), name=os.path.join(outf,'fake_img_iter_%d.png' %iternum)), global_step=iternum)
                sw.add_image(tag='data', image=visual('data', data.asnumpy(), name=os.path.join(outf,'real_img_iter_%d.png' %iternum)), global_step=iternum)

            iternum = iternum + 1
            btic = time.time()

        name, acc = metric.get()
        metric.reset()
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))