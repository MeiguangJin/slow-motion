from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import numpy as np
import utils
import os
from os import listdir
from os.path import isfile, join
import fnmatch
from model import RDN_residual_interp
from model import RDN_residual_deblur
# Training settings
parser = argparse.ArgumentParser(description='parser for video prediction')
parser.add_argument('--model', type=str, default='', help='model file to estimate shared frame')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--out', type=str, required=True, help='output image ')
parser.add_argument('--input', type=str, required=True, help='input directory ')

args = parser.parse_args()
if not os.path.exists(args.out):
    os.makedirs(args.out)
# load model
model_deblur = RDN_residual_deblur()
model_interp = RDN_residual_interp()

if args.model:
    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model)
        model_deblur.load_state_dict(checkpoint['state_dict_deblur'])
        model_interp.load_state_dict(checkpoint['state_dict_interp'])
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

if args.cuda:
    model_deblur.cuda()
    model_interp.cuda()

model_deblur.eval()
model_interp.eval()
count = 0
input_transform = transforms.Compose([
    transforms.ToTensor(),
])
with torch.no_grad():
    for ii in range(1, 61):
        count = count + 1

        input01 = utils.load_image('%s/%03d.png' % (args.input,ii-1) )
        width, height= input01.size
        input01 = input01.crop((0, 0, width-width%2, height-height%2))
        input01 = input_transform(input01)
        input01 = input01.unsqueeze(0)

        input02 = utils.load_image('%s/%03d.png' % (args.input,ii) )
        input02 = input02.crop((0, 0, width-width%2, height-height%2))
        input02 = input_transform(input02)
        input02 = input02.unsqueeze(0)

        input03 = utils.load_image('%s/%03d.png' % (args.input,ii+1) )
        input03 = input03.crop((0, 0, width-width%2, height-height%2))
        input03 = input_transform(input03)
        input03 = input03.unsqueeze(0)

        input04 = utils.load_image('%s/%03d.png' % (args.input,ii+2) )
        input04 = input04.crop((0, 0, width-width%2, height-height%2))
        input04 = input_transform(input04)
        input04 = input04.unsqueeze(0)

        if args.cuda:
            input01 = input01.cuda()
            input02 = input02.cuda()
            input03 = input03.cuda()
            input04 = input04.cuda()

        output05, output09, output11 = model_deblur(input01, input02, input03, input04)
        output07 = model_interp(input02, output05, output09, input03)
        output06 = model_interp(input02, output05, output07, input03)
        output08 = model_interp(input02, output07, output09, input03)
        output10 = model_interp(input02, output09, output11, input03)

        if ii > 1:
            previous_out = utils.load_image('%s/%04d.png' % (args.out, (ii-1) * 10 - 4) )
            previous_out = input_transform(previous_out)
            previous_out = previous_out.unsqueeze(0)
            previous_out = previous_out.cuda()
            output03 = model_interp(input01, previous_out, output05, input02)
            output02 = model_interp(input01, previous_out, output03, input02)
            output04 = model_interp(input01, output03, output05, input02)

        if args.cuda:
            output05 = output05.cpu()
            output06 = output06.cpu()
            output07 = output07.cpu()
            output08 = output08.cpu()
            output09 = output09.cpu()
            output10 = output10.cpu()
            output11 = output11.cpu()

            if ii > 1:
                output02 = output02.cpu()
                output03 = output03.cpu()
                output04 = output04.cpu()


        output05_data = output05.data[0]*255
        output06_data = output06.data[0]*255
        output07_data = output07.data[0]*255
        output08_data = output08.data[0]*255
        output09_data = output09.data[0]*255
        output10_data = output10.data[0]*255
        output11_data = output11.data[0]*255


        if ii > 1:
            output02_data = output02.data[0]*255
            output03_data = output03.data[0]*255
            output04_data = output04.data[0]*255

        utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 10), output05_data)
        utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 9),  output06_data)
        utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 8),  output07_data)
        utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 7),  output08_data)
        utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 6),  output09_data)
        utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 5),  output10_data)
        utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 4),  output11_data)
        if ii > 1:
            utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 13), output02_data)
            utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 12), output03_data)
            utils.save_image( '%s/%04d.png' % (args.out, ii * 10 - 11), output04_data)

        print('finish {:03d} processing example'.format(count))
