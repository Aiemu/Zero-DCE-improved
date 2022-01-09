import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 16
		self.e_conv1 = nn.Conv2d(3, number_f, 3, stride=1, padding=1)
		self.e_conv2 = nn.Conv2d(number_f, number_f, 3, stride=1, padding=1)
		self.e_conv3 = nn.Conv2d(number_f, number_f, 3, stride=1, padding=1)
		self.e_conv4 = nn.Conv2d(number_f, number_f, 3, stride=1, padding=1)
		self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, stride=1, padding=1)
		self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, stride=1, padding=1)
		self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, stride=1, padding=1)

		self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
		# self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
		# self.conv_transpose = nn.ConvTranspose2d(number_f, number_f, kernel_size=2, stride=2, padding=0)


		
	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))  # 512, 512, 3 => 512, 512, number_f 
		# p1 = self.maxpool(x1) # 512, 512, number_f => 256, 256, number_f 
		x2 = self.relu(self.e_conv2(x1)) # 256, 256, number_f => 256, 256, number_f 
		# p2 = self.maxpool(x2) # 256, 256, number_f => 128, 128, number_f 
		x3 = self.relu(self.e_conv3(x2)) # 128, 128, number_f => 128, 128, number_f 
		p3 = self.maxpool(x3) # 128, 128, number_f => 64, 64, number_f 
		x4 = self.relu(self.e_conv4(p3)) # 64, 64, number_f => 64, 64, number_f 

		# x4 = self.upsample(x4)
		# x4 = self.conv_transpose(x4)
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		# x5 = self.conv_transpose(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		# x6 = self.upsample(x6)
		# x6 = self.conv_transpose(x6)
		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		enhance_image = x + r8*(torch.pow(x,2)-x)
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		return enhance_image_1,enhance_image,r



