import torch
from dark_channel import*

im1 = torch.randn(1, 3, 4, 5).cuda()
dk1 = torch.zeros(1, 1, 4, 5).cuda()
id1 = torch.zeros(1, 3, 4, 5).cuda()
dark_channel(im1, dk1, id1, 3)

dk2 = torch.zeros(1, 1, 4, 5).cuda()
dark_extract(im1, id1, dk2)

print(dk1)
print(dk2)

pb1 = torch.zeros(1, 3, 4, 5).cuda()
ac1 = torch.zeros(1, 3, 4, 5).cuda()
place_back(dk1, id1, pb1, ac1, 3)
print(pb1)
print(ac1)
print(torch.sum(ac1))
