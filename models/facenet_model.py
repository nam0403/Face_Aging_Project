import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetBackbone(nn.Module):
    def __init__(self, embedding_size=512, pretrained='vggface2'):
        super(FaceNetBackbone, self).__init__()
        # Load InceptionResnetV1
        self.net = InceptionResnetV1(pretrained=pretrained, classify=False)
        
        # FaceNet gốc ra vector 512, nhưng ta thêm lớp Linear để chắc chắn
        # map đúng kích thước embedding mong muốn (nếu bạn muốn đổi size)
        self.fc = nn.Linear(512, embedding_size)

    def forward(self, x):
        # facenet-pytorch output ra vector raw
        x = self.net(x)
        x = self.fc(x)
        return x