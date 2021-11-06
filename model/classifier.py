from torch import nn

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()

        self.drop1 = nn.Dropout(drop)
        self.lin1 = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(drop)
        self.lin2 = nn.Linear(mid_features, out_features)

    def forward(self, x):
        x = self.lin1(self.drop1(x))
        x = self.relu(x)
        x = self.lin2(self.drop2(x))

        return x