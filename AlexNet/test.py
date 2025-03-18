import torch
import torch.nn as nn
from model import Model
from preprocessing import LoadDataset
from colorama import Fore, Style, init
from tqdm import tqdm

init(autoreset=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestModel(nn.Module):
    def __init__(self, modeL_path, data_path):
        super(TestModel, self).__init__()
        self.momentum = 0.9
        self.batch_size = 64
        self.learning_rate = 0.1
        self.weight_decay = 0.0005
        self.batch_size = 64
        self.model_path = modeL_path
        self.data_path = data_path
        self.model_name = "AlexNet"
        self.model = Model(num_classes=10)
        self.loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.load_model()
        self.load_test_data()

    def load_model(self):
        checkpoint = torch.load(self.model_path, weights_only=True)
        self.classes = checkpoint['classes']

        self.model.load_state_dict(
            checkpoint['model_state_dict']
        )

        self.model.to(device)


    def load_test_data(self):
        test_data = LoadDataset(self.data_path, batch_size=self.batch_size, is_test=True, num_classes=10, test_classes=self.classes)
        self.test_loader = test_data.augment_dataset()


    def test_model(self):
        self.model.eval()
        current_loss = 0.0
        current_accuracy = 0.0

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.test_loader), desc='Testing'):
                images, labels = self.to_device(batch)
                pred = self.model(images)
                loss = self.loss(pred, labels)

                current_loss += loss.item()
                current_accuracy += (pred.argmax(1) == labels).float().mean().item()

            current_loss /= len(self.test_loader)
            current_accuracy /= len(self.test_loader)

            print(f"{Fore.LIGHTMAGENTA_EX} The current validation loss is ==> {current_loss:.3f}")
            print(f"{Fore.LIGHTYELLOW_EX} The current validation accuracy is ==> {current_accuracy:.3f}")
            

            return current_loss, current_accuracy
        

if __name__ == "__main__":
    model_path = "AlexNet/final_model.pth"
    data_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals"
    model = TestModel(model_path, data_path)