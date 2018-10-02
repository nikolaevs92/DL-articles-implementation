import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np


DATA_MODES = ['train', 'val']
RESCALE_SIZE = (224, 224)


class SimpsonDataset(Dataset):

    def __init__(self, files, mode, rescale_size=RESCALE_SIZE):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        self.rescale_size = rescale_size
        
        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError
        
        self.samples = []
        self.len_ = len(self.files)
        self.label_encoder = LabelEncoder()
        
        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)
            tqdm.write(f"\nUsing label encoder with {len(self.label_encoder.classes_)} labels")
            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)
        
        with tqdm(desc=f"Loading {self.mode} files", total=self.len_) as pbar:
            p = ThreadPool()
            for sample in p.imap(self._load_sample, self.files):
                self.samples.append(sample)
                pbar.update()
        
    
    def __len__(self):
        return self.len_
    
    
    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = self.samples[index]
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
        return x, y
    
    
    def _load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image
    
    
    def _prepare_sample(self, image):
        image = image.resize(self.rescale_size)
        return np.array(image)