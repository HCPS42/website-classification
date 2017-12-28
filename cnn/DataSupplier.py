import boto3
import itertools
import json
from multiprocessing.pool import ThreadPool
import numpy as np
from operator import itemgetter
import os 
import scipy.misc

class DataSupplier(object): 
    
    def __init__(self, split=None): 
        # load AWS attributes
        s3 = boto3.resource('s3', 
            aws_access_key_id='AKIAJ3P3SB5XRB2IYKJQ', 
            aws_secret_access_key='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')
        self.download_num_threads = 20
        self.image_bucket = s3.Bucket('temirkhan-screenshots-new') 
        self.data_path = os.path.join(os.getcwd(), 'data') 
        self.pool = ThreadPool(self.download_num_threads)

        # read metadata
        try: 
            os.makedirs(self.data_path)
        except OSError: 
            pass 
        metadata_name = 'metadata.json'
        metadata_path = os.path.join(self.data_path, metadata_name)
        label_bucket = s3.Bucket('temirkhan-metadata-new') 
        label_bucket.download_file(metadata_name, metadata_path)
        with open(metadata_path) as fp: 
            self.metadata = json.load(fp) 

        # make sure they're balanced 
        malware_examples = list(filter(lambda md: md['class'] == 'malware', self.metadata))
        safe_examples = list(filter(lambda md: md['class'] == 'safe', self.metadata))
        min_examples_per_class = min(len(malware_examples), len(safe_examples))
        self.malware_examples = malware_examples[:min_examples_per_class]
        self.safe_examples = safe_examples[:min_examples_per_class]
        self.metadata = np.hstack([safe_examples[:min_examples_per_class],\
                                   malware_examples[:min_examples_per_class]])
        
        if split == None: 
            split = {'train':0.7, 'val':0.2, 'test':0.1}
        num_examples = len(self.metadata) 
        print(num_examples)
        shuffle_indices = np.random.choice(np.arange(num_examples), size=num_examples, replace=False)
        self.metadata = np.array(self.metadata)[shuffle_indices]
        splits = np.array([split['train'], split['train']+split['val']])*num_examples
        train, validation, test = np.split(self.metadata, splits.astype(np.uint32))
        self.sets = {
            'train': train, 
            'validation': validation,
            'test': test, 
        }

    def download_class_images(self, metadata, output_dir): 
        ids = map(itemgetter('id'), metadata) 
        self.__get_screenshots(ids, output_dir)



    def __initialize_classification_directory(self, classification, output_dir): 
        malware_path = os.path.join(output_dir, classification)
        if not os.path.isdir(malware_path): 
            os.makedirs(malware_path) 
        if classification == 'malware':
            metadata = self.malware_examples
        elif classification == 'safe': 
            metadata = self.safe_examples
        self.download_class_images(metadata, malware_path) 
        
    def get_all_images(self, output_dir): 
        self.__initialize_classification_directory('malware', output_dir)
        self.__initialize_classification_directory('safe', output_dir)

    def __get_screenshot_helper(self, args):
        id_, save_path = args
        image_fname = '%s.png' % id_
        image_path = os.path.join(save_path, image_fname) 
        if not os.path.isdir(image_path): 
            self.image_bucket.download_file(image_fname, image_path)
        img = scipy.misc.imread(image_path)[:,:,:3]
        return img

    def __get_screenshots(self, ids, save_path=None): 
        if not save_path: 
            save_path = self.data_path
        return self.pool.map(self.__get_screenshot_helper, zip(ids, itertools.repeat(save_path)))

    def get(self, batch_size, set_name='train'):
        batch_metadata = np.random.choice(self.sets[set_name], batch_size, replace=False)
        ids = map(itemgetter('id'), batch_metadata)
        classes = [1 if b_md['class'] == 'malware' else 0 for b_md in batch_metadata]
        imgs = self.__get_screenshots(ids)
        X = np.array(imgs).astype(np.float32)
        y = np.array(classes)
        return X, y