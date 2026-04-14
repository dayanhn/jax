import tensorflow_datasets as tfds
import numpy as np
import os


def download_mnist_datasets(data_dir='./mnist_datasets'):
    """手动下载 MNIST 数据集"""
    import urllib.request
    import gzip
    
    os.makedirs(data_dir, exist_ok=True)
    
    # 使用多个镜像源,提高下载成功率
    mirror_urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",  # AWS S3 镜像
        "https://storage.googleapis.com/cvdf-datasets/mnist/",  # Google Cloud 镜像
        "http://yann.lecun.com/exdb/mnist/",  # 原始地址(备用)
    ]
    
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz", 
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            downloaded = False
            
            # 尝试从不同的镜像源下载
            for base_url in mirror_urls:
                try:
                    url = base_url + filename
                    print(f"  Trying: {url}")
                    urllib.request.urlretrieve(url, filepath)
                    print(f"✓ Downloaded {filename} from {base_url}")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  ✗ Failed from {base_url}: {e}")
                    continue
            
            if not downloaded:
                print(f"✗ Failed to download {filename} from all mirrors")
                print("请检查网络连接或尝试其他镜像源")
                raise Exception(f"无法从任何镜像源下载 {filename}")
        else:
            print(f"✓ {filename} already exists")


def load_mnist_from_files(data_dir='./mnist_datasets', kind='train'):
    """从手动下载的文件中加载 MNIST 数据"""
    import gzip
    
    labels_path = os.path.join(data_dir, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(data_dir, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    return images, labels


def get_datasets(batch_size=128, use_manual_download=False):
    """Return train and test iterators yielding (images, labels) as NumPy arrays.

    Images are normalized to float32 in [0,1] and flattened to shape (B, 784).
    
    Args:
        batch_size: 批次大小
        use_manual_download: 是否使用手动下载方式
    """
    
    if use_manual_download:
        # 使用手动下载方式
        data_dir = '/data3/zhongzhw/code/google/jax/jax_test/mnist/mnist_datasets'
        
        # 检查文件是否存在,不存在则下载
        required_files = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
        
        if not all(os.path.exists(os.path.join(data_dir, f)) for f in required_files):
            print("MNIST 数据文件不存在,开始手动下载...")
            download_mnist_datasets(data_dir)
        
        # 加载数据
        print("Loading MNIST data from manual download...")
        train_images, train_labels = load_mnist_from_files(data_dir, 'train')
        test_images, test_labels = load_mnist_from_files(data_dir, 't10k')
        
        # 转换为 float32 并归一化
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
        
        print(f"Train set: {train_images.shape[0]} samples")
        print(f"Test set: {test_images.shape[0]} samples")
        
        def train_iter():
            num_samples = len(train_images)
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                yield train_images[batch_idx], train_labels[batch_idx]
        
        def test_iter():
            num_samples = len(test_images)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                yield test_images[start:end], test_labels[start:end]
        
        return train_iter, test_iter
    
    else:
        # 使用 tensorflow_datasets 方式
        print("Using tensorflow_datasets to load MNIST...")
        print("(首次运行会下载数据,请耐心等待...)")
        
        # 设置数据下载路径和使用镜像加速
        os.environ['TFDS_DATA_DIR'] = './tensorflow_datasets'
        
        # 尝试加载数据集,如果不存在则下载
        try:
            ds_train = tfds.load('mnist', split='train', as_supervised=True, 
                                download=True, data_dir='./tensorflow_datasets')
            ds_test = tfds.load('mnist', split='test', as_supervised=True,
                               download=True, data_dir='./tensorflow_datasets')
        except Exception as e:
            print(f"加载数据集时出错: {e}")
            print("请确保已正确安装 tensorflow_datasets")
            raise
        
        ds_train = ds_train.shuffle(10000).batch(batch_size).prefetch(1)
        ds_test = ds_test.batch(batch_size).prefetch(1)

        train_np = tfds.as_numpy(ds_train)
        test_np = tfds.as_numpy(ds_test)

        def train_iter():
            for imgs, labels in train_np:
                imgs = imgs.astype(np.float32) / 255.0
                imgs = imgs.reshape(imgs.shape[0], -1)
                yield imgs, labels

        def test_iter():
            for imgs, labels in test_np:
                imgs = imgs.astype(np.float32) / 255.0
                imgs = imgs.reshape(imgs.shape[0], -1)
                yield imgs, labels

        return train_iter, test_iter
