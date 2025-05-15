import os
import pandas as pd

def load_reviews_from_folder(folder_path, label):
    reviews = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), encoding='utf-8') as f:
                reviews.append((f.read().strip(), label))
    return reviews

def create_imdb_csv(data_dir, output_csv):
    train_pos = load_reviews_from_folder(os.path.join(data_dir, 'train/pos'), 'positive')
    train_neg = load_reviews_from_folder(os.path.join(data_dir, 'train/neg'), 'negative')
    test_pos = load_reviews_from_folder(os.path.join(data_dir, 'test/pos'), 'positive')
    test_neg = load_reviews_from_folder(os.path.join(data_dir, 'test/neg'), 'negative')

    all_reviews = train_pos + train_neg + test_pos + test_neg
    df = pd.DataFrame(all_reviews, columns=['review', 'sentiment'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')

# 使用示例
create_imdb_csv('aclImdb_v1/aclImdb', 'data/imdb.csv')
