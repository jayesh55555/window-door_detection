import os
import shutil
from sklearn.model_selection import train_test_split

images = [f for f in os.listdir('images') if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(images)} images")

train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

os.makedirs('dataset/train/images', exist_ok=True)
os.makedirs('dataset/train/labels', exist_ok=True)
os.makedirs('dataset/val/images', exist_ok=True)
os.makedirs('dataset/val/labels', exist_ok=True)

for img in train_images:
    shutil.copy(f'images/{img}', f'dataset/train/images/{img}')
    label_file = img.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    if os.path.exists(f'labels/{label_file}'):
        shutil.copy(f'labels/{label_file}', f'dataset/train/labels/{label_file}')

for img in val_images:
    shutil.copy(f'images/{img}', f'dataset/val/images/{img}')
    label_file = img.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    if os.path.exists(f'labels/{label_file}'):
        shutil.copy(f'labels/{label_file}', f'dataset/val/labels/{label_file}')

print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
