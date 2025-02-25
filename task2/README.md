The task involved finding or collecting an animal classification/detection dataset that contains at least 10 classes of animals. A dataset from Kaggle was found.
After reviewing the dataset, the image sizes were adjusted while maintaining their aspect ratio. Additionally, the dataset was split into training, validation, and test sets. A CNN model for classification was built and trained. To optimize resource usage, the image sizes were further reduced during training.
This project includes:
- exploratory.ipynb - a notebook for exploring the dataset, images, and their distribution. It was discovered that there is class imbalance.
- train.ipynb - a notebook that contains the CNN classification model and the training results.