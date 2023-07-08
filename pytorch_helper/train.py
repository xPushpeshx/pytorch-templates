import os
import torch
import data_setup ,model_builder, engine, loss,metrics, utils
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# Setup hyperparameters
NUM_WORKERS=os.cpu_count()
NUM_EPOCHS = 50
BATCH_SIZE = 2
HIDDEN_UNITS = 10
LEARNING_RATE = 0.0001
IMG_SIZE=128

train_dir=''
test_dir=''

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = A.Compose([
    A.HorizontalFlip(),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
    ], p=0.3),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    A.ShiftScaleRotate(),
    A.Resize(IMG_SIZE,IMG_SIZE,always_apply=True),
    ToTensorV2()
])

train_dataloader,test_dataloader,class_names=data_setup.create_dataloader_for_mask(train_dir,
                                                                      test_dir,
                                                                      transform,
                                                                      BATCH_SIZE,
                                                                      NUM_WORKERS)


#train_dl,test_dl=data_setup.create_dataloader_images( X_train,
#                                                y_train,
#                                                X_val,
#                                                y_val,
#                                                train_transform,
#                                               test_transform,
#                                               BATCH_SIZE,
#                                                NUM_WORKERS)

model=model_builder ## get model.to(device)

loss_fn=loss ## get loss function
optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

engine.train(model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model,
                 target_dir="models",
                 model_name="test.pth")