import os
import torch
import data_setup ,model_builder, engine, loss,metrics, utils

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.0001
SHAPE=(224,224)

train_dir=''
test_dir=''

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform=transforms.Compose([
    transforms.Resize(SHAPE),
    transforms.ToTensor(),
])

train_dataloader,testdataloader,class_names=data_setup.get_dataloader(train_dir,
                                                                      test_dir,
                                                                      data_transform,
                                                                      BATCH_SIZE)

model=model_builder ## get model.to(device)

loss_fn=loss ## get loss function
optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

engine.train(model,
             train_dataloader=train_dataloader,
             test_dataloader=testdataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model,
                 target_dir="models",
                 model_name="test.pth")