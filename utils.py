import torch
from pathlib import Path

def save_model(model,
               target_dir,
               model_name,LOSS,optimizer,EPOCH):
    
    target_dir=Path(target_dir)
    target_dir.mkdir(parents=True,exist_ok=True)
    model_save_path = target_dir/ model_name

    print(f"Saving model to {model_save_path}...")
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': LOSS,
        }, model_save_path)
