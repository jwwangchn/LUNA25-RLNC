import torch
import io
import uuid
from getpass import getpass

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

def modify_and_encrypt_model(input_path, output_path, password):
    print(f"load params from: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu')

    checkpoint = {
        'model_state_dict': state_dict,
        'uuid': str(uuid.uuid4()),
    }

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    plaintext = buffer.read()

    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32, count=1000000) # 派生一个256位的密钥

    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    with open(output_path, 'wb') as f:
        f.write(salt)
        f.write(cipher.nonce)
        f.write(tag)
        f.write(ciphertext)


if __name__ == '__main__':
    # set password
    password = "luna25-12345"
    
    # full dataset model
    k_fold1_model_dir = "results/LUNA25-RLNC-epoch50-3D-epoch50-k_fold1"
    input_model_path = f'{k_fold1_model_dir}/best_metric_model.pth'
    encrypted_output_path = f'{k_fold1_model_dir}/best_metric_model.bin'
    
    modify_and_encrypt_model(input_model_path, encrypted_output_path, password)
    
    # k_fold dataset models
    k_fold5_model_dir = "results/LUNA25-RLNC-epoch50-3D-epoch50-k_fold5"
    epoch_name = "49_ema"
    fold_name = 5
    for fold_idx in range(fold_name):
        if "best" in epoch_name:
            model_name = f"fold_{fold_idx}_{epoch_name}"
        else:
            model_name = f"fold_{fold_idx}_epoch{epoch_name}"

        input_model_path = f'{k_fold5_model_dir}/{model_name}.pth'
        encrypted_output_path = f'{k_fold5_model_dir}/{model_name}.bin'
        
        modify_and_encrypt_model(input_model_path, encrypted_output_path, password)